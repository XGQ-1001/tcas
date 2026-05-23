"""Super-DAG 构造：将单任务 DAG 复制 K 份，组成一个大的调度图。

============================================================================
核心思想：
  GNN 已经学会调度 1 个 batch=1 的 DAG。现在 K 个并发推理任务 = 把 K 个
  独立的 DAG 拼在一起丢给同一个 GNN →  GNN 只看到一个更大的图，然后
  输出一个全局最优的调度顺序。

  这样就把"多任务调度"问题彻底转化成了"更大 DAG 的算子调度"问题，
  可以直接复用现有的 GraphState / SchedulingEnv / DynamicActorCritic /
  PPO 训练代码，零改动！

关键实现要点：
  1. 复制 K 份节点，id 平移 K 次，边关系也随之平移
  2. K 份子图彼此之间没有依赖边（任务独立）
  3. movable_mask / is_placeholder / is_output 等都是 K 倍拼接
  4. GraphState.x (静态特征) 保持 K 份副本 → GNN 看到相同的节点特征
     但不同的拓扑位置 (ready 状态会不同)

不做的事：
  - 不改 FX 计算图 (保持 gnn-strategy 的清晰边界)
  - 不做真实 GPU 捕获 (super-DAG 的真实捕获需要 K 份模型副本，代码复杂度
    太高；我们用「模拟 makespan」作为训练和评估的指标)
  - 评估 vs. PyTorch 默认/Opara 时，真实 GPU 基线用「K 次串行调用」等价模拟
============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

import os
import sys

# 允许 import gnn_strategy
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GNN_STRATEGY_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, '..', '..', 'gnn-strategy')
)
_REPO_ROOT = os.path.dirname(_GNN_STRATEGY_DIR)
for _p in (_GNN_STRATEGY_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gnn_strategy.graph_state import GraphState, D_STATIC


@dataclass
class SuperDAGInfo:
    """Super-DAG 的元信息，用于追溯每个节点属于哪个任务。

    node_task_id[v] = 该节点属于第几个任务副本 (0..K-1)
    node_base_id[v] = 该节点在 base DAG 中的原始 id
    base_n         = base DAG 的节点数
    num_tasks      = K，任务副本数
    """
    node_task_id: torch.Tensor   # [K*N] long, 0..K-1
    node_base_id: torch.Tensor   # [K*N] long, 0..N-1
    base_n: int
    num_tasks: int


def build_super_dag(
    base_gs: GraphState,
    K: int,
) -> tuple[GraphState, SuperDAGInfo]:
    """将 base GraphState 复制 K 份，拼成 super-DAG。

    参数:
        base_gs: 单任务 GraphState（由 build_graph_state 得到）
        K:       任务副本数（并发推理任务数）

    返回:
        super_gs:  K 份拼接后的 GraphState，节点数 = K * base_gs.N
        info:      SuperDAGInfo 元信息

    设计约定：
      - 新节点 id = base_id + task_id * N
      - 新节点名 = "t{task_id}_{base_name}"
      - 新节点特征 = 原特征直接复制（K 份完全相同）
      - 父/子关系 = 仅在同一任务副本内部的平移版本，跨任务无依赖边
    """
    assert K >= 1
    N = len(base_gs.node_names)
    KN = K * N

    # ------------------------------------------------------------------
    # 1. 节点名 / op_kinds：K 份拼接
    # ------------------------------------------------------------------
    new_names: List[str] = []
    new_op_kinds: List[str] = []
    for k in range(K):
        for i in range(N):
            new_names.append(f't{k}_{base_gs.node_names[i]}')
            new_op_kinds.append(base_gs.op_kinds[i])

    # ------------------------------------------------------------------
    # 2. 邻接表：在每份副本内部平移 id = base_id + k*N
    # ------------------------------------------------------------------
    new_parents: List[List[int]] = [[] for _ in range(KN)]
    new_children: List[List[int]] = [[] for _ in range(KN)]
    for k in range(K):
        offset = k * N
        for i in range(N):
            new_parents[i + offset] = [p + offset for p in base_gs.parents[i]]
            new_children[i + offset] = [c + offset for c in base_gs.children[i]]

    # ------------------------------------------------------------------
    # 3. 静态特征矩阵：x 直接 repeat K 份
    # ------------------------------------------------------------------
    new_x = base_gs.x.repeat(K, 1)  # [K*N, D_STATIC]

    new_durations = base_gs.durations.repeat(K)
    new_is_placeholder = base_gs.is_placeholder.repeat(K)
    new_is_output = base_gs.is_output.repeat(K)
    new_movable_mask = base_gs.movable_mask.repeat(K)
    new_is_mem_bound = base_gs.is_mem_bound.repeat(K)
    new_descendant_work = base_gs.descendant_work.repeat(K)
    new_slack = base_gs.slack.repeat(K)

    super_gs = GraphState(
        node_names=new_names,
        op_kinds=new_op_kinds,
        parents=new_parents,
        children=new_children,
        x=new_x,
        durations=new_durations,
        is_placeholder=new_is_placeholder,
        is_output=new_is_output,
        movable_mask=new_movable_mask,
        is_mem_bound=new_is_mem_bound,
        descendant_work=new_descendant_work,
        slack=new_slack,
    )

    node_task_id = torch.zeros(KN, dtype=torch.long)
    node_base_id = torch.zeros(KN, dtype=torch.long)
    for k in range(K):
        offset = k * N
        node_task_id[offset:offset + N] = k
        node_base_id[offset:offset + N] = torch.arange(N, dtype=torch.long)

    info = SuperDAGInfo(
        node_task_id=node_task_id,
        node_base_id=node_base_id,
        base_n=N,
        num_tasks=K,
    )

    return super_gs, info


def super_graph_stats(super_gs: GraphState, info: SuperDAGInfo) -> dict:
    """返回 super-DAG 的统计信息，用于日志 / 调试。"""
    n_movable = int(super_gs.movable_mask.sum().item())
    return {
        'num_tasks': info.num_tasks,
        'base_nodes': info.base_n,
        'total_nodes': len(super_gs.node_names),
        'total_movable': n_movable,
        'base_movable': n_movable // info.num_tasks,
    }
