"""静态图描述 + 丰富的逐节点特征。

============================================================================
核心作用：
  将 PyTorch FX 计算图转换为 GNN 可用的图结构 + 特征矩阵。
  这一步在训练开始前只做一次（对每个模型），之后反复使用。

  输入：FX 计算图 (torch.fx.Graph) + 算子性能分析数据 (node_profiles)
  输出：GraphState 对象，包含：
    - 图拓扑：parents[v], children[v]（DAG 的邻接表）
    - 静态特征矩阵 x: [N, 16]（每个节点 16 维特征，归一化到 [0,1]）
    - 各种辅助张量（duration, slack, 可调度掩码等）

静态特征 (D_STATIC = 16) 的含义：
  维度  名称                      含义
  ──────────────────────────────────────────────────
   0    is_mem_intensive          是否是访存密集型（0/1）
   1    is_compute_intensive      是否是计算密集型（0/1）
   2    shared_mem_norm           共享内存使用量（归一化）
   3    regs_norm                 寄存器使用量（归一化）
   4    threads_norm              线程数（归一化）
   5    occupancy_norm            SM 占用率（归一化）
   6    is_critical_path          是否在关键路径上（0/1）
   7    duration_norm             执行时间（归一化）
   8    num_predecessors_norm     前驱节点数（归一化）
   9    num_successors_norm       后继节点数（归一化）
  10    cp_depth_forward_norm     从根到该节点的最长路径（归一化）
  11    cp_depth_backward_norm    从该节点到叶的最长路径（归一化）
  12    descendant_work_norm      所有后代的总工作量（归一化）
  13    ancestor_work_norm        所有祖先的总工作量（归一化）
  14    slack_norm                松弛时间（归一化），= 最迟开始 - 最早开始
  15    fan_ratio                 后继数 / (前驱数 + 后继数 + 1)

  其中 6-14 来自 CPM（关键路径法）分析，帮助 GNN 理解 DAG 结构。
============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from .utils import (
    compute_ancestor_work,
    compute_critical_path,
    compute_descendant_work,
    compute_longest_path_from_root,
    compute_longest_path_to_leaf,
    duration_from_profile,
    extract_resource_vector,
    is_mem_intensive,
    normalize_by_max,
    safe_div,
    topo_sort,
)

D_STATIC = 16  # 静态特征维度（GNN 输入维度）


@dataclass
class GraphState:
    """静态图描述 + 逐节点特征。

    动态调度状态（哪些节点已调度、流的繁忙程度等）不在这里，
    而是在 env.py 的 SchedulingEnv 中维护。

    属性:
        node_names:     每个节点的名称（如 'conv2d', 'relu_1' 等）
        op_kinds:       每个节点的操作类型（'placeholder', 'call_function', 'output' 等）
        parents[v]:     节点 v 的所有前驱节点 id 列表（依赖关系）
        children[v]:    节点 v 的所有后继节点 id 列表
        x:              [N, D_STATIC=16] 静态特征矩阵（GNN 的输入）
        durations:      [N] 每个节点的执行时间（微秒），来自 profiling
        movable_mask:   [N] 可调度掩码，1.0 = 可以被 RL 调度的算子
                        （排除 placeholder 和 output）
        is_mem_bound:   [N] 是否是访存密集型，用于资源竞争分析
        descendant_work:[N] 后代总工作量（用于环境中计算紧迫性）
        slack:          [N] 松弛时间（用于环境中计算调度灵活度）
    """

    node_names: List[str]
    op_kinds: List[str]
    parents: List[List[int]]
    children: List[List[int]]

    x: torch.Tensor           # [N, D_STATIC]
    durations: torch.Tensor    # [N] 原始执行时间（微秒）

    is_placeholder: torch.Tensor
    is_output: torch.Tensor
    movable_mask: torch.Tensor  # float [N], 1.0 表示可调度
    is_mem_bound: torch.Tensor  # float [N], 1.0 表示访存密集

    descendant_work: torch.Tensor  # [N] 原始后代工作量
    slack: torch.Tensor            # [N] 原始松弛时间


def build_graph_state(
    fx_graph,
    node_profiles: Optional[Dict] = None,
    device_props: Optional[Dict] = None,
) -> GraphState:
    """从 FX 计算图构建 GraphState。

    这是整个流程的第一步：把 PyTorch 模型的计算图变成 GNN 可以理解的格式。

    参数:
        fx_graph:      torch.fx.Graph，由 torch._dynamo.explain 得到
        node_profiles: {节点名: profiling信息}，由 OperatorLauncher.recompile 得到
        device_props:  GPU 设备属性（共享内存大小、寄存器数等）

    返回:
        GraphState 对象
    """
    node_profiles = node_profiles or {}

    # ---- 构建图拓扑 ----
    nodes = list(fx_graph.nodes)
    node_to_id = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    node_names = [n.name for n in nodes]
    op_kinds = [getattr(n, 'op', '') for n in nodes]

    # 构建邻接表（DAG 的边）
    parents: List[List[int]] = [[] for _ in range(N)]
    children: List[List[int]] = [[] for _ in range(N)]
    for v in nodes:
        v_id = node_to_id[v]
        for u in v.all_input_nodes:   # FX 节点的输入 = DAG 中的父节点
            u_id = node_to_id[u]
            parents[v_id].append(u_id)
            children[u_id].append(v_id)

    # ---- 提取每个节点的执行时间（来自 GPU profiling）----
    durations_raw = [duration_from_profile(node_profiles.get(n.name)) for n in nodes]

    # ---- 关键路径法（CPM）分析 ----
    # est=最早开始时间, eft=最早结束时间, lst=最迟开始时间, lft=最迟结束时间
    est, eft, lst, lft, makespan = compute_critical_path(N, parents, children, durations_raw)
    slack_raw = [lst[i] - est[i] for i in range(N)]        # 松弛时间 = 最迟开始 - 最早开始
    is_crit = [1.0 if abs(slack_raw[i]) < 1e-6 else 0.0 for i in range(N)]  # 松弛=0 → 关键路径

    # ---- 后代/祖先工作量分析 ----
    desc_work = compute_descendant_work(N, children, durations_raw)   # 后代总耗时
    anc_work = compute_ancestor_work(N, parents, children, durations_raw)  # 祖先总耗时

    # ---- 最长路径深度（正向/反向）----
    cp_fwd = compute_longest_path_from_root(N, parents, children, durations_raw)  # 从根到节点
    cp_bwd = compute_longest_path_to_leaf(N, parents, children, durations_raw)    # 从节点到叶

    # ---- 归一化到 [0, 1] ----
    dur_norm = normalize_by_max(durations_raw)
    npred = [float(len(parents[i])) for i in range(N)]
    nsucc = [float(len(children[i])) for i in range(N)]
    npred_norm = normalize_by_max(npred)
    nsucc_norm = normalize_by_max(nsucc)
    cp_fwd_norm = normalize_by_max(cp_fwd)
    cp_bwd_norm = normalize_by_max(cp_bwd)
    desc_norm = normalize_by_max(desc_work)
    anc_norm = normalize_by_max(anc_work)
    slack_norm = normalize_by_max([max(0.0, s) for s in slack_raw])

    # ---- 构建 16 维特征矩阵 [N, D_STATIC=16] ----
    feats: List[List[float]] = []
    mem_bound_list: List[float] = []
    for i, n in enumerate(nodes):
        name = n.name
        mem = 1.0 if is_mem_intensive(name) else 0.0
        compute = 1.0 - mem
        mem_bound_list.append(mem)

        # 从 profiling 数据提取 GPU 资源使用（归一化）
        shared_mem, regs, threads, occ = extract_resource_vector(
            node_profiles.get(name), device_props=device_props,
        )

        # fan_ratio: 衡量节点是"汇聚型"还是"扩散型"
        fan = safe_div(nsucc[i], npred[i] + nsucc[i] + 1.0)

        feats.append([
            mem,                   #  0: 是否访存密集
            compute,               #  1: 是否计算密集
            shared_mem,            #  2: 共享内存使用（归一化）
            regs,                  #  3: 寄存器使用（归一化）
            threads,               #  4: 线程数（归一化）
            occ,                   #  5: SM 占用率（归一化）
            is_crit[i],            #  6: 是否在关键路径上
            dur_norm[i],           #  7: 执行时间（归一化）
            npred_norm[i],         #  8: 前驱数（归一化）
            nsucc_norm[i],         #  9: 后继数（归一化）
            cp_fwd_norm[i],        # 10: 正向关键路径深度（归一化）
            cp_bwd_norm[i],        # 11: 反向关键路径深度（归一化）
            desc_norm[i],          # 12: 后代工作量（归一化）
            anc_norm[i],           # 13: 祖先工作量（归一化）
            slack_norm[i],         # 14: 松弛时间（归一化）
            fan,                   # 15: 扩散比
        ])

    x = torch.tensor(feats, dtype=torch.float32)

    # ---- 构建掩码 ----
    # placeholder = 输入张量, output = 输出节点, 这两类不可调度
    is_placeholder = torch.tensor(
        [1.0 if op == 'placeholder' else 0.0 for op in op_kinds], dtype=torch.float32,
    )
    is_output = torch.tensor(
        [1.0 if op == 'output' else 0.0 for op in op_kinds], dtype=torch.float32,
    )
    movable_mask = ((is_placeholder == 0.0) & (is_output == 0.0)).float()

    return GraphState(
        node_names=node_names,
        op_kinds=op_kinds,
        parents=parents,
        children=children,
        x=x,
        durations=torch.tensor(durations_raw, dtype=torch.float32),
        is_placeholder=is_placeholder,
        is_output=is_output,
        movable_mask=movable_mask,
        is_mem_bound=torch.tensor(mem_bound_list, dtype=torch.float32),
        descendant_work=torch.tensor(desc_work, dtype=torch.float32),
        slack=torch.tensor(slack_raw, dtype=torch.float32),
    )
