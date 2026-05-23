"""非学习型调度基线 + 统一的模拟 makespan 评估接口。

============================================================================
在 super-DAG 上，我们需要几种"非学习"基线来对比 GNN 策略：

  1. topological_order       朴素拓扑排序 (每个任务内部先算完再算下一个)
  2. round_robin_order       轮流从每个任务拿一个 ready 节点 (任务交织)
  3. opara_like_order        仿 Opara 风格的贪心: 优先选关键路径/长后代
  4. random_order            随机拓扑有效顺序 (下限对照)

所有基线都只输出一个「调度顺序」(node id list)，然后用同一个
SchedulingEnv 模拟出 makespan，保证评估公平 (apples-to-apples)。

这样的评估协议对应"PyTorch 默认 / Opara 串行 / Opara 合并 / GNN"四种
方案在「理论调度最优性」上的比较，与真实 GPU benchmark 互为补充。
============================================================================
"""

from __future__ import annotations

import random
from typing import List, Optional

import torch

from gnn_strategy.env import SchedulingEnv
from gnn_strategy.graph_state import GraphState


# --------------------------------------------------------------------------
# 统一的模拟 makespan 评估：所有算法都走同一个 SchedulingEnv，保证公平
# --------------------------------------------------------------------------

def simulate_makespan(
    gs: GraphState,
    order: List[int],
    n_streams: int = 8,
    device: Optional[torch.device] = None,
) -> float:
    """给定一个节点顺序，在 SchedulingEnv 中按序调度，返回 makespan。

    注意：
      - order 中只需包含 movable 节点即可，placeholder 由 env.reset() 自动处理
      - 若顺序不是拓扑有效的，会跳过当前不 ready 的节点，留到后面
      - 这与我们训练时策略网络的行为一致（只从 ready 集合中选）
    """
    env = SchedulingEnv(gs, n_streams=n_streams, device=device or torch.device('cpu'))
    env.reset()

    pending = list(order)
    while not env.is_done():
        ready = env.ready_mask()
        ready_ids = [i for i in range(len(ready)) if ready[i].item() == 1.0]
        if not ready_ids:
            break

        chosen = None
        for idx, v in enumerate(pending):
            if v in ready_ids:
                chosen = v
                pending.pop(idx)
                break

        if chosen is None:
            # order 已枯竭或错位，退化为 FIFO
            chosen = ready_ids[0]

        env.step(chosen)

    return float(env.current_makespan())


# --------------------------------------------------------------------------
# 基线 1: 朴素拓扑排序 (Kahn's algorithm, 按 task-major 输出)
# --------------------------------------------------------------------------

def topological_order(gs: GraphState) -> List[int]:
    """朴素拓扑排序。在 super-DAG 上，由于任务副本之间无依赖，
    最终结果会按「任务 0 先做完，再做任务 1 ...」的顺序（task-major），
    不利用并行任务交织。这是"PyTorch 默认串行"的近似。
    """
    N = len(gs.node_names)
    in_deg = [len(gs.parents[i]) for i in range(N)]

    queue: List[int] = []
    for i in range(N):
        if in_deg[i] == 0 and gs.movable_mask[i].item() == 1.0:
            queue.append(i)

    order: List[int] = []
    visited = [False] * N
    while queue:
        v = queue.pop(0)
        if visited[v]:
            continue
        visited[v] = True
        order.append(v)
        for c in gs.children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0 and gs.movable_mask[c].item() == 1.0 and not visited[c]:
                queue.append(c)

    return order


# --------------------------------------------------------------------------
# 基线 2: Round-robin (严格任务交织)
# --------------------------------------------------------------------------

def round_robin_order(gs: GraphState, num_tasks: int) -> List[int]:
    """轮流从每个任务的 ready 集合中挑一个节点。

    模拟「多 stream，任务 0 发射一个 op，任务 1 发射一个 op ...」的行为。
    这在 GPU 有足够并行空间时是一个很强的基线（手工任务交织）。

    参数:
        gs:        super-DAG GraphState
        num_tasks: 任务数 K（用于识别每个节点属于哪个任务）
    """
    N = len(gs.node_names)
    base_n = N // num_tasks
    in_deg = [len(gs.parents[i]) for i in range(N)]
    done = [False] * N
    order: List[int] = []

    while True:
        progressed = False
        for k in range(num_tasks):
            offset = k * base_n
            # 在任务 k 的节点中找一个 ready 的
            for i in range(base_n):
                v = offset + i
                if done[v] or gs.movable_mask[v].item() != 1.0:
                    continue
                if in_deg[v] == 0:
                    order.append(v)
                    done[v] = True
                    for c in gs.children[v]:
                        in_deg[c] -= 1
                    progressed = True
                    break
        if not progressed:
            break

    return order


# --------------------------------------------------------------------------
# 基线 3: Opara-style 贪心 (按后代工作量降序从 ready 集合中选)
# --------------------------------------------------------------------------

def opara_like_order(gs: GraphState) -> List[int]:
    """仿 Opara 的贪心调度：每一步从 ready 集合中选择
    "后代工作量最大" 的节点（让关键路径上的算子尽早开始）。

    这与 Opara 在单任务上的行为一致；应用到 super-DAG 时会自动产生
    合理的跨任务交织（因为两个任务的关键路径起点会交替成为 top-1）。
    """
    N = len(gs.node_names)
    in_deg = [len(gs.parents[i]) for i in range(N)]
    done = [False] * N
    order: List[int] = []

    desc_work = gs.descendant_work.tolist()

    while True:
        ready = [
            i for i in range(N)
            if not done[i] and in_deg[i] == 0 and gs.movable_mask[i].item() == 1.0
        ]
        if not ready:
            break

        # 按 descendant_work 降序，tie-break 按 duration 降序
        ready.sort(key=lambda v: (-desc_work[v], -gs.durations[v].item()))
        v = ready[0]
        order.append(v)
        done[v] = True
        for c in gs.children[v]:
            in_deg[c] -= 1

    return order


# --------------------------------------------------------------------------
# 基线 4: 随机顺序 (下限对照)
# --------------------------------------------------------------------------

def random_order(gs: GraphState, seed: Optional[int] = None) -> List[int]:
    """随机拓扑有效顺序：每一步从 ready 集合中均匀随机选一个。"""
    rng = random.Random(seed)
    N = len(gs.node_names)
    in_deg = [len(gs.parents[i]) for i in range(N)]
    done = [False] * N
    order: List[int] = []

    while True:
        ready = [
            i for i in range(N)
            if not done[i] and in_deg[i] == 0 and gs.movable_mask[i].item() == 1.0
        ]
        if not ready:
            break
        v = rng.choice(ready)
        order.append(v)
        done[v] = True
        for c in gs.children[v]:
            in_deg[c] -= 1

    return order


# --------------------------------------------------------------------------
# 便捷接口：一次评估所有基线
# --------------------------------------------------------------------------

def evaluate_all_baselines(
    super_gs: GraphState,
    num_tasks: int,
    n_streams: int = 8,
    random_seed: int = 0,
    random_trials: int = 5,
) -> dict:
    """一次评估所有基线的 makespan，便于和 GNN 策略对比。"""
    results = {}

    topo = topological_order(super_gs)
    results['Topological'] = simulate_makespan(super_gs, topo, n_streams=n_streams)

    rr = round_robin_order(super_gs, num_tasks=num_tasks)
    results['RoundRobin'] = simulate_makespan(super_gs, rr, n_streams=n_streams)

    op = opara_like_order(super_gs)
    results['Opara-like'] = simulate_makespan(super_gs, op, n_streams=n_streams)

    # 随机基线取多次平均
    rand_vals = []
    for t in range(random_trials):
        ro = random_order(super_gs, seed=random_seed + t)
        rand_vals.append(simulate_makespan(super_gs, ro, n_streams=n_streams))
    results['Random(mean)'] = float(sum(rand_vals) / len(rand_vals))
    results['Random(min)'] = float(min(rand_vals))

    return results
