"""事件驱动的调度模拟环境 — 强化学习的「世界」。

============================================================================
核心作用：
  这是 RL 智能体与之交互的环境。每一步（step），智能体从 ready 集合中
  选择一个算子节点，环境执行调度并返回动态特征。

  在 real-latency 训练模式下，环境只负责提供「动态特征」给策略网络，
  reward 来自真实 GPU 的 benchmark 结果（而不是环境计算的模拟 reward）。

工作流程（每个 episode = 调度一个完整的 DAG）：
  1. reset()  → 初始化状态，placeholder 节点自动调度
  2. 循环直到所有可调度节点完成：
     a. 策略网络读取 dynamic_node_features() + global_features() + ready_mask()
     b. 策略网络选择一个 action（要调度的节点 id）
     c. env.step(action) → 模拟该节点的流分配和执行
  3. scheduled_order() → 返回完整调度顺序

流分配策略（模拟 Opara 的贪心分配）：
  当节点 v 被调度时：
  - 计算最早可开始时间 = max(所有父节点完成时间)
  - 在 n_streams 条流中，选择 busy_until 最早的流
  - 开始时间 = max(流空闲时间, 依赖完成时间)
  - 结束时间 = 开始时间 + 节点执行时间

动态节点特征 (D_DYN = 10)：
  维度  名称                      含义（每一步都会更新）
  ──────────────────────────────────────────────────
   0    is_done                   该节点是否已调度完成
   1    is_running                该节点是否正在执行（已调度但未完成）
   2    is_ready                  该节点是否就绪（所有父节点已完成，等待调度）
   3    remaining_parent_ratio    未完成的父节点占比
   4    time_since_ready_norm     该节点等待了多久（归一化）
   5    remaining_slack_norm      剩余松弛时间（归一化）
   6    unfinished_desc_ratio     未完成的后代工作占比
   7    ready_competition_norm    当前 ready 集合大小（归一化）
   8    same_type_competition     同类型（访存/计算）的竞争比例
   9    urgency                   紧迫性 = 反向关键路径 / 剩余总工作量

全局特征 (D_GLOBAL = 12)：
  维度  名称                      含义
  ──────────────────────────────────────────────────
   0    progress                  已调度节点的比例
   1    num_ready_norm            就绪节点数（归一化）
   2    num_running_norm          正在执行的节点数 / 流数
   3    num_remaining_norm        剩余未调度节点比例
   4    sim_time_norm             当前模拟时间（归一化）
   5    stream_utilization        流利用率（繁忙流的占比）
   6    running_shared_mem        正在执行的节点的共享内存总量
   7    running_regs              正在执行的节点的寄存器总量
   8    running_threads           正在执行的节点的线程总量
   9    remaining_cp_norm         剩余关键路径长度（归一化）
  10    remaining_work_norm       剩余工作量比例
  11    makespan_estimate_norm    当前估计的 makespan（归一化）
============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .graph_state import GraphState, D_STATIC
from .utils import safe_div

D_DYN = 10      # 动态节点特征维度
D_GLOBAL = 12   # 全局特征维度


@dataclass
class StepResult:
    """env.step() 的返回值。"""
    done: bool       # 所有节点是否都已调度
    reward: float    # 该步的环境奖励（real 模式下会被覆盖）
    info: Dict       # 额外信息


class SchedulingEnv:
    """调度模拟环境 — RL 智能体的交互对象。

    在 real-latency 训练模式下：
    - 动态特征（dyn_node, global）仍由本环境提供
    - reward 被 train.py 中的真实 GPU latency 替代
    - 本环境的 reward 仅用于 surrogate 训练模式
    """

    def __init__(
        self,
        gs: GraphState,
        n_streams: int = 8,                                    # [可调超参] CUDA 流数量
        device: Optional[torch.device] = None,
        reward_weights: Optional[Dict[str, float]] = None,     # [可调超参] 奖励各项权重
    ):
        self.gs = gs
        self.N = len(gs.node_names)              # 图中总节点数
        self.n_streams = max(n_streams, 1)       # CUDA 流数量
        self.device = device or torch.device('cpu')

        # 按类型分组节点
        self._placeholders = [i for i, op in enumerate(gs.op_kinds) if op == 'placeholder']
        self._outputs = [i for i, op in enumerate(gs.op_kinds) if op == 'output']
        self._movables = [i for i in range(self.N) if gs.movable_mask[i].item() == 1.0]
        self._n_movable = len(self._movables)    # 可调度节点数（GoogLeNet ≈ 197）

        # surrogate 模式下的奖励权重
        rw = reward_weights or {}
        self._w_makespan = rw.get('makespan', 1.0)
        self._w_contention = rw.get('contention', 0.1)
        self._w_overlap = rw.get('overlap', 0.05)
        self._w_idle = rw.get('idle', 0.05)

        # 从静态特征中获取初始 makespan 估计
        self._total_work = float(gs.durations.sum().item())
        self._initial_makespan = max(float(gs.durations.max().item()), 1e-9)
        for i in range(self.N):
            cp_bwd_i = gs.x[i, 11].item() * self._initial_makespan
            self._initial_makespan = max(self._initial_makespan, cp_bwd_i)

        # ---- 可变状态（每次 reset 时重新初始化）----
        self._scheduled: List[int] = []         # 已调度节点的顺序
        self._done_set: set = set()             # 已完成的节点集合
        self._in_degree: List[int] = []         # 每个节点的剩余入度

        # 流模拟状态
        self._stream_busy_until: List[float] = []   # 每条流的忙碌截止时间
        self._node_start: List[float] = []           # 每个节点的开始时间
        self._node_finish: List[float] = []          # 每个节点的结束时间
        self._node_stream: List[int] = []            # 每个节点被分配到的流
        self._sim_time: float = 0.0                  # 当前模拟时间
        self._prev_makespan: float = 0.0
        self._step_count: int = 0                    # 已执行的步数
        self._node_ready_step: List[int] = []        # 每个节点变为 ready 的步数

    # ------------------------------------------------------------------
    # 重置 / 执行一步
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """重置环境到初始状态（每个 episode 开头调用一次）。"""
        self._scheduled = []
        self._done_set = set()
        self._in_degree = [len(self.gs.parents[i]) for i in range(self.N)]

        self._stream_busy_until = [0.0] * self.n_streams
        self._node_start = [-1.0] * self.N
        self._node_finish = [-1.0] * self.N
        self._node_stream = [-1] * self.N
        self._sim_time = 0.0
        self._prev_makespan = 0.0
        self._step_count = 0
        self._node_ready_step = [-1] * self.N

        # placeholder 节点（输入张量）自动调度，不需要 RL 决策
        for pid in self._placeholders:
            self._dispatch_node(pid)

        # 标记初始 ready 节点
        for i in self._movables:
            if self._in_degree[i] == 0 and self._node_ready_step[i] < 0:
                self._node_ready_step[i] = self._step_count

    def step(self, action_node_id: int) -> StepResult:
        """执行一步调度：将 action_node_id 分配到最优流上。

        参数:
            action_node_id: 策略网络选择的节点 id（必须在 ready 集合中）

        返回:
            StepResult(done, reward, info)
        """
        if self.is_done():
            return StepResult(done=True, reward=0.0, info={})

        ready = self.ready_mask()
        if ready[action_node_id].item() != 1.0:
            raise ValueError(f"Node {action_node_id} is not in the ready set")

        prev_mk = self._current_makespan()

        # 执行调度（分配流、计算时序、更新依赖）
        self._dispatch_node(action_node_id)
        self._step_count += 1

        # 更新新变为 ready 的节点
        for i in self._movables:
            if i not in self._done_set and self._in_degree[i] == 0 and self._node_ready_step[i] < 0:
                self._node_ready_step[i] = self._step_count

        new_mk = self._current_makespan()

        reward = self._compute_reward(action_node_id, prev_mk, new_mk)
        done = self.is_done()

        return StepResult(done=done, reward=reward, info={
            'makespan': new_mk,
            'step': self._step_count,
        })

    # ------------------------------------------------------------------
    # 查询接口（策略网络使用）
    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        """所有可调度节点是否都已完成。"""
        return all(i in self._done_set for i in self._movables)

    def ready_mask(self) -> torch.Tensor:
        """返回 [N] 的 0/1 掩码，标记哪些节点当前可以被调度。

        节点 ready 的条件：(1) 是可调度节点  (2) 未完成  (3) 所有父节点已完成
        """
        mask = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        for v in self._movables:
            if v in self._done_set:
                continue
            if self._in_degree[v] == 0:   # 入度为 0 = 所有父节点已完成
                mask[v] = 1.0
        return mask

    def scheduled_order(self) -> List[int]:
        """返回当前已调度的节点顺序（用于 CUDA Graph 捕获）。"""
        return list(self._scheduled)

    def current_makespan(self) -> float:
        """返回当前模拟的 makespan（所有已调度节点的最大结束时间）。"""
        return self._current_makespan()

    # ------------------------------------------------------------------
    # 动态特征（每一步都会重新计算，反映当前调度状态）
    # ------------------------------------------------------------------

    def dynamic_node_features(self) -> torch.Tensor:
        """返回 [N, D_DYN=10] 的动态节点特征。

        这些特征每一步都在变化，是策略网络做决策的关键输入。
        """
        feats = torch.zeros(self.N, D_DYN, dtype=torch.float32, device=self.device)

        # 统计当前 ready 集合
        ready_ids = []
        for v in self._movables:
            if v not in self._done_set and self._in_degree[v] == 0:
                ready_ids.append(v)
        n_ready = len(ready_ids)
        n_ready_norm = safe_div(n_ready, self._n_movable)

        # 统计 ready 集合中的类型分布（访存型 vs 计算型）
        ready_mem_count = sum(1 for v in ready_ids if self.gs.is_mem_bound[v].item() > 0.5)
        ready_comp_count = n_ready - ready_mem_count

        # 剩余总工作量
        remaining_work = sum(
            self.gs.durations[v].item() for v in self._movables if v not in self._done_set
        )
        remaining_work = max(remaining_work, 1e-9)

        for i in range(self.N):
            done = i in self._done_set
            running = done and self._node_finish[i] > self._sim_time
            ready = (not done) and (self._in_degree[i] == 0) and (i in set(self._movables))

            feats[i, 0] = 1.0 if done else 0.0       # 是否已完成
            feats[i, 1] = 1.0 if running else 0.0     # 是否正在执行
            feats[i, 2] = 1.0 if ready else 0.0       # 是否就绪（可选择）

            # 剩余父节点占比 — 衡量离就绪还有多远
            total_parents = max(len(self.gs.parents[i]), 1)
            unfinished_parents = sum(1 for p in self.gs.parents[i] if p not in self._done_set)
            feats[i, 3] = safe_div(unfinished_parents, total_parents)

            # 等待时间 — 就绪后等了多久没被调度
            if ready and self._node_ready_step[i] >= 0:
                feats[i, 4] = safe_div(self._step_count - self._node_ready_step[i], self._n_movable)

            # 剩余松弛 — 越小越紧迫
            raw_slack = self.gs.slack[i].item()
            feats[i, 5] = safe_div(max(raw_slack, 0.0), max(self._initial_makespan, 1e-9))

            # 未完成后代工作量占比 — 衡量该节点"卡住了"多少后续工作
            if not done:
                desc_total = self.gs.descendant_work[i].item()
                desc_done = sum(
                    self.gs.durations[c].item()
                    for c in self._all_descendants(i) if c in self._done_set
                )
                feats[i, 6] = safe_div(desc_total - desc_done, max(desc_total, 1e-9))

            if ready:
                feats[i, 7] = n_ready_norm                    # ready 集合大小
                is_mem = self.gs.is_mem_bound[i].item() > 0.5
                same_type = ready_mem_count if is_mem else ready_comp_count
                feats[i, 8] = safe_div(same_type, max(n_ready, 1))  # 同类型竞争

                # 紧迫性 = 该节点到叶子的最长路径 / 剩余总工作量
                cp_bwd = self.gs.x[i, 11].item() * self._initial_makespan
                feats[i, 9] = safe_div(cp_bwd, remaining_work)

        return feats

    def global_features(self) -> torch.Tensor:
        """返回 [D_GLOBAL=12] 的全局资源/进度特征。"""
        g = torch.zeros(D_GLOBAL, dtype=torch.float32, device=self.device)

        n_done = sum(1 for v in self._movables if v in self._done_set)
        n_ready = sum(
            1 for v in self._movables if v not in self._done_set and self._in_degree[v] == 0
        )
        n_remaining = self._n_movable - n_done

        busy_streams = sum(1 for t in self._stream_busy_until if t > self._sim_time)

        # 统计正在执行的节点的 GPU 资源占用
        running_shared = 0.0
        running_regs = 0.0
        running_threads = 0.0
        for v in self._movables:
            if v in self._done_set and self._node_finish[v] > self._sim_time:
                running_shared += self.gs.x[v, 2].item()
                running_regs += self.gs.x[v, 3].item()
                running_threads += self.gs.x[v, 4].item()

        remaining_work = sum(
            self.gs.durations[v].item() for v in self._movables if v not in self._done_set
        )
        mk = self._current_makespan()

        # 剩余关键路径长度
        remaining_cp = 0.0
        for v in self._movables:
            if v not in self._done_set:
                remaining_cp = max(remaining_cp, self.gs.x[v, 11].item() * self._initial_makespan)

        g[0] = safe_div(n_done, self._n_movable)                    # 完成进度
        g[1] = safe_div(n_ready, self._n_movable)                   # 就绪节点比例
        g[2] = safe_div(busy_streams, self.n_streams)               # 流繁忙率
        g[3] = safe_div(n_remaining, self._n_movable)               # 剩余比例
        g[4] = safe_div(self._sim_time, max(self._initial_makespan, 1e-9))  # 模拟时间进度
        g[5] = safe_div(busy_streams, self.n_streams)               # 流利用率
        g[6] = running_shared                                       # 运行中的共享内存
        g[7] = running_regs                                         # 运行中的寄存器
        g[8] = running_threads                                      # 运行中的线程
        g[9] = safe_div(remaining_cp, max(self._initial_makespan, 1e-9))  # 剩余关键路径
        g[10] = safe_div(remaining_work, max(self._total_work, 1e-9))     # 剩余工作量
        g[11] = safe_div(mk, max(self._initial_makespan, 1e-9))          # 当前 makespan 估计
        return g

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _dispatch_node(self, v: int):
        """模拟调度节点 v：分配到最优流，计算时序，更新依赖。

        贪心流分配策略（与 Opara 一致）：
        选择 max(流空闲时间, 依赖完成时间) 最小的那条流。
        """
        if v in self._done_set:
            return

        dur = self.gs.durations[v].item()  # 该节点的执行时间

        # 最早可开始时间 = 所有已完成父节点中最晚结束的时间
        earliest_dep = max(
            (self._node_finish[p] for p in self.gs.parents[v] if p in self._done_set),
            default=0.0,
        )

        # 贪心选流：选 start 最早的流
        best_stream = 0
        best_start = float('inf')
        for s in range(self.n_streams):
            start = max(self._stream_busy_until[s], earliest_dep)
            if start < best_start:
                best_start = start
                best_stream = s

        finish = best_start + dur
        self._node_start[v] = best_start
        self._node_finish[v] = finish
        self._node_stream[v] = best_stream
        self._stream_busy_until[best_stream] = finish
        self._sim_time = max(self._sim_time, best_start)

        self._done_set.add(v)
        self._scheduled.append(v)

        # 更新子节点的入度（父节点完成 → 子节点入度-1）
        for c in self.gs.children[v]:
            self._in_degree[c] -= 1

    def _current_makespan(self) -> float:
        """当前 makespan = 所有已调度节点中最大的结束时间。"""
        if not self._done_set:
            return 0.0
        return max(self._node_finish[v] for v in self._done_set)

    def _compute_reward(self, action: int, prev_mk: float, new_mk: float) -> float:
        """计算 surrogate 环境的 dense reward（仅 surrogate 模式使用）。

        reward = makespan 改善 - 资源竞争惩罚 + 并行重叠奖励 - 流空闲惩罚
        """
        mk_delta = prev_mk - new_mk
        mk_reward = mk_delta / max(self._initial_makespan, 1e-9)

        dur = self.gs.durations[action].item()
        start = self._node_start[action]
        stream = self._node_stream[action]

        contention = 0.0
        overlap = 0.0
        for s in range(self.n_streams):
            if s == stream:
                continue
            if self._stream_busy_until[s] > start:
                other_busy = self._stream_busy_until[s] - start
                contention += self.gs.x[action, 2].item()
                overlap += safe_div(min(dur, other_busy), max(self._initial_makespan, 1e-9))

        idle_penalty = 0.0
        for s in range(self.n_streams):
            if self._stream_busy_until[s] < self._sim_time:
                idle_penalty += safe_div(
                    self._sim_time - self._stream_busy_until[s],
                    max(self._initial_makespan, 1e-9),
                )

        r = (
            self._w_makespan * mk_reward
            - self._w_contention * contention
            + self._w_overlap * overlap
            - self._w_idle * idle_penalty
        )
        return float(r)

    def _all_descendants(self, v: int) -> List[int]:
        """BFS 获取节点 v 的所有后代。"""
        visited = set()
        queue = list(self.gs.children[v])
        while queue:
            c = queue.pop()
            if c in visited:
                continue
            visited.add(c)
            queue.extend(self.gs.children[c])
        return list(visited)
