"""Time-Constrained + Resource-Aware Scheduler (TCAS-RA)

默认模式（``opara_primary_cp_tiebreak=False``）：
- 全局仍以 **DepValue** 为主：就绪集中仅当 depvalue 落在「与当前最大 DepValue 同一梯队」时，
  才在 **该批节点内** 做访存/计算双队列交错 + 队内 shared_mem 最小，避免为交错牺牲更高 DepValue 的节点。
- DepValue 由 ``TimeConstrainedScheduler.compute_depvalue_map_from_fx_graph`` 计算，与 profile 无关。

"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from Opara.OperatorLauncher import is_mem_access_intensive_by_name
from Opara.TimeConstrainedScheduler import compute_depvalue_map_from_fx_graph


# ============================================================================
# 诊断：DepValue 梯队与算/存交错是否「有机会生效」
# ============================================================================


@dataclass
class DepValueBandDiagnostics:
    """由与 `_list_scheduling_ra` 一致的模拟得到，用于判断新算法是否可能带来与纯 DepValue 不同的序。"""

    total_steps: int = 0
    steps_band_singleton: int = 0  # |band|==1，仅结构优先级起作用，无队内交错
    steps_band_multi: int = 0  # |band|>1，同一 DepValue 梯队内有多候选
    steps_interleave_both_queues: int = 0  # |band|>1 且访存类与计算类均非空（真正走交替分支）
    steps_band_multi_mem_only: int = 0  # |band|>1 但全为访存类
    steps_band_multi_comp_only: int = 0  # |band|>1 但全为计算类
    depvalue_distinct_count: int = 0
    depvalue_min_positive_gap: float = 0.0  # 全体 depvalue 排序后相邻差的最小值（结构尺度）
    epsilon_threshold_max: float = 0.0  # 模拟使用的 max(ε_abs, ε_rel*dmax) 在全程中的最大值（量级参考）

    def interleave_effective_ratio(self) -> float:
        if self.total_steps <= 0:
            return 0.0
        return self.steps_interleave_both_queues / self.total_steps


# ============================================================================
# 数据结构
# ============================================================================


@dataclass
class ScheduleNodeRA:
    node: object
    name: str
    duration: float = 0.0

    # timing
    est: float = 0.0
    eft: float = 0.0
    lst: float = float('inf')
    lft: float = float('inf')
    slack: float = float('inf')

    is_critical: bool = False

    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)

    # resource (normalized)
    shared_mem: float = 0.0
    regs: float = 0.0
    threads: float = 0.0
    occupancy: float = 0.0
    is_mem_bound: bool = False


def _estimate_duration_from_profile(profile_info) -> float:
    # 与原 TCAS 保持一致：profile list 的 dur 求和，否则使用 op 类型默认值
    if profile_info and isinstance(profile_info, list) and len(profile_info) > 0:
        return float(sum(k.get('dur', 1.0) for k in profile_info))
    return 1.0


def _extract_resource_vector(
    profile_info,
    device_props: Optional[Dict] = None,
) -> Tuple[float, float, float, float]:
    """Return (shared_mem, regs, threads, occupancy) normalized.

    - shared_mem: max(shared bytes / sharedMemPerBlock)
    - threads: max(threads per block / maxThreadsPerBlock)
    - regs: (threads_norm * regs_per_thread/regsPerBlock) 的最大值
    - occupancy: duration-weighted average achieved occupancy [0,1]

    若缺少 profile 或 device_props，则返回 0。
    """

    if not (profile_info and isinstance(profile_info, list) and len(profile_info) > 0 and device_props):
        return 0.0, 0.0, 0.0, 0.0

    sharedMemPerBlock = float(device_props.get('sharedMemPerBlock', 1.0) or 1.0)
    regsPerBlock = float(device_props.get('regsPerBlock', 1.0) or 1.0)
    maxThreadsPerBlock = float(device_props.get('maxThreadsPerBlock', 1.0) or 1.0)

    max_shared = 0.0
    max_regs = 0.0
    max_threads = 0.0
    occ_weighted = 0.0
    dur_sum = 0.0

    for evt in profile_info:
        if not isinstance(evt, dict):
            continue

        args = evt.get('args', {}) if isinstance(evt.get('args', {}), dict) else {}
        dur = float(evt.get('dur', 0.0) or 0.0)

        block = args.get('block', [0, 0, 0])
        try:
            threads = float(block[0]) * float(block[1]) * float(block[2])
        except Exception:
            threads = 0.0

        shared = float(args.get('shared memory', 0.0) or 0.0)
        regs_per_thread = float(args.get('registers per thread', 0.0) or 0.0)
        occ = float(args.get('est. achieved occupancy %', 0.0) or 0.0) / 100.0

        threads_norm = threads / maxThreadsPerBlock if maxThreadsPerBlock > 0 else 0.0

        max_shared = max(max_shared, shared / sharedMemPerBlock)
        max_threads = max(max_threads, threads_norm)
        max_regs = max(max_regs, threads_norm * (regs_per_thread / regsPerBlock))

        occ_weighted += occ * dur
        dur_sum += dur

    occupancy = (occ_weighted / dur_sum) if dur_sum > 0 else 0.0
    return float(max_shared), float(max_regs), float(max_threads), float(occupancy)


# ============================================================================
# 关键路径分析（与原 TCAS 思路保持一致）
# ============================================================================


class CriticalPathAnalyzerRA:
    def __init__(self, nodes: Dict[str, ScheduleNodeRA]):
        self.nodes = nodes
        self.critical_path: List[str] = []
        self.makespan: float = 0.0

    def analyze(self) -> Tuple[List[str], float]:
        self._forward_pass()
        self._backward_pass()
        self._identify_critical_path()
        return self.critical_path, self.makespan

    def _topological_sort(self) -> List[str]:
        in_degree = {name: len(n.predecessors) for name, n in self.nodes.items()}
        queue = [name for name, deg in in_degree.items() if deg == 0]
        out: List[str] = []
        while queue:
            name = queue.pop(0)
            out.append(name)
            for succ in self.nodes[name].successors:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        return out

    def _forward_pass(self):
        topo = self._topological_sort()
        for name in topo:
            n = self.nodes[name]
            n.est = 0.0 if not n.predecessors else max(self.nodes[p].eft for p in n.predecessors)
            n.eft = n.est + n.duration
        self.makespan = max((n.eft for n in self.nodes.values()), default=0.0)

    def _backward_pass(self):
        topo = self._topological_sort()
        topo.reverse()
        for name in topo:
            n = self.nodes[name]
            n.lft = self.makespan if not n.successors else min(self.nodes[s].lst for s in n.successors)
            n.lst = n.lft - n.duration

    def _identify_critical_path(self):
        self.critical_path = []
        for name, n in self.nodes.items():
            n.slack = n.lst - n.est
            if abs(n.slack) < 1e-6:
                n.is_critical = True
                self.critical_path.append(name)


# ============================================================================
# TCAS-RA 调度器
# ============================================================================


class TimeConstrainedResourceAwareScheduler:
    """TCAS-RA：默认 DepValue 梯队内交错；可选 Opara 主序 + CP tie-break。"""

    def __init__(
        self,
        tie_delta: float = 0.05,
        w_res: float = 0.5,
        ema_decay: float = 0.7,
        opara_primary_cp_tiebreak: bool = False,
        depvalue_epsilon_abs: float = 1e-6,
        depvalue_epsilon_rel: float = 1e-9,
    ):
        # 兼容旧调用；当前 DepValue 梯队调度不使用 tie_delta / w_res / ema_decay
        self.tie_delta = float(tie_delta)
        self.w_res = float(w_res)
        self.ema_decay = float(ema_decay)
        self.opara_primary_cp_tiebreak = bool(opara_primary_cp_tiebreak)
        self.depvalue_epsilon_abs = float(depvalue_epsilon_abs)
        self.depvalue_epsilon_rel = float(depvalue_epsilon_rel)

        self.nodes: Dict[str, ScheduleNodeRA] = {}
        self.schedule_order: List[str] = []
        self._fx_graph: Optional[object] = None
        self._device_props: Optional[Dict] = None
        self._depvalue: Dict[str, float] = {}
        # 与 OperatorLauncher.launch 一致：首次进入「双队列交替」时先取计算密集队列
        self._mem_alt_flag: bool = True

    def build_schedule_graph(
        self,
        fx_graph,
        node_profiles: Optional[Dict] = None,
        device_props: Optional[Dict] = None,
    ):
        self.nodes.clear()
        self._fx_graph = fx_graph
        self._device_props = device_props
        node_profiles = node_profiles or {}

        for node in fx_graph.nodes:
            prof = node_profiles.get(node.name, {})
            dur = _estimate_duration_from_profile(prof) if isinstance(prof, list) else 1.0
            shared_mem, regs, threads, occ = _extract_resource_vector(prof, device_props=device_props)

            # mem_pressure：名字启发式 + 低 occ 辅助（与 OperatorLauncher 分类一致）
            is_mem = is_mem_access_intensive_by_name(node.name) or (occ < 0.25 and shared_mem < 0.2)

            self.nodes[node.name] = ScheduleNodeRA(
                node=node,
                name=node.name,
                duration=dur,
                predecessors=[inp.name for inp in node.all_input_nodes],
                successors=[u.name for u in node.users],
                shared_mem=shared_mem,
                regs=regs,
                threads=threads,
                occupancy=occ,
                is_mem_bound=is_mem,
            )

        self._depvalue = compute_depvalue_map_from_fx_graph(fx_graph)

    def schedule(self) -> List[str]:
        cp_analyzer = CriticalPathAnalyzerRA(self.nodes)
        critical_path, makespan = cp_analyzer.analyze()

        print(f"[TCAS-RA] 关键路径长度(时长估计): {makespan:.2f}")
        print(f"[TCAS-RA] 关键路径节点数: {len(critical_path)}")

        if self.opara_primary_cp_tiebreak:
            if self._fx_graph is None or not self._device_props:
                raise ValueError(
                    "opara_primary_cp_tiebreak requires device_props from profile; build_schedule_graph must set _device_props."
                )
            print("[TCAS-RA] 调度：Opara 主序（launch），关键路径仅作 metric 并列 tie-break")
            from Opara import OperatorLauncher

            slack_by_name = {name: float(n.slack) for name, n in self.nodes.items()}
            critical_by_name = {name: bool(n.is_critical) for name, n in self.nodes.items()}
            result, _ = OperatorLauncher.get_topo_with_cp_tiebreak(
                self._fx_graph.nodes,
                float(self._device_props["sharedMemPerBlock"]),
                float(self._device_props["regsPerBlock"]),
                float(self._device_props["maxThreadsPerBlock"]),
                slack_by_name,
                critical_by_name,
            )
            self.schedule_order = list(result)
            return self.schedule_order

        print(
            f"[TCAS-RA] DepValue 主序 + 梯队内算存交错 "
            f"(ε_abs={self.depvalue_epsilon_abs:g}, ε_rel={self.depvalue_epsilon_rel:g})"
        )
        self.schedule_order = self._list_scheduling_ra()
        return self.schedule_order

    def _topological_order(self) -> List[str]:
        in_degree = {name: len(n.predecessors) for name, n in self.nodes.items()}
        q = deque([name for name, deg in in_degree.items() if deg == 0])
        out: List[str] = []
        while q:
            name = q.popleft()
            out.append(name)
            for succ in self.nodes[name].successors:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    q.append(succ)
        return out

    def _pop_min_shared_mem(self, names: List[str]) -> str:
        """同类队列内选 shared_mem 最小（与 Opara pop_from_queue 一致）；全 0 时取列表首。"""
        if len(names) == 1:
            return names[0]
        best = names[0]
        min_m = self.nodes[best].shared_mem
        for n in names[1:]:
            m = self.nodes[n].shared_mem
            if m < min_m - 1e-15:
                min_m = m
                best = n
        return best

    def _list_scheduling_ra(self) -> List[str]:
        """DepValue 同一梯队内的就绪节点之间做算存交错；梯队由 max(depvalue)-ε 界定。"""
        schedule_order: List[str] = []
        in_degree = {name: len(n.predecessors) for name, n in self.nodes.items()}
        ready: List[str] = []
        for name in self._topological_order():
            if in_degree[name] == 0:
                ready.append(name)

        while ready:
            dvals = [self._depvalue.get(n, 0.0) for n in ready]
            dmax = max(dvals)
            thr = max(self.depvalue_epsilon_abs, self.depvalue_epsilon_rel * max(dmax, 1e-30))
            band = [n for n in ready if self._depvalue.get(n, 0.0) >= dmax - thr]

            if len(band) == 1:
                pick = band[0]
            else:
                mem_names: List[str] = []
                comp_names: List[str] = []
                for name in band:
                    if is_mem_access_intensive_by_name(name):
                        mem_names.append(name)
                    else:
                        comp_names.append(name)

                if not mem_names:
                    pick = self._pop_min_shared_mem(comp_names)
                elif not comp_names:
                    pick = self._pop_min_shared_mem(mem_names)
                else:
                    self._mem_alt_flag = not self._mem_alt_flag
                    q = mem_names if self._mem_alt_flag else comp_names
                    pick = self._pop_min_shared_mem(q)

            ready.remove(pick)
            schedule_order.append(pick)

            for succ_name in self.nodes[pick].successors:
                in_degree[succ_name] -= 1
                if in_degree[succ_name] == 0:
                    ready.append(succ_name)

        return schedule_order

    def collect_depvalue_band_diagnostics(self) -> DepValueBandDiagnostics:
        """不修改 FX；按与 `_list_scheduling_ra` 相同规则逐步模拟，统计梯队大小与算/存交错是否发生。

        用于在跑 benchmark 前判断：若 ``steps_interleave_both_queues`` 长期为 0，
        则「DepValue + 梯队内交错」相对纯 DepValue 往往 **序相同或极少不同**，加速可能不明显。
        """
        diag = DepValueBandDiagnostics()
        if not self.nodes or not self._depvalue:
            return diag

        vals = sorted(set(self._depvalue.values()))
        diag.depvalue_distinct_count = len(vals)
        if len(vals) >= 2:
            diag.depvalue_min_positive_gap = min(
                vals[i + 1] - vals[i] for i in range(len(vals) - 1)
            )

        saved_flag = self._mem_alt_flag
        self._mem_alt_flag = True

        in_degree = {name: len(n.predecessors) for name, n in self.nodes.items()}
        ready: List[str] = []
        for name in self._topological_order():
            if in_degree[name] == 0:
                ready.append(name)

        thr_max = 0.0
        while ready:
            diag.total_steps += 1
            dvals = [self._depvalue.get(n, 0.0) for n in ready]
            dmax = max(dvals)
            thr = max(self.depvalue_epsilon_abs, self.depvalue_epsilon_rel * max(dmax, 1e-30))
            thr_max = max(thr_max, thr)
            band = [n for n in ready if self._depvalue.get(n, 0.0) >= dmax - thr]

            if len(band) == 1:
                diag.steps_band_singleton += 1
                pick = band[0]
            else:
                diag.steps_band_multi += 1
                mem_names = [n for n in band if is_mem_access_intensive_by_name(n)]
                comp_names = [n for n in band if not is_mem_access_intensive_by_name(n)]
                if not mem_names:
                    diag.steps_band_multi_comp_only += 1
                    pick = self._pop_min_shared_mem(comp_names)
                elif not comp_names:
                    diag.steps_band_multi_mem_only += 1
                    pick = self._pop_min_shared_mem(mem_names)
                else:
                    diag.steps_interleave_both_queues += 1
                    self._mem_alt_flag = not self._mem_alt_flag
                    q = mem_names if self._mem_alt_flag else comp_names
                    pick = self._pop_min_shared_mem(q)

            ready.remove(pick)
            for succ_name in self.nodes[pick].successors:
                in_degree[succ_name] -= 1
                if in_degree[succ_name] == 0:
                    ready.append(succ_name)

        diag.epsilon_threshold_max = thr_max
        self._mem_alt_flag = saved_flag
        return diag


def diagnose_depvalue_band_interleaving(
    fx_graph,
    depvalue_epsilon_abs: float = 1e-6,
    depvalue_epsilon_rel: float = 1e-9,
) -> DepValueBandDiagnostics:
    """对给定 FX Graph 构建结构信息与 DepValue，并返回梯队/交错统计（无需 profile）。"""
    sched = TimeConstrainedResourceAwareScheduler(
        depvalue_epsilon_abs=depvalue_epsilon_abs,
        depvalue_epsilon_rel=depvalue_epsilon_rel,
    )
    sched.build_schedule_graph(fx_graph, node_profiles=None, device_props=None)
    return sched.collect_depvalue_band_diagnostics()


# ============================================================================
# 与现有系统集成：仅重排 FX Graph 节点顺序（不做 stream 分配）
# ============================================================================


def assign_stream_with_tcas_ra(
    fx_module,
    node_profiles: Optional[Dict] = None,
    device_props: Optional[Dict] = None,
    tie_delta: float = 0.05,
    w_res: float = 0.5,
    ema_decay: float = 0.7,
    opara_primary_cp_tiebreak: bool = False,
    depvalue_epsilon_abs: float = 1e-6,
    depvalue_epsilon_rel: float = 1e-9,
):
    """TCAS-RA：应用新的算子发射顺序到 FX Graph。

    opara_primary_cp_tiebreak：为 True 时主序与 OperatorLauncher.launch 一致，
    关键路径仅在队内 metric（shared）并列时 tie-break。

    depvalue_epsilon_abs / depvalue_epsilon_rel：默认调度下 DepValue「同一梯队」阈值
   （绝对值与相对 dmax 取 max）。

    返回值保持与原 `assign_stream_with_tcas` 一致：返回空 (streams, events)，
    由外部继续调用 StreamAllocator 分配 stream。
    """

    graph = fx_module.graph

    sched = TimeConstrainedResourceAwareScheduler(
        tie_delta=tie_delta,
        w_res=w_res,
        ema_decay=ema_decay,
        opara_primary_cp_tiebreak=opara_primary_cp_tiebreak,
        depvalue_epsilon_abs=depvalue_epsilon_abs,
        depvalue_epsilon_rel=depvalue_epsilon_rel,
    )
    sched.build_schedule_graph(graph, node_profiles=node_profiles, device_props=device_props)
    schedule_order = sched.schedule()

    if schedule_order:
        if opara_primary_cp_tiebreak:
            print("[TCAS-RA] 正在应用算子顺序（Opara 主序 + CP tie-break）...")
        else:
            print("[TCAS-RA] 正在应用算子顺序（DepValue 主序；梯队内访存/计算交替）...")

        nodes_map = {n.name: n for n in graph.nodes}
        placeholders = [n for n in graph.nodes if n.op == 'placeholder']
        anchor = placeholders[-1] if placeholders else None

        movable: List[object] = []
        for name in schedule_order:
            n = nodes_map.get(name)
            if n is None:
                continue
            if n.op in ('placeholder', 'output'):
                continue
            movable.append(n)

        try:
            if anchor is not None and movable:
                last = anchor
                for n in movable:
                    last.append(n)
                    last = n

            graph.lint()
            fx_module.recompile()
        except Exception as e:
            print(f"[TCAS-RA] 应用调度顺序失败，保持原顺序: {e}")

    return [], []
