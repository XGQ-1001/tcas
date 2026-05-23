"""
DepValue
反向递归 + 后继价值分摊 + 纯拓扑结构计算
核心：从叶子节点反向遍历计算所有算子的 DepValue
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import heapq

# ============================================================================
# 1. 调度节点定义（存储 DepValue）
# ============================================================================
@dataclass
class ScheduleNode:
    name: str                       # 算子节点名称
    predecessors: List[str] = field(default_factory=list)  # 前驱节点
    successors: List[str] = field(default_factory=list)    # 后继节点
    in_degree: int = 0              # 入度（核心公式参数）
    depvalue: float = 0.0           # 依赖价值
    priority: float = 0.0           # 调度优先级 = DepValue

# ============================================================================
# 2. DepValue （匹配公式）
# ============================================================================
class DepValueCalculator:
    def __init__(self, nodes: Dict[str, ScheduleNode]):
        self.nodes = nodes
        self.node_in_degree = {name: len(node.predecessors) for name, node in nodes.items()}

    def compute_all_depvalue(self) -> None:
        """
        核心方法：反向拓扑排序计算所有节点的 DepValue
        计算顺序：叶子节点 → 上游节点（必须反向！）
        """
        reverse_topo_order = self._get_reverse_topological_order()

        for node_name in reverse_topo_order:
            node = self.nodes[node_name]
            # 叶子节点：DepValue = 1.0
            if not node.successors:
                node.depvalue = 1.0
            # 非叶子节点：严格按公式计算
            else:
                total = 0.0
                for succ_name in node.successors:
                    succ_node = self.nodes[succ_name]
                    idg = succ_node.in_degree
                    if idg <= 0:
                        continue
                    total += succ_node.depvalue / idg
                node.depvalue = 1.0 + total

    def _get_reverse_topological_order(self) -> List[str]:
        """获取反向拓扑序列（叶子→根节点）"""
        forward_topo = []
        in_degree_copy = self.node_in_degree.copy()
        queue = [n for n, deg in in_degree_copy.items() if deg == 0]

        while queue:
            curr = queue.pop(0)
            forward_topo.append(curr)
            for succ in self.nodes[curr].successors:
                in_degree_copy[succ] -= 1
                if in_degree_copy[succ] == 0:
                    queue.append(succ)
        # 反转得到反向拓扑
        return forward_topo[::-1]

# ============================================================================
# 3. DepValue 优先调度器
# ============================================================================
class DepValueScheduler:
    def __init__(self):
        self.nodes: Dict[str, ScheduleNode] = {}

    def build_graph(self, dag_edges: Dict[str, List[str]]):
        """构建DAG图；dag_edges 须包含图中每个节点名为键（后继可为空列表）。"""
        # 初始化节点与后继
        for node_name, successors in dag_edges.items():
            if node_name not in self.nodes:
                self.nodes[node_name] = ScheduleNode(name=node_name)
            self.nodes[node_name].successors = list(successors)

        # 填充前驱
        for node_name, node in list(self.nodes.items()):
            for succ_name in node.successors:
                if succ_name not in self.nodes:
                    self.nodes[succ_name] = ScheduleNode(name=succ_name)
                if node_name not in self.nodes[succ_name].predecessors:
                    self.nodes[succ_name].predecessors.append(node_name)

        # 计算入度
        for node in self.nodes.values():
            node.in_degree = len(node.predecessors)

    def schedule(self) -> List[str]:
        """执行调度，返回算子顺序（拓扑合法）。"""
        calculator = DepValueCalculator(self.nodes)
        calculator.compute_all_depvalue()

        # 优先级=DepValue
        for node in self.nodes.values():
            node.priority = node.depvalue

        return self._list_scheduling()

    def _list_scheduling(self) -> List[str]:
        """高 DepValue 优先列表调度"""
        schedule_order = []
        in_degree_copy = {n: node.in_degree for n, node in self.nodes.items()}
        ready_queue: List[Tuple[float, str]] = []

        for node_name, deg in in_degree_copy.items():
            if deg == 0:
                heapq.heappush(
                    ready_queue,
                    (-self.nodes[node_name].priority, node_name),
                )

        while ready_queue:
            _, node_name = heapq.heappop(ready_queue)
            schedule_order.append(node_name)

            for succ_name in self.nodes[node_name].successors:
                in_degree_copy[succ_name] -= 1
                if in_degree_copy[succ_name] == 0:
                    heapq.heappush(
                        ready_queue,
                        (-self.nodes[succ_name].priority, succ_name),
                    )

        return schedule_order

    def print_depvalue(self):
        """打印计算结果"""
        print("=" * 50)
        print("节点 DepValue 计算结果（严格匹配论文公式）")
        print("=" * 50)
        for name, node in sorted(self.nodes.items(), key=lambda x: x[1].depvalue, reverse=True):
            print(f"节点 {name} | 入度={node.in_degree} | DepValue={node.depvalue:.2f}")


# ============================================================================
# 4. 与 GraphCapturer / StreamAllocator 集成
# ============================================================================

def _dag_edges_from_fx_graph(graph) -> Dict[str, List[str]]:
    """每个 FX 节点一条出边表，保证与 torch.fx 依赖一致。"""
    return {n.name: [u.name for u in n.users] for n in graph.nodes}


def compute_depvalue_map_from_fx_graph(graph) -> Dict[str, float]:
    """按 DepValue 公式计算 FX 图中各节点名的 depvalue，供 TCAS-RA 等与 profile 无关的结构优先级使用。"""
    dag_edges = _dag_edges_from_fx_graph(graph)
    sched = DepValueScheduler()
    sched.build_graph(dag_edges)
    DepValueCalculator(sched.nodes).compute_all_depvalue()
    return {name: float(node.depvalue) for name, node in sched.nodes.items()}


def assign_stream_with_tcas(
    fx_module: Any,
    node_profiles: Optional[Dict] = None,
    epsilon: float = 1e-4,
) -> Tuple[list, list]:
    """
    将 DepValue 调度顺序写回 FX Graph（仅改节点链表顺序，不改 args/kwargs）。

    保留名称 ``assign_stream_with_tcas`` 与 ``epsilon`` / ``node_profiles``，
    供 ``GraphCapturer.capturer(..., use_tcas=True)`` 调用；DepValue 路径不使用 profile。

    Returns:
        ([], [])：流仍由调用方 ``StreamAllocator.assign_stream`` 分配。
    """
    del node_profiles  # 保留接口
    del epsilon

    graph = fx_module.graph
    dag_edges = _dag_edges_from_fx_graph(graph)
    sched = DepValueScheduler()
    sched.build_graph(dag_edges)
    schedule_order = sched.schedule()

    if not schedule_order:
        return [], []

    print("[DepValue] 正在应用算子发射顺序（高 DepValue 优先列表调度）...")
    nodes_map = {n.name: n for n in graph.nodes}
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    anchor = placeholders[-1] if placeholders else None

    movable = []
    for name in schedule_order:
        n = nodes_map.get(name)
        if n is None:
            continue
        if n.op in ("placeholder", "output"):
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
        print(f"[DepValue] 应用调度顺序失败，保持原顺序: {e}")

    return [], []
