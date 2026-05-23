"""Multi-task GNN scheduling package.

将"多任务推理调度"问题转化为"更大 DAG 的算子调度"问题：
  - 将 K 个 batch=1 的独立 DAG 拼成 1 个大 super-DAG
  - 同一个 GNN + PPO 策略可以直接应用（graph abstraction 的威力）
  - GNN 学会「跨任务交织算子」以最大化 GPU 利用率
"""

from .super_dag import build_super_dag, SuperDAGInfo
from .baselines import (
    topological_order,
    round_robin_order,
    opara_like_order,
    random_order,
    simulate_makespan,
)

__all__ = [
    'build_super_dag',
    'SuperDAGInfo',
    'topological_order',
    'round_robin_order',
    'opara_like_order',
    'random_order',
    'simulate_makespan',
]
