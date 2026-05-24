"""CUDA Graph 捕获与延迟评估 — 连接 RL 策略与真实 GPU 性能的桥梁。

============================================================================
核心作用：
  将 RL 策略网络输出的「调度顺序」应用到真实 GPU 上：
    1. 按调度顺序重排 FX 计算图的节点
    2. 分配 CUDA 流（使用 Opara 的 StreamAllocator）
    3. 捕获 CUDA Graph（录制一次，之后可重放）
    4. benchmark 真实延迟

这是 real-latency 训练模式的关键组件。
每个 episode 的流程：
  策略网络输出调度顺序 → fast_eval_latency() → 真实延迟 → reward

CUDA Graph 原理：
  - 普通 PyTorch: 每次推理都要一个个发射 kernel, CPU 开销大
  - CUDA Graph: 第一次「录制」所有 kernel 的执行序列,
    之后「重放」只需一个 API 调用, 几乎零 CPU 开销
  - 多流并行: 把独立的 kernel 放到不同 CUDA stream, 并行执行
============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import warnings

import os
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from Opara import GraphCapturer
from Opara import OperatorLauncher
from Opara import StreamAllocator
from .utils import extract_first_fx_graph


@dataclass
class BenchResult:
    """benchmark 结果。"""
    mean_ms: float         # 平均延迟（毫秒）
    std_ms: float          # 标准差
    times_ms: List[float]  # 每次测量的原始值


def benchmark_runner(
    runner: Callable,
    inputs: Tuple[torch.Tensor, ...],
    iterations: int,
    warmups: int,
) -> BenchResult:
    """精确测量 CUDA runner 的执行延迟。

    使用 CUDA Event 计时（纳秒精度），先预热再正式测量。
    """
    # 预热：让 GPU 进入稳定状态
    with torch.no_grad():
        for _ in range(warmups):
            runner(*inputs)

    torch.cuda.synchronize()
    times: List[float] = []

    # 正式测量
    with torch.no_grad():
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            runner(*inputs)     # 重放 CUDA Graph
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))  # 毫秒

    mean_ms = float(np.mean(times)) if times else float('inf')
    std_ms = float(np.std(times)) if times else 0.0
    return BenchResult(mean_ms=mean_ms, std_ms=std_ms, times_ms=times)


def _apply_schedule_order_to_fx(
    fx_module,
    schedule_order: Sequence[str],
):
    """按指定顺序重排 FX 图的节点链。

    FX 图中节点的顺序决定了 kernel 的发射顺序。
    通过 .append() 方法移动节点位置来改变执行顺序。
    placeholder 节点（输入）保持在最前面。
    """
    graph = fx_module.graph
    nodes_map = {n.name: n for n in graph.nodes}

    placeholders = [n for n in graph.nodes if n.op == 'placeholder']
    anchor = placeholders[-1] if placeholders else None

    seen = set()
    movable = []
    for name in schedule_order:
        if name in seen:
            continue
        seen.add(name)
        n = nodes_map.get(name)
        if n is None:
            continue
        if n.op in ('placeholder', 'output'):
            continue
        movable.append(n)

    if anchor is not None and movable:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r'Trying to prepend a node to itself\..*',
                category=UserWarning,
            )
            last = anchor
            for n in movable:
                last.append(n)   # 将节点 n 移动到 last 之后
                last = n

    graph.lint()           # 验证图的合法性
    fx_module.recompile()  # 重新编译 forward 方法


def capturer_gnn(
    inputs: Sequence[torch.Tensor],
    model,
    schedule_order: Sequence[str],
    copy_outputs: bool = False,
):
    """从头构建 CUDA Graph runner（包含 dynamo.explain + recompile，较慢）。

    完整流程：
      1. torch._dynamo.explain → FX 图
      2. OperatorLauncher.recompile → profiling
      3. _apply_schedule_order_to_fx → 重排节点
      4. StreamAllocator.assign_stream → 分配流
      5. CUDA Graph capture → runner
    """

    assert isinstance(inputs, (list, tuple))

    static_inputs = [torch.zeros_like(x, device='cuda') for x in inputs]

    fx_module = extract_first_fx_graph(model, inputs)
    fx_module.cuda()

    model_class_name = model.__class__.__name__

    node_profiles, device_props = OperatorLauncher.recompile(
        model_class_name,
        fx_module,
        inputs,
        apply_opara_schedule=False,
    )

    _apply_schedule_order_to_fx(fx_module, schedule_order=schedule_order)

    # Opara 的流分配算法：将独立的 kernel 分配到不同流以并行执行
    all_streams, _ = StreamAllocator.assign_stream(fx_module.graph)

    # CUDA Graph 捕获
    all_events = [torch.cuda.Event() for _ in range(len(all_streams))]
    first_stream = all_streams[0]
    first_event = all_events[0]

    interpreter = GraphCapturer.Scheduler(fx_module)

    # 预热 interpreter
    with torch.no_grad():
        for _ in range(3):
            interpreter.run(*inputs)

    # 正式捕获 CUDA Graph
    with torch.no_grad():
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=first_stream):
            # 所有流等待第一个流的 event（同步起点）
            first_event.record(first_stream)
            for i, s in enumerate(all_streams):
                if i > 0:
                    s.wait_event(first_event)

            # 执行计算图（被录制到 CUDA Graph 中）
            static_outputs = interpreter.run(*static_inputs)

            # 所有流同步回第一个流（同步终点）
            torch.cuda.set_stream(first_stream)
            for i, e in enumerate(all_events):
                if i > 0:
                    e.record(all_streams[i])
            for i, e in enumerate(all_events):
                if i > 0:
                    first_stream.wait_event(e)

        torch.cuda.synchronize()

        if not isinstance(static_outputs, (list, tuple)):
            static_outputs = (static_outputs,)

    def run(*new_inputs):
        """重放 CUDA Graph — 极低开销的推理。"""
        assert isinstance(new_inputs, (list, tuple))
        assert len(static_inputs) == len(new_inputs)
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)      # 只需拷贝输入
        with torch.no_grad():
            g.replay()          # 重放录制的所有 kernel
        if copy_outputs:
            return [x.clone() for x in static_outputs]
        return static_outputs

    try:
        setattr(run, '_opara_meta', {
            'algorithm': 'GNN-RL',
            'num_nodes': len(list(fx_module.graph.nodes)),
            'num_streams': len(all_streams),
        })
    except Exception:
        pass

    return run, node_profiles, device_props


def capturer_gnn_from_fx(
    fx_module,
    inputs: Sequence[torch.Tensor],
    schedule_order: Sequence[str],
    copy_outputs: bool = False,
):
    """从已有的 FX 模块捕获 CUDA Graph（跳过 dynamo.explain，更快）。

    与 capturer_gnn 的区别：不重新编译，直接使用已有的 fx_module。
    训练时每个 episode 调用一次（配合 deepcopy）。
    """

    assert isinstance(inputs, (list, tuple))

    static_inputs = [torch.zeros_like(x, device='cuda') for x in inputs]

    # 就地修改 FX 图节点顺序
    _apply_schedule_order_to_fx(fx_module, schedule_order=schedule_order)

    # 分配 CUDA 流
    all_streams, _ = StreamAllocator.assign_stream(fx_module.graph)

    all_events = [torch.cuda.Event() for _ in range(len(all_streams))]
    first_stream = all_streams[0]
    first_event = all_events[0]

    interpreter = GraphCapturer.Scheduler(fx_module)

    with torch.no_grad():
        for _ in range(3):
            interpreter.run(*inputs)

    with torch.no_grad():
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=first_stream):
            first_event.record(first_stream)
            for i, s in enumerate(all_streams):
                if i > 0:
                    s.wait_event(first_event)

            static_outputs = interpreter.run(*static_inputs)

            torch.cuda.set_stream(first_stream)
            for i, e in enumerate(all_events):
                if i > 0:
                    e.record(all_streams[i])
            for i, e in enumerate(all_events):
                if i > 0:
                    first_stream.wait_event(e)

        torch.cuda.synchronize()

        if not isinstance(static_outputs, (list, tuple)):
            static_outputs = (static_outputs,)

    def run(*new_inputs):
        assert isinstance(new_inputs, (list, tuple))
        assert len(static_inputs) == len(new_inputs)
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)
        with torch.no_grad():
            g.replay()
        if copy_outputs:
            return [x.clone() for x in static_outputs]
        return static_outputs

    try:
        setattr(run, '_opara_meta', {
            'algorithm': 'GNN-Policy(Greedy)',
            'num_nodes': len(list(fx_module.graph.nodes)),
            'num_streams': len(all_streams),
        })
    except Exception:
        pass

    return run


def measure_latency_ms(
    runner: Callable,
    inputs: Tuple[torch.Tensor, ...],
    iterations: int,
    warmups: int,
) -> float:
    """快捷方法：benchmark 并返回平均延迟（毫秒）。"""
    return benchmark_runner(runner, inputs=inputs, iterations=iterations, warmups=warmups).mean_ms


def fast_eval_latency(
    fx_module,
    inputs: Sequence[torch.Tensor],
    schedule_order: Sequence[str],
    iterations: int = 10,
    warmups: int = 3,
) -> float:
    """★ 训练时每 episode 调用的快速评估函数。

    流程：
      1. deep copy FX 模块（不修改原始模块）
      2. 应用调度顺序 + 流分配 + CUDA Graph 捕获
      3. benchmark 真实延迟
      4. 清理资源

    耗时约 0.25 秒/次（H20, GoogLeNet），是训练的主要时间瓶颈。
    """
    import copy

    fx_copy = copy.deepcopy(fx_module)
    runner = capturer_gnn_from_fx(fx_copy, inputs, schedule_order, copy_outputs=False)
    latency = benchmark_runner(runner, inputs=inputs, iterations=iterations, warmups=warmups).mean_ms
    del runner, fx_copy
    return latency
