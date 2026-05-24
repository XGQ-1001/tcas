"""Shared utility functions for gnn_strategy."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import math

# ---------------------------------------------------------------------------
# Operator type classification (aligned with OperatorLauncher)
# ---------------------------------------------------------------------------

_COMPUTE_HEAVY_SUBSTRINGS = (
    "addmm", "bmm", "matmul", "linear", "conv2d", "conv1d", "conv3d",
    "convolution", "cudnn_convolution", "_convolution", "einsum",
    "scaled_dot_product", "sdpa", "flash_attn", "grouped_gemm", "gemm",
)

_MEMORY_INTENSIVE_SUBSTRINGS = (
    "add", "cast", "ceil", "clip", "concat", "exp", "floor", "log",
    "gelu", "neg", "pow", "reciprocal", "relu", "sigmoid", "slice",
    "sqrt", "sub", "tanh", "transpose", "unsqueeze", "view", "avg_pool",
    "reshape", "max_pool", "adaptive_avg_pool", "adaptive_max_pool",
    "permute", "flatten", "dropout", "batch_norm", "layer_norm",
    "instance_norm", "contiguous", "ones", "to", "softmax",
    "native_layer_norm", "rms_norm", "masked_fill", "embedding",
    "embedding_bag", "index_select", "gather", "where", "silu",
    "hardswish", "pad", "clone", "split", "chunk", "repeat", "expand",
    "cumsum", "one_hot", "arange",
)


def is_mem_intensive(node_name: str) -> bool:
    lower = (node_name or "").lower()
    if lower == "mm":
        return False
    if any(s in lower for s in _COMPUTE_HEAVY_SUBSTRINGS):
        return False
    if any(s in lower for s in _MEMORY_INTENSIVE_SUBSTRINGS):
        return True
    return False


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, x)))


def duration_from_profile(profile_info) -> float:
    if profile_info and isinstance(profile_info, list):
        return float(sum(evt.get('dur', 0.0) or 0.0 for evt in profile_info))
    return 0.0


def extract_resource_vector(
    profile_info,
    device_props: Optional[Dict] = None,
) -> Tuple[float, float, float, float]:
    """Return normalized (shared_mem, regs, threads, occupancy)."""
    if not (profile_info and isinstance(profile_info, list) and device_props):
        return 0.0, 0.0, 0.0, 0.0

    smpb = float(device_props.get('sharedMemPerBlock', 1.0) or 1.0)
    rpb = float(device_props.get('regsPerBlock', 1.0) or 1.0)
    mtpb = float(device_props.get('maxThreadsPerBlock', 1.0) or 1.0)

    max_shared, max_regs, max_threads = 0.0, 0.0, 0.0
    occ_weighted, dur_sum = 0.0, 0.0

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
        rpt = float(args.get('registers per thread', 0.0) or 0.0)
        occ = float(args.get('est. achieved occupancy %', 0.0) or 0.0) / 100.0

        tn = safe_div(threads, mtpb)
        max_shared = max(max_shared, safe_div(shared, smpb))
        max_threads = max(max_threads, tn)
        max_regs = max(max_regs, tn * safe_div(rpt, rpb))

        occ_weighted += occ * dur
        dur_sum += dur

    occupancy = safe_div(occ_weighted, dur_sum)
    return float(max_shared), float(max_regs), float(max_threads), clamp(float(occupancy))


def extract_first_fx_graph(model, inputs):
    """Return the first FX graph from `torch._dynamo.explain`.

    PyTorch versions differ here:
    - older versions returned a tuple where `graphs` was the 3rd item
    - newer versions return an `ExplainOutput` object with a `.graphs` field
    """
    import torch._dynamo as dynamo

    dynamo.reset()
    with __import__("torch").no_grad():
        try:
            explain_out = dynamo.explain(model)(*inputs)
        except TypeError:
            explain_out = dynamo.explain(model, *inputs)

    if hasattr(explain_out, 'graphs'):
        graphs = explain_out.graphs
    elif isinstance(explain_out, tuple) and len(explain_out) >= 3:
        graphs = explain_out[2]
    else:
        raise TypeError(f"Unsupported torch._dynamo.explain output type: {type(explain_out)!r}")

    if not graphs:
        raise RuntimeError("torch._dynamo.explain returned no graphs")

    return graphs[0]


# ---------------------------------------------------------------------------
# DAG analysis helpers
# ---------------------------------------------------------------------------

def normalize_by_max(vals: List[float], eps: float = 1e-9) -> List[float]:
    if not vals:
        return vals
    m = max(vals)
    if m < eps:
        return [0.0 for _ in vals]
    return [float(v / m) for v in vals]


def topo_sort(n: int, parents: List[List[int]], children: List[List[int]]) -> List[int]:
    in_deg = [len(parents[i]) for i in range(n)]
    q = [i for i in range(n) if in_deg[i] == 0]
    out: List[int] = []
    head = 0
    while head < len(q):
        v = q[head]; head += 1
        out.append(v)
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                q.append(c)
    return out if len(out) == n else list(range(n))


def compute_critical_path(
    n: int,
    parents: List[List[int]],
    children: List[List[int]],
    durations: List[float],
) -> Tuple[List[float], List[float], List[float], List[float], float]:
    """CPM forward/backward pass.

    Returns (est, eft, lst, lft, makespan).
    """
    topo = topo_sort(n, parents, children)
    est = [0.0] * n
    eft = [0.0] * n
    for v in topo:
        est[v] = max((eft[p] for p in parents[v]), default=0.0)
        eft[v] = est[v] + durations[v]
    makespan = max(eft) if eft else 0.0

    lft = [makespan] * n
    lst = [0.0] * n
    for v in reversed(topo):
        lft[v] = min((lst[c] for c in children[v]), default=makespan)
        lst[v] = lft[v] - durations[v]

    return est, eft, lst, lft, makespan


def compute_descendant_work(
    n: int,
    children: List[List[int]],
    durations: List[float],
) -> List[float]:
    """Total duration of all descendants (inclusive of self)."""
    topo_rev = list(reversed(topo_sort(n, [[] for _ in range(n)], children)))
    # Use the fact that topo_sort with empty parents gives all nodes with 0 in-degree first
    # We need reverse topo for bottom-up
    in_deg_for_rev = [len(children[i]) for i in range(n)]
    q = [i for i in range(n) if in_deg_for_rev[i] == 0]
    order: List[int] = []
    head = 0
    while head < len(q):
        v = q[head]; head += 1
        order.append(v)
        for p in range(n):
            if v in children[p]:
                in_deg_for_rev[p] -= 1
                if in_deg_for_rev[p] == 0:
                    q.append(p)

    desc_work = [0.0] * n
    for v in order:
        desc_work[v] = durations[v] + sum(desc_work[c] for c in children[v])
    return desc_work


def compute_ancestor_work(
    n: int,
    parents: List[List[int]],
    children: List[List[int]],
    durations: List[float],
) -> List[float]:
    """Total duration of all ancestors (inclusive of self)."""
    topo = topo_sort(n, parents, children)
    anc_work = [0.0] * n
    for v in topo:
        anc_work[v] = durations[v] + sum(anc_work[p] for p in parents[v])
    return anc_work


def compute_longest_path_from_root(
    n: int,
    parents: List[List[int]],
    children: List[List[int]],
    durations: List[float],
) -> List[float]:
    """Longest weighted path from any root to each node."""
    topo = topo_sort(n, parents, children)
    dist = [0.0] * n
    for v in topo:
        dist[v] = durations[v] + max((dist[p] for p in parents[v]), default=0.0)
    return dist


def compute_longest_path_to_leaf(
    n: int,
    parents: List[List[int]],
    children: List[List[int]],
    durations: List[float],
) -> List[float]:
    """Longest weighted path from each node to any leaf."""
    topo = topo_sort(n, parents, children)
    dist = [0.0] * n
    for v in reversed(topo):
        dist[v] = durations[v] + max((dist[c] for c in children[v]), default=0.0)
    return dist


def topo_edges_from_fx_graph(fx_graph) -> Tuple[List[Tuple[int, int]], List[str]]:
    nodes = list(fx_graph.nodes)
    name_list = [n.name for n in nodes]
    node_to_id = {n: i for i, n in enumerate(nodes)}
    edges: List[Tuple[int, int]] = []
    for v in nodes:
        v_id = node_to_id[v]
        for u in v.all_input_nodes:
            edges.append((node_to_id[u], v_id))
    return edges, name_list
