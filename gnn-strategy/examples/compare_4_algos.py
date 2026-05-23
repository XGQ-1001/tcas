"""Compare PyTorch, CUDA Graph, Opara, and GNN-PPO schedules.

This script is intended for paper figures:

1. Batch=1 comparison across multiple models:
   - speedup over PyTorch eager
   - latency in milliseconds
   - throughput in samples/sec

2. Inception-v3 batch-size sweep:
   - batch sizes 2, 4, 16, 32 (auto-includes batch=1 if available)
   - speedup over PyTorch eager
   - throughput vs batch size (line plot, scheduler-only view)
   - throughput speedup of GNN-PPO over Opara across batch sizes

Throughput is derived per trial as ``batch_size * 1000 / mean_ms`` so the
plotted standard deviation reflects measurement noise rather than a derived
quantity.

Example:
  python gnn-strategy/examples/compare_4_algos.py \
      --models resnet50,googlenet,inception_v3,deepfm,bert_base \
      --trials 10 --iterations 300 --warmups 30 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

_GNN_STRATEGY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_GNN_STRATEGY_DIR)
_TCAS_EXAMPLES_DIR = os.path.join(_REPO_ROOT, "examples")
for _p in (_GNN_STRATEGY_DIR, _REPO_ROOT, _TCAS_EXAMPLES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torchvision

from Opara import GraphCapturer
from Opara import OperatorLauncher

from gnn_strategy.capturer import benchmark_runner, capturer_gnn_from_fx
from gnn_strategy.env import SchedulingEnv
from gnn_strategy.graph_state import D_STATIC, build_graph_state
from gnn_strategy.policy import DynamicActorCritic
from gnn_strategy.utils import extract_first_fx_graph

warnings.filterwarnings("ignore", message=r"Trying to prepend a node to itself\..*", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torchvision\..*")


ALGO_ORDER = ["PyTorch-Eager", "CUDA-Graph", "Opara", "GNN-PPO"]
DEFAULT_MODELS = ["resnet50", "googlenet", "inception_v3", "deepfm", "bert_base"]
DISPLAY_NAMES = {
    "resnet50": "ResNet50",
    "googlenet": "GoogLeNet",
    "inception_v3": "Inception-v3",
    "deepfm": "DeepFM",
    "bert_base": "BERT-base",
}


@dataclass
class ModelSpec:
    model: nn.Module
    inputs: Tuple[torch.Tensor, ...]


def project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    norm = os.path.normpath(path)
    if norm == "artifacts" or norm.startswith(f"artifacts{os.sep}"):
        return os.path.join(_GNN_STRATEGY_DIR, norm)
    if norm == "gnn-strategy" or norm.startswith(f"gnn-strategy{os.sep}"):
        prefix = f"gnn-strategy{os.sep}"
        suffix = norm[len(prefix):] if norm.startswith(prefix) else ""
        return os.path.join(_GNN_STRATEGY_DIR, suffix)
    return os.path.abspath(path)


def make_googlenet(device: str, batch_size: int = 1) -> ModelSpec:
    x = torch.randint(0, 256, (batch_size, 3, 224, 224), dtype=torch.float32, device=device)
    model = torchvision.models.googlenet().to(device=device).eval()
    return ModelSpec(model, (x,))


def make_inception_v3(device: str, batch_size: int = 1) -> ModelSpec:
    x = torch.randint(0, 256, (batch_size, 3, 299, 299), dtype=torch.float32, device=device)
    model = torchvision.models.inception_v3(aux_logits=False).to(device=device).eval()
    return ModelSpec(model, (x,))


def make_resnet50(device: str, batch_size: int = 1) -> ModelSpec:
    x = torch.randint(0, 256, (batch_size, 3, 224, 224), dtype=torch.float32, device=device)
    model = torchvision.models.resnet50().to(device=device).eval()
    return ModelSpec(model, (x,))


def make_deepfm(device: str, batch_size: int = 1) -> ModelSpec:
    from NCF import DeepFM

    cate_fea_nuniqs = [100 * (i + 1) for i in range(32)]
    nume_fea_size = 16
    model = DeepFM(
        cate_fea_nuniqs,
        nume_fea_size,
        emb_size=8,
        hid_dims=[256, 128],
        num_classes=1,
        dropout=[0.2, 0.2],
    ).to(device=device).eval()
    x_sparse = torch.randint(
        0, 100, (batch_size, len(cate_fea_nuniqs)),
        dtype=torch.long, device=device,
    )
    x_dense = torch.rand(batch_size, nume_fea_size, device=device)
    return ModelSpec(model, (x_sparse, x_dense))


class BertLastHiddenState(nn.Module):
    def __init__(self, local_path: str):
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(local_path, local_files_only=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]


def make_bert_base(device: str, batch_size: int = 1) -> ModelSpec:
    local_path = os.environ.get(
        "GNN_BERT_BASE_PATH",
        "/mnt/workspace/xiaguoqing/models/bert-base-uncased",
    )
    seq_len = int(os.environ.get("GNN_BERT_SEQ_LEN", "256"))
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"BERT local path not found: {local_path}. "
            "Set GNN_BERT_BASE_PATH to a local HuggingFace BertModel directory."
        )
    input_ids = torch.randint(0, 30522, (batch_size, seq_len), dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    model = BertLastHiddenState(local_path).to(device=device).eval()
    return ModelSpec(model, (input_ids, attention_mask))


MODEL_FACTORIES = {
    "googlenet": make_googlenet,
    "inception_v3": make_inception_v3,
    "resnet50": make_resnet50,
    "deepfm": make_deepfm,
    "bert_base": make_bert_base,
}


DEFAULT_POLICY_PATHS = {
    "resnet50": "/mnt/workspace/xiaguoqing/x-ky/TCAS/gnn-data/resnet50/ResNet_v1/resnet50_real_final.pt",
    "googlenet": "/mnt/workspace/xiaguoqing/x-ky/TCAS/gnn-data/GoogLeNet/google-net-v3/googlenet_real_final.pt",
    "inception_v3": "/mnt/workspace/xiaguoqing/x-ky/TCAS/gnn-data/inception_v3/Inception3_v1/inception_v3_real_final.pt",
    "deepfm": "/mnt/workspace/xiaguoqing/x-ky/TCAS/gnn-data/DeepFM/deepfm_real_final.pt",
    "bert_base": "/mnt/workspace/xiaguoqing/x-ky/TCAS/gnn-data/BertLastHiddenState/bert_base_real_final.pt",
}


def capturer_eager(model: nn.Module, copy_outputs: bool = False):
    def run(*inputs):
        with torch.no_grad():
            out = model(*inputs)
        if copy_outputs:
            if isinstance(out, torch.Tensor):
                return out.clone()
            if isinstance(out, (list, tuple)):
                return [x.clone() if isinstance(x, torch.Tensor) else x for x in out]
        return out

    setattr(run, "_opara_meta", {"algorithm": "PyTorch-Eager"})
    return run


def capturer_cudagraph_serial(inputs: Sequence[torch.Tensor], model: nn.Module, copy_outputs: bool = False):
    static_inputs = [x.clone() for x in inputs]
    with torch.no_grad():
        for _ in range(3):
            model(*inputs)
    torch.cuda.synchronize()

    with torch.no_grad():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_outputs = model(*static_inputs)
    torch.cuda.synchronize()

    if not isinstance(static_outputs, (list, tuple)):
        static_outputs = (static_outputs,)

    def run(*new_inputs):
        assert len(static_inputs) == len(new_inputs)
        for dst, src in zip(static_inputs, new_inputs):
            dst.copy_(src)
        with torch.no_grad():
            graph.replay()
        if copy_outputs:
            return [x.clone() if isinstance(x, torch.Tensor) else x for x in static_outputs]
        return static_outputs

    setattr(run, "_opara_meta", {"algorithm": "CUDA-Graph"})
    return run


def load_policy(path: str, device: torch.device) -> DynamicActorCritic:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    saved_cfg = ckpt.get("config", None) if isinstance(ckpt, dict) else None
    hidden = getattr(saved_cfg, "hidden_dim", 256) if saved_cfg is not None else 256
    emb = getattr(saved_cfg, "emb_dim", 256) if saved_cfg is not None else 256
    n_heads = getattr(saved_cfg, "n_heads", 8) if saved_cfg is not None else 8
    policy = DynamicActorCritic(
        static_in_dim=D_STATIC,
        hidden_dim=hidden,
        emb_dim=emb,
        n_heads=n_heads,
        dropout=0.0,
    ).to(device)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy


def greedy_schedule_order(policy: DynamicActorCritic, gs, device: torch.device, n_streams: int) -> List[str]:
    env = SchedulingEnv(gs, n_streams=n_streams, device=device)
    env.reset()
    with torch.no_grad():
        h_static = policy.encode_static(gs.x.to(device), parents=gs.parents, children=gs.children)
    while not env.is_done():
        with torch.no_grad():
            dist, _ = policy.act(
                h_static,
                env.dynamic_node_features().to(device),
                env.global_features().to(device),
                env.ready_mask().to(device),
            )
        env.step(int(dist.probs.argmax().item()))
    order_ids = env.scheduled_order()
    return [gs.node_names[i] for i in order_ids if gs.movable_mask[i].item() == 1.0]


def build_gnn_runner(
    model_name: str,
    model: nn.Module,
    inputs: Tuple[torch.Tensor, ...],
    policy_path: str,
    device: torch.device,
    streams: int,
):
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"No GNN checkpoint for {model_name}: {policy_path}")
    policy = load_policy(policy_path, device)
    fx_module = extract_first_fx_graph(model, inputs)
    fx_module.cuda()
    node_profiles, device_props = OperatorLauncher.recompile(
        model.__class__.__name__,
        fx_module,
        inputs,
        apply_opara_schedule=False,
    )
    gs = build_graph_state(fx_module.graph, node_profiles=node_profiles, device_props=device_props)
    order_names = greedy_schedule_order(policy, gs, device=device, n_streams=streams)
    return capturer_gnn_from_fx(fx_module, inputs, schedule_order=order_names, copy_outputs=False)


def parse_policy_overrides(items: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Policy override must be model=path, got: {item}")
        model, path = item.split("=", 1)
        out[model.strip()] = project_path(path.strip())
    return out


def benchmark_algorithms(
    model_name: str,
    batch_size: int,
    policy_path: str,
    trials: int,
    iterations: int,
    warmups: int,
    streams: int,
    device: torch.device,
) -> List[Dict[str, object]]:
    spec = MODEL_FACTORIES[model_name](str(device), batch_size)
    model, inputs = spec.model, spec.inputs
    runners: Dict[str, Callable] = {
        "PyTorch-Eager": capturer_eager(model),
        "CUDA-Graph": capturer_cudagraph_serial(inputs, model),
        "Opara": GraphCapturer.capturer(inputs, model, use_tcas=False),
        "GNN-PPO": build_gnn_runner(model_name, model, inputs, policy_path, device, streams),
    }

    rows: List[Dict[str, object]] = []
    for algo in ALGO_ORDER:
        runner = runners[algo]
        print(f"[bench] model={model_name} batch={batch_size} algo={algo}")
        for trial in range(trials):
            br = benchmark_runner(runner, inputs=inputs, iterations=iterations, warmups=warmups)
            meta = getattr(runner, "_opara_meta", {}) if hasattr(runner, "_opara_meta") else {}
            throughput = (batch_size * 1000.0 / br.mean_ms) if br.mean_ms and br.mean_ms > 0 else 0.0
            rows.append({
                "model": model_name,
                "model_display": DISPLAY_NAMES.get(model_name, model_name),
                "batch_size": int(batch_size),
                "algo": algo,
                "trial": int(trial),
                "mean_ms": float(br.mean_ms),
                "std_ms": float(br.std_ms),
                "throughput_samples_per_sec": float(throughput),
                "num_nodes": int(meta.get("num_nodes", -1)) if isinstance(meta, dict) else -1,
                "num_streams": int(meta.get("num_streams", -1)) if isinstance(meta, dict) else -1,
                "policy_path": policy_path if algo == "GNN-PPO" else "",
            })
            print(
                f"  trial={trial:02d} mean={br.mean_ms:.4f}ms"
                f" std={br.std_ms:.4f}ms thpt={throughput:.1f} samples/s"
            )
    return rows


def write_rows(path: str, rows: List[Dict[str, object]], overwrite: bool):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if overwrite and os.path.exists(path):
        os.remove(path)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def read_rows(path: str) -> List[Dict[str, object]]:
    """Load benchmark rows from CSV; back-fill throughput if missing."""
    rows: List[Dict[str, object]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                batch_size = int(row["batch_size"])
                mean_ms = float(row["mean_ms"])
            except (KeyError, ValueError):
                continue
            row["batch_size"] = batch_size
            row["mean_ms"] = mean_ms
            try:
                row["trial"] = int(row.get("trial", 0))
            except ValueError:
                row["trial"] = 0
            try:
                row["std_ms"] = float(row.get("std_ms", 0.0))
            except ValueError:
                row["std_ms"] = 0.0
            existing = row.get("throughput_samples_per_sec", "")
            try:
                row["throughput_samples_per_sec"] = float(existing) if existing != "" else float("nan")
            except ValueError:
                row["throughput_samples_per_sec"] = float("nan")
            if not np.isfinite(row["throughput_samples_per_sec"]):
                row["throughput_samples_per_sec"] = (
                    batch_size * 1000.0 / mean_ms if mean_ms > 0 else 0.0
                )
            rows.append(row)
    return rows


def aggregate(
    rows: List[Dict[str, object]],
    column: str = "mean_ms",
) -> Dict[Tuple[str, int, str], Tuple[float, float]]:
    """Average ``column`` across trials for each (model, batch_size, algo) bucket.

    Returns a dict mapping the triple to ``(mean, std)``.
    """
    buckets: Dict[Tuple[str, int, str], List[float]] = {}
    for row in rows:
        if column not in row:
            continue
        try:
            value = float(row[column])
        except (TypeError, ValueError):
            continue
        key = (str(row["model"]), int(row["batch_size"]), str(row["algo"]))
        buckets.setdefault(key, []).append(value)
    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in buckets.items()}


def plot_batch1(rows: List[Dict[str, object]], models: List[str], out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = aggregate([r for r in rows if int(r["batch_size"]) == 1])
    labels = [DISPLAY_NAMES.get(m, m) for m in models]
    x = np.arange(len(models))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=220)
    for j, algo in enumerate(ALGO_ORDER):
        vals, errs = [], []
        for m in models:
            vals.append(agg.get((m, 1, algo), (np.nan, np.nan))[0])
            errs.append(agg.get((m, 1, algo), (np.nan, np.nan))[1])
        ax.bar(x + (j - 1.5) * width, vals, width, yerr=errs, capsize=3, label=algo)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Batch=1 Inference Latency")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"batch1_latency.{ext}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=220)
    all_latency_vals: List[float] = []
    for j, algo in enumerate(ALGO_ORDER):
        vals, errs = [], []
        for m in models:
            mean, std = agg.get((m, 1, algo), (np.nan, np.nan))
            vals.append(mean)
            errs.append(std)
            if np.isfinite(mean):
                all_latency_vals.append(float(mean))
        ax.bar(x + (j - 1.5) * width, vals, width, yerr=errs, capsize=3, label=algo)
    if all_latency_vals:
        lo, hi = min(all_latency_vals), max(all_latency_vals)
        pad = max((hi - lo) * 0.08, hi * 0.02)
        ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Batch=1 Inference Latency (Zoomed Y-axis)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"batch1_latency_zoomed.{ext}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=220)
    for j, algo in enumerate(ALGO_ORDER):
        vals = []
        for m in models:
            eager = agg.get((m, 1, "PyTorch-Eager"), (np.nan, np.nan))[0]
            cur = agg.get((m, 1, algo), (np.nan, np.nan))[0]
            vals.append(eager / cur if cur and cur > 0 else np.nan)
        ax.bar(x + (j - 1.5) * width, vals, width, label=algo)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup over PyTorch-Eager (x)")
    ax.set_title("Batch=1 Speedup Comparison")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"batch1_speedup.{ext}"))
    plt.close(fig)

    # Zoomed speedup around the scheduled execution backends. This keeps the
    # PyTorch-Eager baseline at 1x but avoids an excessively tall axis.
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=220)
    all_speedups: List[float] = []
    for j, algo in enumerate(ALGO_ORDER):
        vals = []
        for m in models:
            eager = agg.get((m, 1, "PyTorch-Eager"), (np.nan, np.nan))[0]
            cur = agg.get((m, 1, algo), (np.nan, np.nan))[0]
            v = eager / cur if cur and cur > 0 else np.nan
            vals.append(v)
            if np.isfinite(v):
                all_speedups.append(float(v))
        ax.bar(x + (j - 1.5) * width, vals, width, label=algo)
    ax.axhline(1.0, color="black", linewidth=0.8)
    finite_speedups = [v for v in all_speedups if np.isfinite(v)]
    if finite_speedups:
        lo, hi = min(finite_speedups), max(finite_speedups)
        pad = max((hi - lo) * 0.08, 0.05)
        ax.set_ylim(max(0.8, lo - pad), hi + pad)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup over PyTorch-Eager (x)")
    ax.set_title("Batch=1 Speedup Comparison (Zoomed Y-axis)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"batch1_speedup_zoomed.{ext}"))
    plt.close(fig)

    # Scheduler-only view: how much extra speedup GNN-PPO gets over Opara.
    fig, ax = plt.subplots(figsize=(9, 4.3), dpi=220)
    deltas = []
    for m in models:
        opara = agg.get((m, 1, "Opara"), (np.nan, np.nan))[0]
        gnn = agg.get((m, 1, "GNN-PPO"), (np.nan, np.nan))[0]
        deltas.append((opara - gnn) / opara * 100.0 if opara and opara > 0 else np.nan)
    colors = ["#388E3C" if np.isfinite(v) and v >= 0 else "#D32F2F" for v in deltas]
    ax.bar(x, deltas, width=0.55, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Extra Speedup over Opara (%)")
    ax.set_title("GNN-PPO Improvement over Opara")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"batch1_gnn_vs_opara_speedup.{ext}"))
    plt.close(fig)

    # Throughput (samples / sec) at batch=1. Mostly equivalent to 1/latency
    # but provides a more "system-oriented" axis label that paper readers
    # in the systems community are used to.
    thpt_agg = aggregate(
        [r for r in rows if int(r["batch_size"]) == 1],
        column="throughput_samples_per_sec",
    )
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=220)
    for j, algo in enumerate(ALGO_ORDER):
        vals, errs = [], []
        for m in models:
            mean, std = thpt_agg.get((m, 1, algo), (np.nan, np.nan))
            vals.append(mean)
            errs.append(std)
        ax.bar(x + (j - 1.5) * width, vals, width, yerr=errs, capsize=3, label=algo)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Throughput (samples / sec)")
    ax.set_title("Batch=1 Throughput")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"batch1_throughput.{ext}"))
    plt.close(fig)

    # No-DeepFM throughput chart. DeepFM has sub-millisecond latency and
    # therefore throughput on a totally different order of magnitude, which
    # squashes the other models in a single chart. This filtered version
    # keeps the y-axis on a comparable scale.
    models_nodeepfm = [m for m in models if m != "deepfm"]
    if models_nodeepfm:
        labels_nodeepfm = [DISPLAY_NAMES.get(m, m) for m in models_nodeepfm]
        x_nodeepfm = np.arange(len(models_nodeepfm))
        fig, ax = plt.subplots(figsize=(11, 4.5), dpi=220)
        for j, algo in enumerate(ALGO_ORDER):
            vals, errs = [], []
            for m in models_nodeepfm:
                mean, std = thpt_agg.get((m, 1, algo), (np.nan, np.nan))
                vals.append(mean)
                errs.append(std)
            ax.bar(x_nodeepfm + (j - 1.5) * width, vals, width, yerr=errs, capsize=3, label=algo)
        ax.set_xticks(x_nodeepfm)
        ax.set_xticklabels(labels_nodeepfm)
        ax.set_ylabel("Throughput (samples / sec)")
        ax.set_title("Batch=1 Throughput (no DeepFM)")
        ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        ax.legend(ncol=4, fontsize=9)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(out_dir, f"batch1_throughput_nodeepfm.{ext}"))
        plt.close(fig)


def plot_inception_sweep(rows: List[Dict[str, object]], batches: List[int], out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = aggregate([r for r in rows if str(r["model"]) == "inception_v3"])
    x = np.arange(len(batches))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=220)
    for j, algo in enumerate(ALGO_ORDER):
        vals = []
        for b in batches:
            eager = agg.get(("inception_v3", b, "PyTorch-Eager"), (np.nan, np.nan))[0]
            cur = agg.get(("inception_v3", b, algo), (np.nan, np.nan))[0]
            vals.append(eager / cur if cur and cur > 0 else np.nan)
        ax.bar(x + (j - 1.5) * width, vals, width, label=algo)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speedup over PyTorch-Eager (x)")
    ax.set_title("Inception-v3 Batch-size Speedup")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"inception_batch_sweep_speedup.{ext}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=220)
    all_speedups: List[float] = []
    for j, algo in enumerate(ALGO_ORDER):
        vals = []
        for b in batches:
            eager = agg.get(("inception_v3", b, "PyTorch-Eager"), (np.nan, np.nan))[0]
            cur = agg.get(("inception_v3", b, algo), (np.nan, np.nan))[0]
            v = eager / cur if cur and cur > 0 else np.nan
            vals.append(v)
            if np.isfinite(v):
                all_speedups.append(float(v))
        ax.bar(x + (j - 1.5) * width, vals, width, label=algo)
    ax.axhline(1.0, color="black", linewidth=0.8)
    finite_speedups = [v for v in all_speedups if np.isfinite(v)]
    if finite_speedups:
        lo, hi = min(finite_speedups), max(finite_speedups)
        pad = max((hi - lo) * 0.08, 0.04)
        ax.set_ylim(max(0.9, lo - pad), hi + pad)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speedup over PyTorch-Eager (x)")
    ax.set_title("Inception-v3 Batch-size Speedup (Zoomed Y-axis)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"inception_batch_sweep_speedup_zoomed.{ext}"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.3), dpi=220)
    deltas = []
    for b in batches:
        opara = agg.get(("inception_v3", b, "Opara"), (np.nan, np.nan))[0]
        gnn = agg.get(("inception_v3", b, "GNN-PPO"), (np.nan, np.nan))[0]
        deltas.append((opara - gnn) / opara * 100.0 if opara and opara > 0 else np.nan)
    colors = ["#388E3C" if np.isfinite(v) and v >= 0 else "#D32F2F" for v in deltas]
    ax.bar(x, deltas, width=0.55, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Extra Speedup over Opara (%)")
    ax.set_title("Inception-v3: GNN-PPO Improvement over Opara")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"inception_gnn_vs_opara_speedup.{ext}"))
    plt.close(fig)

    # ---- Throughput vs batch-size line plots ----
    # Use ALL Inception-v3 rows we have, including batch=1 from the cross-model
    # benchmark. Sort batch sizes ascending so the line is well-defined.
    inception_rows = [r for r in rows if str(r["model"]) == "inception_v3"]
    thpt_agg = aggregate(inception_rows, column="throughput_samples_per_sec")
    all_batches = sorted({int(r["batch_size"]) for r in inception_rows})

    algo_colors = {
        "PyTorch-Eager": "#666666",
        "CUDA-Graph": "#1976D2",
        "Opara": "#F57C00",
        "GNN-PPO": "#2E7D32",
    }
    algo_markers = {
        "PyTorch-Eager": "s",
        "CUDA-Graph": "D",
        "Opara": "^",
        "GNN-PPO": "o",
    }

    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=220)
    for algo in ALGO_ORDER:
        means, stds, xs = [], [], []
        for b in all_batches:
            mean, std = thpt_agg.get(("inception_v3", b, algo), (np.nan, np.nan))
            if not np.isfinite(mean):
                continue
            xs.append(b)
            means.append(mean)
            stds.append(std)
        if not xs:
            continue
        ax.errorbar(
            xs, means, yerr=stds, marker=algo_markers[algo], markersize=6,
            linewidth=2.0, capsize=3, label=algo, color=algo_colors[algo],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_batches)
    ax.set_xticklabels([str(b) for b in all_batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (samples / sec)")
    ax.set_title("Inception-v3 Throughput vs Batch Size")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"inception_throughput_vs_batch.{ext}"))
    plt.close(fig)

    # Scheduler-only view: drop the PyTorch-Eager line so the y-axis zooms onto
    # the schedulers (CUDA-Graph / Opara / GNN-PPO) where the interesting
    # differences live.
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=220)
    for algo in ("CUDA-Graph", "Opara", "GNN-PPO"):
        means, stds, xs = [], [], []
        for b in all_batches:
            mean, std = thpt_agg.get(("inception_v3", b, algo), (np.nan, np.nan))
            if not np.isfinite(mean):
                continue
            xs.append(b)
            means.append(mean)
            stds.append(std)
        if not xs:
            continue
        ax.errorbar(
            xs, means, yerr=stds, marker=algo_markers[algo], markersize=6,
            linewidth=2.0, capsize=3, label=algo, color=algo_colors[algo],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(all_batches)
    ax.set_xticklabels([str(b) for b in all_batches])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (samples / sec)")
    ax.set_title("Inception-v3 Throughput vs Batch Size (Schedulers Only)")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"inception_throughput_vs_batch_zoomed.{ext}"))
    plt.close(fig)

    # Relative throughput gain of GNN-PPO over Opara as batch grows. This is
    # the story plot: if the gap widens with batch size, we have a strong
    # narrative even though batch=1 differences look small.
    fig, ax = plt.subplots(figsize=(8.5, 4.3), dpi=220)
    gain_xs, gain_vals = [], []
    for b in all_batches:
        opara = thpt_agg.get(("inception_v3", b, "Opara"), (np.nan, np.nan))[0]
        gnn = thpt_agg.get(("inception_v3", b, "GNN-PPO"), (np.nan, np.nan))[0]
        if not (np.isfinite(opara) and np.isfinite(gnn) and opara > 0):
            continue
        gain_xs.append(b)
        gain_vals.append((gnn - opara) / opara * 100.0)
    if gain_xs:
        ax.plot(
            gain_xs, gain_vals, marker="o", markersize=7, linewidth=2.2,
            color="#2E7D32", label="GNN-PPO vs Opara",
        )
        for bx, v in zip(gain_xs, gain_vals):
            ax.annotate(f"{v:+.1f}%", xy=(bx, v), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8.5)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(gain_xs or all_batches)
    ax.set_xticklabels([str(b) for b in (gain_xs or all_batches)])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput gain over Opara (%)")
    ax.set_title("Inception-v3: GNN-PPO Extra Throughput over Opara")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"inception_throughput_speedup_vs_opara.{ext}"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare PyTorch/CUDA-Graph/Opara/GNN-PPO algorithms.")
    parser.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    parser.add_argument("--batches", type=str, default="2,4,16,32",
                        help="Batch sizes for the Inception-v3 sweep.")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--warmups", type=int, default=30)
    parser.add_argument("--streams", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="artifacts/compare_4_algos")
    parser.add_argument("--policy", action="append", default=[],
                        help="Override a default checkpoint path, format: model=/path/to/policy.pt")
    parser.add_argument("--skip-batch1", action="store_true")
    parser.add_argument("--skip-inception-sweep", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--replot-only", action="store_true",
        help="Skip benchmarking and re-plot from the existing CSV at --out-dir.",
    )
    args = parser.parse_args()

    if args.replot_only:
        out_dir = project_path(args.out_dir)
        csv_path = os.path.join(out_dir, "compare_4_algos.csv")
        if not os.path.exists(csv_path):
            raise SystemExit(f"--replot-only requires existing CSV: {csv_path}")
        rows = read_rows(csv_path)
        if not rows:
            raise SystemExit(f"No rows loaded from {csv_path}.")
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        models_present = [m for m in models if any(r["model"] == m and int(r["batch_size"]) == 1 for r in rows)]
        batches_present = sorted({int(r["batch_size"]) for r in rows
                                   if str(r["model"]) == "inception_v3" and int(r["batch_size"]) > 1})
        if not args.skip_batch1 and models_present:
            plot_batch1(rows, models=models_present, out_dir=out_dir)
        if not args.skip_inception_sweep and batches_present:
            plot_inception_sweep(rows, batches=batches_present, out_dir=out_dir)
        print(f"\nReplotted from: {csv_path}")
        print(f"  out_dir: {out_dir}")
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required for this benchmark.")

    out_dir = project_path(args.out_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    batches = [int(x.strip()) for x in args.batches.split(",") if x.strip()]
    policy_paths = dict(DEFAULT_POLICY_PATHS)
    policy_paths.update(parse_policy_overrides(args.policy))

    for m in models:
        if m not in MODEL_FACTORIES:
            raise SystemExit(f"Unknown model: {m}. Choices: {sorted(MODEL_FACTORIES)}")
        if not os.path.exists(policy_paths.get(m, "")):
            raise SystemExit(f"Missing default GNN checkpoint for {m}: {policy_paths.get(m)}")

    device = torch.device("cuda")
    all_rows: List[Dict[str, object]] = []

    if not args.skip_batch1:
        for m in models:
            all_rows.extend(benchmark_algorithms(
                model_name=m,
                batch_size=1,
                policy_path=policy_paths[m],
                trials=args.trials,
                iterations=args.iterations,
                warmups=args.warmups,
                streams=args.streams,
                device=device,
            ))

    if not args.skip_inception_sweep:
        for b in batches:
            all_rows.extend(benchmark_algorithms(
                model_name="inception_v3",
                batch_size=b,
                policy_path=policy_paths["inception_v3"],
                trials=args.trials,
                iterations=args.iterations,
                warmups=args.warmups,
                streams=args.streams,
                device=device,
            ))

    if not all_rows:
        raise SystemExit("No benchmark rows produced.")

    csv_path = os.path.join(out_dir, "compare_4_algos.csv")
    write_rows(csv_path, all_rows, overwrite=args.overwrite)
    if not args.skip_batch1:
        plot_batch1(all_rows, models=models, out_dir=out_dir)
    if not args.skip_inception_sweep:
        plot_inception_sweep(all_rows, batches=batches, out_dir=out_dir)

    print("\nSaved:")
    print(f"  CSV: {csv_path}")
    if not args.skip_batch1:
        print(f"  Batch=1 latency:    {os.path.join(out_dir, 'batch1_latency.png')}")
        print(f"  Batch=1 speedup:    {os.path.join(out_dir, 'batch1_speedup.png')}")
        print(f"  Batch=1 throughput: {os.path.join(out_dir, 'batch1_throughput.png')}")
    if not args.skip_inception_sweep:
        print(f"  Inception speedup:  {os.path.join(out_dir, 'inception_batch_sweep_speedup.png')}")
        print(f"  Inception thpt:     {os.path.join(out_dir, 'inception_throughput_vs_batch.png')}")
        print(f"  Inception thpt (Z): {os.path.join(out_dir, 'inception_throughput_vs_batch_zoomed.png')}")
        print(f"  Inception thpt vs Opara: {os.path.join(out_dir, 'inception_throughput_speedup_vs_opara.png')}")


if __name__ == "__main__":
    main()
