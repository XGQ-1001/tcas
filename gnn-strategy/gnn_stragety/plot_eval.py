"""评估结果绘图 — 多算法对比的柱状图（延迟 & 加速比）。

本模块用于评估阶段（evaluate.py），将不同算法（Opara / TCAS / GNN）
在多个模型上的延迟结果绘制为对比柱状图。

与 plot_training.py（训练过程曲线）区分。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import csv
import os

import numpy as np


@dataclass
class Agg:
    mean: float
    std: float


def read_eval_csv(path: str) -> List[Dict]:
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        return [row for row in r]


def aggregate(rows: List[Dict]) -> Dict[Tuple[str, str], Agg]:
    """Aggregate by (model, algo) across trials using the per-trial mean_ms."""
    buckets: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        model = str(row['model'])
        algo = str(row['algo'])
        v = float(row['mean_ms'])
        buckets.setdefault((model, algo), []).append(v)

    out: Dict[Tuple[str, str], Agg] = {}
    for k, vals in buckets.items():
        out[k] = Agg(mean=float(np.mean(vals)), std=float(np.std(vals)))
    return out


def plot_latency_and_speedup(
    csv_path: str,
    out_latency_png: str,
    out_speedup_png: str,
    model_order: List[str],
    algo_order: List[str],
    baseline_algo: str = 'Opara',
):
    """Make two plots:
    1) latency bar chart with error bars
    2) speedup vs baseline bar chart

    Requires matplotlib at runtime.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib is required for plotting. Install it via `pip install matplotlib` "
            f"(original error: {e})"
        )

    rows = read_eval_csv(csv_path)
    agg = aggregate(rows)

    # Prepare matrices
    lat_mean = np.zeros((len(model_order), len(algo_order)), dtype=np.float64)
    lat_std = np.zeros_like(lat_mean)

    for i, m in enumerate(model_order):
        for j, a in enumerate(algo_order):
            key = (m, a)
            if key in agg:
                lat_mean[i, j] = agg[key].mean
                lat_std[i, j] = agg[key].std
            else:
                lat_mean[i, j] = np.nan
                lat_std[i, j] = np.nan

    # --- Plot 1: latency ---
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
    x = np.arange(len(model_order))
    width = 0.22

    for j, algo in enumerate(algo_order):
        ax.bar(
            x + (j - (len(algo_order) - 1) / 2) * width,
            lat_mean[:, j],
            width,
            yerr=lat_std[:, j],
            capsize=3,
            label=algo,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=0)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('End-to-End Inference Latency')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.legend(ncol=len(algo_order), fontsize=9)

    os.makedirs(os.path.dirname(out_latency_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_latency_png)
    plt.close(fig)

    # --- Plot 2: speedup vs baseline ---
    if baseline_algo not in algo_order:
        raise ValueError(f"baseline_algo {baseline_algo} not in algo_order")

    bidx = algo_order.index(baseline_algo)
    base = lat_mean[:, bidx]

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=160)
    x = np.arange(len(model_order))

    # Only plot non-baseline algos
    others = [a for a in algo_order if a != baseline_algo]
    width = 0.28 if len(others) == 2 else 0.22

    for j, algo in enumerate(others):
        j_src = algo_order.index(algo)
        s = (base - lat_mean[:, j_src]) / base * 100.0
        ax.bar(
            x + (j - (len(others) - 1) / 2) * width,
            s,
            width,
            label=f"{algo} vs {baseline_algo}",
        )

    ax.axhline(0.0, color='black', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(model_order, rotation=0)
    ax.set_ylabel('Speedup (%)')
    ax.set_title(f'Speedup vs {baseline_algo}')
    ax.grid(True, axis='y', linestyle='--', alpha=0.3)
    ax.legend(ncol=len(others), fontsize=9)

    os.makedirs(os.path.dirname(out_speedup_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_speedup_png)
    plt.close(fig)
