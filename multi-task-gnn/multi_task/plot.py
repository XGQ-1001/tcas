"""Multi-task 实验结果可视化。

============================================================================
绘制的图：
  1. training_curves.png    PPO 训练曲线 (速度提升 / PPO loss / entropy)
  2. baseline_compare.png   GNN vs 基线的 makespan 对比柱状图
  3. k_generalization.png   不同 K 值下各算法的 makespan / 加速比曲线
============================================================================
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _moving_average(arr: List[float], window: int = 10) -> np.ndarray:
    if len(arr) == 0:
        return np.array([])
    w = min(window, len(arr))
    cumsum = np.cumsum(np.array(arr, dtype=np.float64))
    ma = np.empty(len(arr), dtype=np.float64)
    for i in range(len(arr)):
        lo = max(0, i - w + 1)
        ma[i] = (cumsum[i] - (cumsum[lo - 1] if lo > 0 else 0.0)) / (i - lo + 1)
    return ma


def plot_training_curves(
    history: Dict,
    save_path: str,
    title_suffix: str = '',
) -> None:
    """PPO 训练曲线：3 子图横排。

    a) Makespan & Speedup vs Episode
    b) PPO pg_loss / v_loss vs gradient step
    c) Policy entropy vs gradient step
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    episodes = history.get('episodes', [])
    grad_steps = history.get('grad_steps', [])
    L_base = history.get('L_baseline', None)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))

    # ---- (a) Makespan curves ----
    ax = axes[0]
    if episodes:
        eps = [r['episode'] for r in episodes]
        L_gnn = [r['L_gnn'] for r in episodes]
        best = [r['best_L'] for r in episodes]
        ax.plot(eps, L_gnn, alpha=0.35, color='tab:blue', label='GNN (current)', lw=0.8)
        ax.plot(eps, _moving_average(L_gnn, 20), color='tab:blue', label='GNN (MA-20)', lw=1.8)
        ax.plot(eps, best, color='tab:green', lw=2, label='GNN (best so far)')
        if L_base is not None:
            ax.axhline(L_base, color='tab:red', ls='--', lw=1.5,
                       label=f'Opara-like baseline ({L_base:.1f})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Simulated Makespan')
    ax.set_title(f'(a) Makespan Curve {title_suffix}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---- (b) PPO losses ----
    ax = axes[1]
    if grad_steps:
        steps = [s['global_step'] for s in grad_steps]
        pg = [s['pg_loss'] for s in grad_steps]
        vl = [s['v_loss'] for s in grad_steps]
        ax.plot(steps, pg, alpha=0.25, color='tab:orange', lw=0.6)
        ax.plot(steps, _moving_average(pg, 30), color='tab:orange', lw=1.8, label='pg_loss (MA-30)')
        ax2 = ax.twinx()
        ax2.plot(steps, vl, alpha=0.25, color='tab:purple', lw=0.6)
        ax2.plot(steps, _moving_average(vl, 30), color='tab:purple', lw=1.8, label='v_loss (MA-30)')
        ax.set_ylabel('Policy Gradient Loss', color='tab:orange')
        ax2.set_ylabel('Value Loss', color='tab:purple')
        ax2.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('Gradient Step')
    ax.set_title(f'(b) PPO Losses {title_suffix}')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3)

    # ---- (c) Policy entropy ----
    ax = axes[2]
    if grad_steps:
        steps = [s['global_step'] for s in grad_steps]
        ent = [s['entropy'] for s in grad_steps]
        ax.plot(steps, ent, alpha=0.25, color='tab:blue', lw=0.6)
        ax.plot(steps, _moving_average(ent, 30), color='tab:blue', lw=1.8, label='entropy (MA-30)')
    ax.set_xlabel('Gradient Step')
    ax.set_ylabel('Policy Entropy')
    ax.set_title(f'(c) Policy Entropy {title_suffix}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"  [plot] saved: {save_path}")


def plot_baseline_compare(
    results: Dict[str, float],
    save_path: str,
    title: str = 'GNN vs Baselines (simulated makespan)',
    baseline_key: str = 'Opara-like',
) -> None:
    """各调度算法的 makespan 对比柱状图 + 相对 baseline 的加速比。"""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    algos = list(results.keys())
    vals = [results[a] for a in algos]

    base = results.get(baseline_key, min(vals))
    speedups = [(base - v) / base * 100.0 for v in vals]

    colors = ['tab:gray'] * len(algos)
    for i, a in enumerate(algos):
        if a == baseline_key:
            colors[i] = 'tab:red'
        elif 'GNN' in a:
            colors[i] = 'tab:green'

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    bars = ax.bar(algos, vals, color=colors, edgecolor='black', lw=0.7)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.1f}',
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Simulated Makespan')
    ax.set_title(f'(a) {title}')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.3)

    ax = axes[1]
    bars = ax.bar(algos, speedups, color=colors, edgecolor='black', lw=0.7)
    for b, v in zip(bars, speedups):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:+.2f}%',
                ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
    ax.axhline(0.0, color='black', lw=0.8)
    ax.set_ylabel(f'Speedup vs {baseline_key} (%)')
    ax.set_title(f'(b) Speedup over {baseline_key}')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"  [plot] saved: {save_path}")


def plot_k_generalization(
    k_results: Dict[int, Dict[str, float]],
    save_path: str,
    title: str = 'K-Generalization',
    baseline_key: str = 'Opara-like',
) -> None:
    """不同 K 值下各算法的 makespan + 相对基线的加速比。

    k_results: {K: {algo_name: makespan}}
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    Ks = sorted(k_results.keys())
    all_algos: List[str] = []
    for K in Ks:
        for a in k_results[K].keys():
            if a not in all_algos:
                all_algos.append(a)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    ax = axes[0]
    for a in all_algos:
        vals = [k_results[K].get(a, np.nan) for K in Ks]
        marker = 'o'
        lw = 1.5
        if a == baseline_key:
            color = 'tab:red'
            lw = 2
        elif 'GNN' in a:
            color = 'tab:green'
            marker = 's'
            lw = 2
        else:
            color = None
        ax.plot(Ks, vals, marker=marker, lw=lw, label=a, color=color)
    ax.set_xlabel('K (number of concurrent tasks)')
    ax.set_ylabel('Simulated Makespan')
    ax.set_title(f'(a) Makespan vs K — {title}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(Ks)

    ax = axes[1]
    for a in all_algos:
        vals = []
        for K in Ks:
            base_v = k_results[K].get(baseline_key, np.nan)
            v = k_results[K].get(a, np.nan)
            if base_v and not np.isnan(base_v) and not np.isnan(v):
                vals.append((base_v - v) / base_v * 100.0)
            else:
                vals.append(np.nan)
        marker = 'o'
        lw = 1.5
        if a == baseline_key:
            color = 'tab:red'
            lw = 2
        elif 'GNN' in a:
            color = 'tab:green'
            marker = 's'
            lw = 2
        else:
            color = None
        ax.plot(Ks, vals, marker=marker, lw=lw, label=a, color=color)
    ax.axhline(0.0, color='black', lw=0.8)
    ax.set_xlabel('K (number of concurrent tasks)')
    ax.set_ylabel(f'Speedup vs {baseline_key} (%)')
    ax.set_title(f'(b) Speedup vs K — {title}')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(Ks)

    plt.tight_layout()
    plt.savefig(save_path, dpi=140)
    plt.close(fig)
    print(f"  [plot] saved: {save_path}")
