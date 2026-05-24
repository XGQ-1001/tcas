"""训练曲线绘图 — 生成论文级质量的 PPO 训练过程可视化图表。

本模块从 train.py 中提取出来，专注于训练阶段的曲线绘制。
与 plot_eval.py（评估阶段的多算法对比柱状图）区分。

生成的图表：
  1. latency_curve     — GNN延迟 vs Opara基线 随 episode 变化
  2. speedup_curve     — 加速比(%) 随 episode 变化
  3. ppo_losses        — 策略损失 / 价值损失 / 熵 三合一
  4. bc_loss           — BC 预训练损失曲线（如果有 BC 阶段）
  5. summary_combined  — 2×2 合并大图（方便论文一张图展示）
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np


def plot_training_curves(
    history: Dict,
    save_dir: str,
    model_name: str = '',
    dpi: int = 300,
) -> List[str]:
    """根据训练 history 绘制论文级训练曲线，保存为 PNG/PDF。

    生成的图表：
      1. latency_curve     — GNN延迟 vs Opara基线 随 episode 变化
      2. speedup_curve     — 加速比(%) 随 episode 变化
      3. ppo_losses        — 策略损失 / 价值损失 / 熵 三合一
      4. bc_loss           — BC 预训练损失曲线（如果有 BC 阶段）
      5. summary_combined  — 2×2 合并大图（方便论文一张图展示）

    参数:
        history:    train_policy_real() 返回的 history dict
        save_dir:   图表保存目录
        model_name: 模型名称（用于图表标题）
        dpi:        输出分辨率，默认 300（论文级）

    返回:
        保存的文件路径列表
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("  [plot] matplotlib not available, skipping plots")
        return []

    os.makedirs(save_dir, exist_ok=True)
    saved_files: List[str] = []

    episodes_data = history.get('episodes', [])
    if not episodes_data:
        print("  [plot] No episode data to plot")
        return []

    # ---- 提取数据 ----
    eps = [r['episode'] for r in episodes_data]
    L_opara = history.get('L_opara_ms', episodes_data[0].get('L_opara_ms', 0))

    has_real = 'L_gnn_ms' in episodes_data[0]
    if has_real:
        L_gnns = [r['L_gnn_ms'] for r in episodes_data]
        speedups = [r['speedup_pct'] for r in episodes_data]
        best_speedups = [r['best_speedup_pct'] for r in episodes_data]

    pg_losses = [r.get('pg_loss', 0) for r in episodes_data]
    v_losses = [r.get('v_loss', 0) for r in episodes_data]
    entropies = [r.get('entropy', 0) for r in episodes_data]

    # 细粒度 per-mini-batch 梯度步数据（如果可用则使用，曲线更密集）
    grad_steps = history.get('grad_steps', [])
    has_grad_steps = len(grad_steps) > 5

    bc_losses = history.get('bc_losses', [])

    title_prefix = f'{model_name} ' if model_name else ''

    # ---- 论文风格设置 ----
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'legend.fontsize': 11,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'lines.linewidth': 1.5,
        'figure.dpi': dpi,
        'savefig.dpi': dpi,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })

    BLUE = '#2176AE'
    RED = '#D32F2F'
    GREEN = '#388E3C'
    ORANGE = '#F57C00'
    PURPLE = '#7B1FA2'

    # ==================================================================
    # 图1: 延迟曲线 — GNN vs Opara
    # ==================================================================
    if has_real:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(eps, L_gnns, color=BLUE, alpha=0.4, linewidth=0.8, label='GNN (per episode)')

        window = max(1, len(eps) // 20)
        if len(L_gnns) > window:
            smooth = np.convolve(L_gnns, np.ones(window)/window, mode='valid')
            smooth_eps = eps[window-1:]
            ax.plot(smooth_eps, smooth, color=BLUE, linewidth=2.0, label=f'GNN (moving avg, w={window})')

        running_best = []
        cur_best = float('inf')
        for v in L_gnns:
            cur_best = min(cur_best, v)
            running_best.append(cur_best)
        ax.plot(eps, running_best, color=GREEN, linewidth=2.0, linestyle='--', label='GNN (best so far)')

        ax.axhline(y=L_opara, color=RED, linewidth=2.0, linestyle=':', label=f'Opara baseline ({L_opara:.4f} ms)')

        ax.set_xlabel('Episode')
        ax.set_ylabel('Latency (ms)')
        ax.set_title(f'{title_prefix}Inference Latency During Training')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        for fmt in ('png', 'pdf'):
            path = os.path.join(save_dir, f'latency_curve.{fmt}')
            fig.savefig(path)
            saved_files.append(path)
        plt.close(fig)

    # ==================================================================
    # 图2: 加速比曲线
    # ==================================================================
    if has_real:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(eps, speedups, color=BLUE, alpha=0.4, linewidth=0.8, label='Speedup (per episode)')
        ax.plot(eps, best_speedups, color=GREEN, linewidth=2.0, linestyle='--', label='Best speedup so far')

        ax.axhline(y=0, color=RED, linewidth=1.5, linestyle=':', label='Opara baseline (0%)')

        ax.fill_between(eps, speedups, 0,
                        where=[s > 0 for s in speedups],
                        color=GREEN, alpha=0.08, interpolate=True)
        ax.fill_between(eps, speedups, 0,
                        where=[s < 0 for s in speedups],
                        color=RED, alpha=0.08, interpolate=True)

        ax.set_xlabel('Episode')
        ax.set_ylabel('Speedup over Opara (%)')
        ax.set_title(f'{title_prefix}Speedup over Opara During Training')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        for fmt in ('png', 'pdf'):
            path = os.path.join(save_dir, f'speedup_curve.{fmt}')
            fig.savefig(path)
            saved_files.append(path)
        plt.close(fig)

    # ==================================================================
    # 图3: PPO 训练诊断（策略损失 / 价值损失 / 熵）
    # 优先使用 per-mini-batch 梯度步数据（更密集），否则回退到 per-episode
    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    if has_grad_steps:
        gs_x = [s['global_step'] for s in grad_steps]
        gs_pg = [s['pg_loss'] for s in grad_steps]
        gs_v = [s['v_loss'] for s in grad_steps]
        gs_ent = [s['entropy'] for s in grad_steps]

        smooth_w = max(1, len(gs_x) // 40)

        axes[0].plot(gs_x, gs_pg, color=BLUE, alpha=0.25, linewidth=0.6)
        if len(gs_pg) > smooth_w:
            sm = np.convolve(gs_pg, np.ones(smooth_w)/smooth_w, mode='valid')
            axes[0].plot(gs_x[smooth_w-1:], sm, color=BLUE, linewidth=1.5, label=f'smoothed (w={smooth_w})')
            axes[0].legend(fontsize=9)
        axes[0].set_xlabel('Gradient Step')
        axes[0].set_ylabel('Policy Loss')
        axes[0].set_title('Policy Gradient Loss')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(gs_x, gs_v, color=ORANGE, alpha=0.25, linewidth=0.6)
        if len(gs_v) > smooth_w:
            sm = np.convolve(gs_v, np.ones(smooth_w)/smooth_w, mode='valid')
            axes[1].plot(gs_x[smooth_w-1:], sm, color=ORANGE, linewidth=1.5, label=f'smoothed (w={smooth_w})')
            axes[1].legend(fontsize=9)
        axes[1].set_xlabel('Gradient Step')
        axes[1].set_ylabel('Value Loss')
        axes[1].set_title('Value Function Loss')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(gs_x, gs_ent, color=PURPLE, alpha=0.25, linewidth=0.6)
        if len(gs_ent) > smooth_w:
            sm = np.convolve(gs_ent, np.ones(smooth_w)/smooth_w, mode='valid')
            axes[2].plot(gs_x[smooth_w-1:], sm, color=PURPLE, linewidth=1.5, label=f'smoothed (w={smooth_w})')
            axes[2].legend(fontsize=9)
        axes[2].set_xlabel('Gradient Step')
        axes[2].set_ylabel('Entropy')
        axes[2].set_title('Policy Entropy')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[0].plot(eps, pg_losses, color=BLUE, linewidth=1.2)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Policy Loss')
        axes[0].set_title('Policy Gradient Loss')
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(eps, v_losses, color=ORANGE, linewidth=1.2)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Value Loss')
        axes[1].set_title('Value Function Loss')
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(eps, entropies, color=PURPLE, linewidth=1.2)
        axes[2].set_xlabel('Episode')
        axes[2].set_ylabel('Entropy')
        axes[2].set_title('Policy Entropy')
        axes[2].grid(True, alpha=0.3)

    fig.suptitle(f'{title_prefix}PPO Training Diagnostics', fontsize=14, y=1.02)
    fig.tight_layout()

    for fmt in ('png', 'pdf'):
        path = os.path.join(save_dir, f'ppo_losses.{fmt}')
        fig.savefig(path)
        saved_files.append(path)
    plt.close(fig)

    # ==================================================================
    # 图4: BC 预训练损失（如果有的话）
    # ==================================================================
    if bc_losses:
        fig, ax = plt.subplots(figsize=(6, 4))
        bc_eps = list(range(len(bc_losses)))
        ax.plot(bc_eps, bc_losses, color=BLUE, linewidth=1.5, marker='o', markersize=3)
        ax.set_xlabel('BC Episode')
        ax.set_ylabel('Imitation Loss')
        ax.set_title(f'{title_prefix}Behaviour Cloning Pre-training Loss')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)

        for fmt in ('png', 'pdf'):
            path = os.path.join(save_dir, f'bc_loss.{fmt}')
            fig.savefig(path)
            saved_files.append(path)
        plt.close(fig)

    # ==================================================================
    # 图5: 2×2 合并大图（论文中一张图展示所有信息）
    # ==================================================================
    if has_real:
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        ax = axes[0, 0]
        ax.plot(eps, L_gnns, color=BLUE, alpha=0.35, linewidth=0.7)
        if len(L_gnns) > window:
            ax.plot(smooth_eps, smooth, color=BLUE, linewidth=1.8, label='GNN (smoothed)')
        ax.plot(eps, running_best, color=GREEN, linewidth=1.8, linestyle='--', label='GNN (best)')
        ax.axhline(y=L_opara, color=RED, linewidth=1.8, linestyle=':', label=f'Opara ({L_opara:.4f} ms)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Latency (ms)')
        ax.set_title('(a) Inference Latency')
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(eps, speedups, color=BLUE, alpha=0.35, linewidth=0.7)
        ax.plot(eps, best_speedups, color=GREEN, linewidth=1.8, linestyle='--', label='Best speedup')
        ax.axhline(y=0, color=RED, linewidth=1.2, linestyle=':')
        ax.fill_between(eps, speedups, 0,
                        where=[s > 0 for s in speedups],
                        color=GREEN, alpha=0.06, interpolate=True)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Speedup (%)')
        ax.set_title('(b) Speedup over Opara')
        ax.legend(fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if has_grad_steps:
            gs_x = [s['global_step'] for s in grad_steps]
            gs_pg = [s['pg_loss'] for s in grad_steps]
            gs_v = [s['v_loss'] for s in grad_steps]
            sm_w = max(1, len(gs_x) // 40)

            ax.plot(gs_x, gs_pg, color=BLUE, alpha=0.2, linewidth=0.5)
            ax2 = ax.twinx()
            ax2.plot(gs_x, gs_v, color=ORANGE, alpha=0.2, linewidth=0.5)
            if len(gs_pg) > sm_w:
                sm_pg = np.convolve(gs_pg, np.ones(sm_w)/sm_w, mode='valid')
                sm_v = np.convolve(gs_v, np.ones(sm_w)/sm_w, mode='valid')
                ax.plot(gs_x[sm_w-1:], sm_pg, color=BLUE, linewidth=1.5, label='Policy loss')
                ax2.plot(gs_x[sm_w-1:], sm_v, color=ORANGE, linewidth=1.5, label='Value loss')
            ax.set_xlabel('Gradient Step')
        else:
            ax.plot(eps, pg_losses, color=BLUE, linewidth=1.2, label='Policy loss')
            ax2 = ax.twinx()
            ax2.plot(eps, v_losses, color=ORANGE, linewidth=1.2, label='Value loss')
            ax.set_xlabel('Episode')
        ax.set_ylabel('Policy Loss', color=BLUE)
        ax2.set_ylabel('Value Loss', color=ORANGE)
        ax.set_title('(c) PPO Losses')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        if has_grad_steps:
            gs_ent = [s['entropy'] for s in grad_steps]
            ax.plot(gs_x, gs_ent, color=PURPLE, alpha=0.2, linewidth=0.5)
            if len(gs_ent) > sm_w:
                sm_ent = np.convolve(gs_ent, np.ones(sm_w)/sm_w, mode='valid')
                ax.plot(gs_x[sm_w-1:], sm_ent, color=PURPLE, linewidth=1.5, label='smoothed')
                ax.legend(fontsize=9)
            ax.set_xlabel('Gradient Step')
        else:
            ax.plot(eps, entropies, color=PURPLE, linewidth=1.2)
            ax.set_xlabel('Episode')
        ax.set_ylabel('Entropy')
        ax.set_title('(d) Policy Entropy')
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'{title_prefix}GNN-RL Scheduling — Training Curves', fontsize=15, y=1.0)
        fig.tight_layout()

        for fmt in ('png', 'pdf'):
            path = os.path.join(save_dir, f'summary_combined.{fmt}')
            fig.savefig(path)
            saved_files.append(path)
        plt.close(fig)

    if saved_files:
        print(f"\n  Training curves saved to {save_dir}/")
        for f in saved_files:
            print(f"    → {os.path.basename(f)}")

    return saved_files
