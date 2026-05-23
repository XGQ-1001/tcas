"""Phase 1: 在 K 任务 super-DAG 上训练 GNN 调度策略。

============================================================================
目标:
  验证「把 K 个 batch=1 DAG 拼成一个大 super-DAG 丢给 GNN 调度」是可行的：
  GNN 能学到一个优于 Opara-like 贪心基线的调度策略。

流程：
  1. 加载基础模型 (默认 GoogLeNet)，用 torch._dynamo.explain 得 FX 图
  2. 构建基础 GraphState (单任务 DAG)
  3. build_super_dag(base_gs, K) → K 份拼接的 super-GraphState
  4. 在 super-GraphState 上跑 PPO，reward = (L_baseline - L_gnn) / L_baseline
  5. 保存策略 + 训练曲线

命令行示例：

  # 默认 K=4, 在 GoogLeNet 上训练 300 episodes
  python multi-task-gnn/examples/phase1_train.py \\
      --model googlenet --num-tasks 4 --episodes 300

  # 更多并发任务
  python multi-task-gnn/examples/phase1_train.py \\
      --model googlenet --num-tasks 8 --episodes 500

  # 从单任务预训练权重 warm-start
  python multi-task-gnn/examples/phase1_train.py \\
      --model googlenet --num-tasks 4 --episodes 300 \\
      --init-from /path/to/googlenet_real_nobc_final.pt
============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import torch

from _common import (
    MODEL_FACTORIES,
    build_base_graph_state,
    project_path,
)

from multi_task.super_dag import build_super_dag, super_graph_stats
from multi_task.train_multi import MultiTaskTrainConfig, train_super_dag
from multi_task.baselines import evaluate_all_baselines
from multi_task.plot import plot_training_curves, plot_baseline_compare


def main():
    p = argparse.ArgumentParser(description='Phase 1: Train GNN on super-DAG')

    p.add_argument('--model', type=str, default='googlenet',
                   choices=sorted(MODEL_FACTORIES.keys()))
    p.add_argument('--num-tasks', type=int, default=4,
                   help='K: 并发推理任务数 (super-DAG 副本数)')
    p.add_argument('--episodes', type=int, default=300)
    p.add_argument('--batch-episodes', type=int, default=8)
    p.add_argument('--mini-batch-size', type=int, default=512)
    p.add_argument('--ppo-epochs', type=int, default=4)
    p.add_argument('--clip-eps', type=float, default=0.2)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--gae-lambda', type=float, default=1.0)
    p.add_argument('--entropy-coef', type=float, default=0.02)
    p.add_argument('--entropy-coef-end', type=float, default=None)

    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--emb', type=int, default=128)
    p.add_argument('--heads', type=int, default=4)
    p.add_argument('--streams', type=int, default=8,
                   help='模拟的 CUDA 流数量')

    p.add_argument('--init-from', type=str, default=None,
                   help='预训练单任务权重 (用于 warm start)')
    p.add_argument('--save', type=str, default=None,
                   help='策略保存路径 (默认 artifacts/phase1/...)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', type=str, default='cuda')

    args = p.parse_args()
    device = torch.device(
        args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu'
    )

    # ---- 1. 基础 GraphState ----
    print(f"\n{'='*60}")
    print(f"[Phase 1] model={args.model} K={args.num_tasks} eps={args.episodes}")
    print(f"{'='*60}")
    base_gs, fx_module, inputs, model_class_name = build_base_graph_state(
        args.model, 'cuda' if torch.cuda.is_available() else 'cpu',
    )
    print(f"  base DAG: {len(base_gs.node_names)} nodes, "
          f"{int(base_gs.movable_mask.sum().item())} movable")

    # ---- 2. Super-DAG ----
    super_gs, info = build_super_dag(base_gs, K=args.num_tasks)
    stats = super_graph_stats(super_gs, info)
    print(f"  super-DAG: K={stats['num_tasks']}  "
          f"total_nodes={stats['total_nodes']}  "
          f"total_movable={stats['total_movable']}")

    # ---- 3. 先跑一次基线评估 (warmup) ----
    baseline_results = evaluate_all_baselines(
        super_gs, num_tasks=args.num_tasks, n_streams=args.streams,
        random_seed=args.seed, random_trials=5,
    )
    print("\n[Baseline makespan before training]")
    for k, v in baseline_results.items():
        print(f"  {k:>16s}: {v:.2f}")

    # ---- 4. 准备保存路径 ----
    if args.save is None:
        args.save = os.path.join(
            'artifacts', 'phase1',
            f'{args.model}_K{args.num_tasks}.pt',
        )
    save_path = project_path(args.save)
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    # ---- 5. 读取 init 权重 ----
    init_sd = None
    if args.init_from:
        init_path = project_path(args.init_from)
        if os.path.exists(init_path):
            ckpt = torch.load(init_path, map_location=device, weights_only=False)
            init_sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
            print(f"  init-from: {init_path}")

    # ---- 6. 训练 ----
    cfg = MultiTaskTrainConfig(
        episodes=args.episodes,
        batch_episodes=args.batch_episodes,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps,
        lr=args.lr,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        entropy_coef_end=args.entropy_coef_end,
        hidden_dim=args.hidden,
        emb_dim=args.emb,
        n_heads=args.heads,
        n_streams=args.streams,
        num_tasks=args.num_tasks,
    )

    policy, history = train_super_dag(
        super_gs=super_gs, cfg=cfg, device=device, seed=args.seed,
        save_path=save_path, init_state_dict=init_sd,
        log_prefix=f'K{args.num_tasks}',
    )

    # ---- 7. 用最终策略贪心评估 + 合入 baseline_results ----
    from multi_task.train_multi import greedy_evaluate
    L_gnn_greedy, _ = greedy_evaluate(policy, super_gs,
                                      n_streams=args.streams, device=device)
    baseline_results['GNN (greedy)'] = float(L_gnn_greedy)
    baseline_results['GNN (best during train)'] = float(history['best_L'])

    print("\n[Final results]")
    for k, v in baseline_results.items():
        print(f"  {k:>24s}: {v:.2f}")

    # ---- 8. 保存结果 + 绘图 ----
    plots_dir = os.path.join(os.path.dirname(save_path), 'plots',
                              f'{args.model}_K{args.num_tasks}')
    plot_training_curves(
        history, os.path.join(plots_dir, 'training_curves.png'),
        title_suffix=f'[{args.model}, K={args.num_tasks}]',
    )
    plot_baseline_compare(
        baseline_results, os.path.join(plots_dir, 'baseline_compare.png'),
        title=f'{args.model} / K={args.num_tasks}',
        baseline_key='Opara-like',
    )

    results_json = os.path.join(plots_dir, 'results.json')
    with open(results_json, 'w') as f:
        json.dump({
            'model': args.model,
            'num_tasks': args.num_tasks,
            'config': vars(args),
            'baseline_results': baseline_results,
            'best_L': history.get('best_L'),
            'best_speedup_pct': history.get('best_speedup_pct'),
            'L_baseline': history.get('L_baseline'),
        }, f, indent=2)
    print(f"\n  [save] {results_json}")
    print(f"  [save] policy: {save_path}")


if __name__ == '__main__':
    main()
