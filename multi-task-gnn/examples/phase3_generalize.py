"""Phase 3: K-generalization — 测试策略在不同 K 值下的泛化能力。

============================================================================
研究问题：
  在 K=4 下训练的 GNN，能否在 K={2, 4, 8, 16} 上都稳定地超过基线？
  这回答了"super-DAG 方法是否具备 K 无关性"这个核心研究问题。

两种模式：
  (A) --mode zero_shot (默认)
      用同一个策略在不同 K 上直接 greedy 评估，看 zero-shot 泛化能力

  (B) --mode finetune
      对每个 K, 从 zero-shot 策略出发短期微调 (默认 50 ep)，看适应能力

产出：
  - k_generalization.png     makespan & speedup 曲线随 K 变化
  - results.csv              每个 (K, algorithm) 的原始 makespan
  - stats.json               汇总

用法：

  # zero-shot
  python multi-task-gnn/examples/phase3_generalize.py \\
      --model googlenet \\
      --policy artifacts/phase1/googlenet_K4.pt \\
      --ks 2,4,8,16

  # 每个 K 微调 50 episodes
  python multi-task-gnn/examples/phase3_generalize.py \\
      --model googlenet \\
      --policy artifacts/phase1/googlenet_K4.pt \\
      --ks 2,4,8,16 --mode finetune --ft-episodes 50
============================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import os

import torch

from _common import MODEL_FACTORIES, build_base_graph_state, project_path

from multi_task.super_dag import build_super_dag, super_graph_stats
from multi_task.baselines import evaluate_all_baselines
from multi_task.train_multi import (
    MultiTaskTrainConfig, train_super_dag, greedy_evaluate,
)
from multi_task.plot import plot_k_generalization


def load_policy_sd(ckpt_path: str, device: torch.device):
    """返回 (state_dict, hidden, emb, heads)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

    hidden, emb, heads = 128, 128, 4
    saved_cfg = ckpt.get('config') if isinstance(ckpt, dict) else None
    if saved_cfg is not None:
        hidden = getattr(saved_cfg, 'hidden_dim', hidden)
        emb = getattr(saved_cfg, 'emb_dim', emb)
        heads = getattr(saved_cfg, 'n_heads', heads)
    return sd, hidden, emb, heads


def build_policy_from_sd(sd, hidden, emb, heads, device):
    from gnn_strategy.graph_state import D_STATIC
    from gnn_strategy.policy import DynamicActorCritic
    policy = DynamicActorCritic(
        static_in_dim=D_STATIC, hidden_dim=hidden,
        emb_dim=emb, n_heads=heads, dropout=0.0,
    ).to(device)
    policy.load_state_dict(sd, strict=True)
    return policy


def main():
    p = argparse.ArgumentParser(description='Phase 3: K-generalization')
    p.add_argument('--model', type=str, default='googlenet',
                   choices=sorted(MODEL_FACTORIES.keys()))
    p.add_argument('--policy', type=str, required=True,
                   help='Phase 1 训练好的 policy 路径')
    p.add_argument('--ks', type=str, default='2,4,8,16',
                   help='要评估的 K 列表，逗号分隔')
    p.add_argument('--streams', type=int, default=8)
    p.add_argument('--mode', type=str, default='zero_shot',
                   choices=['zero_shot', 'finetune'])
    p.add_argument('--ft-episodes', type=int, default=50,
                   help='finetune 模式下每个 K 的微调轮数')
    p.add_argument('--ft-lr', type=float, default=1e-4)
    p.add_argument('--ft-batch-episodes', type=int, default=8)
    p.add_argument('--random-trials', type=int, default=5)
    p.add_argument('--out-dir', type=str, default=None)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    device = torch.device(
        args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu'
    )
    Ks = [int(s.strip()) for s in args.ks.split(',') if s.strip()]

    # ---- 1. 基础图 (只构建一次) ----
    print(f"\n{'='*60}")
    print(f"[Phase 3] model={args.model} Ks={Ks} mode={args.mode}")
    print(f"{'='*60}")
    base_gs, _, _, _ = build_base_graph_state(
        args.model, 'cuda' if torch.cuda.is_available() else 'cpu',
    )

    # ---- 2. 加载策略权重 ----
    policy_path = project_path(args.policy)
    sd, hidden, emb, heads = load_policy_sd(policy_path, device)
    print(f"  policy: {policy_path}  hidden={hidden} emb={emb} heads={heads}")

    k_results = {}

    for K in Ks:
        print(f"\n[K = {K}] building super-DAG ...")
        super_gs, info = build_super_dag(base_gs, K=K)
        stats = super_graph_stats(super_gs, info)
        print(f"  {stats}")

        # 基线
        baseline_res = evaluate_all_baselines(
            super_gs, num_tasks=K, n_streams=args.streams,
            random_seed=args.seed, random_trials=args.random_trials,
        )

        # GNN
        if args.mode == 'zero_shot':
            policy = build_policy_from_sd(sd, hidden, emb, heads, device)
            policy.eval()
            L_gnn, _ = greedy_evaluate(policy, super_gs,
                                       n_streams=args.streams, device=device)
            baseline_res['GNN (zero-shot)'] = float(L_gnn)
        else:
            print(f"  fine-tuning on K={K} for {args.ft_episodes} episodes ...")
            cfg = MultiTaskTrainConfig(
                episodes=args.ft_episodes,
                batch_episodes=args.ft_batch_episodes,
                mini_batch_size=512,
                ppo_epochs=4,
                lr=args.ft_lr,
                gae_lambda=1.0,
                entropy_coef=0.01,
                hidden_dim=hidden, emb_dim=emb, n_heads=heads,
                n_streams=args.streams,
                num_tasks=K,
            )
            policy, hist = train_super_dag(
                super_gs=super_gs, cfg=cfg, device=device, seed=args.seed,
                save_path=None, init_state_dict=sd,
                log_prefix=f'K{K}-FT',
            )
            L_gnn, _ = greedy_evaluate(policy, super_gs,
                                       n_streams=args.streams, device=device)
            baseline_res['GNN (zero-shot)'] = float(hist['episodes'][0]['L_gnn']) \
                if hist.get('episodes') else float('nan')
            baseline_res['GNN (after FT)'] = float(L_gnn)

        k_results[K] = baseline_res
        for name, v in sorted(baseline_res.items(), key=lambda kv: kv[1]):
            print(f"    {name:>24s}: {v:.2f}")

    # ---- 3. 输出 ----
    if args.out_dir is None:
        args.out_dir = os.path.join(
            os.path.dirname(policy_path), 'phase3',
            f'{args.model}_{args.mode}',
        )
    out_dir = project_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    plot_k_generalization(
        k_results, os.path.join(out_dir, 'k_generalization.png'),
        title=f'{args.model} ({args.mode})',
        baseline_key='Opara-like',
    )

    csv_path = os.path.join(out_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        # 收集所有算法名（union 各 K）
        all_algos = []
        for K in Ks:
            for a in k_results[K].keys():
                if a not in all_algos:
                    all_algos.append(a)
        w.writerow(['K'] + all_algos)
        for K in Ks:
            row = [K]
            for a in all_algos:
                v = k_results[K].get(a, '')
                row.append(f'{v:.4f}' if v != '' else '')
            w.writerow(row)

    with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
        json.dump({
            'model': args.model,
            'mode': args.mode,
            'Ks': Ks,
            'results': {int(K): v for K, v in k_results.items()},
            'policy_path': policy_path,
        }, f, indent=2)

    print(f"\n  [save] {csv_path}")
    print(f"  [save] {out_dir}/k_generalization.png")


if __name__ == '__main__':
    main()
