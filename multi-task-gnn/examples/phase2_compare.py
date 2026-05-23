"""Phase 2: 把训练好的 GNN 策略和所有非学习基线做深度对比。

============================================================================
目标:
  对于固定的 K (默认取 Phase 1 训练时的 K)，在 super-DAG 上做严格对比：

    - PyTorch 默认 (Topological, task-major)
    - Round-Robin (手工任务交织)
    - Opara-like 贪心 (GNN 的训练基线)
    - Random (下限对照, 取多次平均)
    - GNN (greedy argmax)
    - GNN (best during training)   # 训练中找到的最优调度

产出:
  - baseline_compare.png            makespan + speedup 柱状图
  - results.csv                      所有算法的原始数据
  - stats.json                       相对 Opara-like 的 speedup、相对 Topological 的 speedup

用法：

  python multi-task-gnn/examples/phase2_compare.py \\
      --model googlenet --num-tasks 4 \\
      --policy artifacts/phase1/googlenet_K4.pt
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
from multi_task.baselines import (
    evaluate_all_baselines,
    simulate_makespan,
)
from multi_task.train_multi import greedy_evaluate
from multi_task.plot import plot_baseline_compare


def load_policy(ckpt_path: str, device: torch.device):
    from gnn_strategy.graph_state import D_STATIC
    from gnn_strategy.policy import DynamicActorCritic

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt

    hidden = ckpt.get('hidden_dim', 128) if isinstance(ckpt, dict) else 128
    emb = ckpt.get('emb_dim', 128) if isinstance(ckpt, dict) else 128
    heads = ckpt.get('n_heads', 4) if isinstance(ckpt, dict) else 4

    saved_cfg = ckpt.get('config') if isinstance(ckpt, dict) else None
    if saved_cfg is not None:
        hidden = getattr(saved_cfg, 'hidden_dim', hidden)
        emb = getattr(saved_cfg, 'emb_dim', emb)
        heads = getattr(saved_cfg, 'n_heads', heads)

    policy = DynamicActorCritic(
        static_in_dim=D_STATIC, hidden_dim=hidden,
        emb_dim=emb, n_heads=heads, dropout=0.0,
    ).to(device)
    policy.load_state_dict(sd, strict=True)
    policy.eval()

    best_L = ckpt.get('best_makespan') if isinstance(ckpt, dict) else None
    baseline = ckpt.get('baseline_makespan') if isinstance(ckpt, dict) else None
    return policy, best_L, baseline


def main():
    p = argparse.ArgumentParser(description='Phase 2: Compare GNN vs baselines')
    p.add_argument('--model', type=str, default='googlenet',
                   choices=sorted(MODEL_FACTORIES.keys()))
    p.add_argument('--num-tasks', type=int, default=4)
    p.add_argument('--streams', type=int, default=8)
    p.add_argument('--policy', type=str, required=True,
                   help='Phase 1 训练好的 policy 路径')
    p.add_argument('--random-trials', type=int, default=10,
                   help='随机基线的重复次数')
    p.add_argument('--out-dir', type=str, default=None,
                   help='输出目录 (默认: 策略同目录的 phase2/)')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    device = torch.device(
        args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu'
    )

    # ---- 1. 构建 super-DAG ----
    print(f"\n{'='*60}")
    print(f"[Phase 2] model={args.model} K={args.num_tasks}")
    print(f"{'='*60}")
    base_gs, _, _, _ = build_base_graph_state(
        args.model, 'cuda' if torch.cuda.is_available() else 'cpu',
    )
    super_gs, info = build_super_dag(base_gs, K=args.num_tasks)
    print(f"  super-DAG: {super_graph_stats(super_gs, info)}")

    # ---- 2. 非学习基线 ----
    baseline_results = evaluate_all_baselines(
        super_gs, num_tasks=args.num_tasks, n_streams=args.streams,
        random_seed=args.seed, random_trials=args.random_trials,
    )

    # ---- 3. GNN policy greedy ----
    policy_path = project_path(args.policy)
    policy, best_train_L, train_baseline_L = load_policy(policy_path, device)
    L_gnn_greedy, order_ids = greedy_evaluate(
        policy, super_gs, n_streams=args.streams, device=device,
    )
    baseline_results['GNN (greedy)'] = float(L_gnn_greedy)
    if best_train_L is not None:
        baseline_results['GNN (best ckpt)'] = float(best_train_L)

    # ---- 4. 打印结果 ----
    print("\n[Comparison — simulated makespan]")
    for k, v in sorted(baseline_results.items(), key=lambda kv: kv[1]):
        print(f"  {k:>24s}: {v:.2f}")

    base_v = baseline_results.get('Opara-like', None)
    topo_v = baseline_results.get('Topological', None)
    gnn_v = baseline_results.get('GNN (greedy)', None)

    print("\n[Speedups]")
    if base_v:
        print(f"  GNN vs Opara-like  : {(base_v - gnn_v)/base_v*100:+.2f}%")
    if topo_v:
        print(f"  GNN vs Topological : {(topo_v - gnn_v)/topo_v*100:+.2f}%")
    if topo_v and base_v:
        print(f"  Opara-like vs Topological : {(topo_v - base_v)/topo_v*100:+.2f}%")

    # ---- 5. 输出 ----
    if args.out_dir is None:
        args.out_dir = os.path.join(
            os.path.dirname(policy_path), 'phase2',
            f'{args.model}_K{args.num_tasks}',
        )
    out_dir = project_path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    plot_baseline_compare(
        baseline_results,
        os.path.join(out_dir, 'baseline_compare.png'),
        title=f'{args.model} / K={args.num_tasks}',
        baseline_key='Opara-like',
    )

    csv_path = os.path.join(out_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['algorithm', 'makespan', 'speedup_vs_opara_pct',
                    'speedup_vs_topo_pct'])
        for k, v in baseline_results.items():
            s_op = (base_v - v) / base_v * 100 if base_v else None
            s_tp = (topo_v - v) / topo_v * 100 if topo_v else None
            w.writerow([k, f'{v:.4f}',
                        f'{s_op:.4f}' if s_op is not None else '',
                        f'{s_tp:.4f}' if s_tp is not None else ''])

    stats_json = os.path.join(out_dir, 'stats.json')
    with open(stats_json, 'w') as f:
        json.dump({
            'model': args.model,
            'num_tasks': args.num_tasks,
            'results': baseline_results,
            'speedup_vs_opara_pct': (
                (base_v - gnn_v) / base_v * 100.0 if base_v else None
            ),
            'speedup_vs_topological_pct': (
                (topo_v - gnn_v) / topo_v * 100.0 if topo_v else None
            ),
            'policy_path': policy_path,
        }, f, indent=2)

    print(f"\n  [save] {csv_path}")
    print(f"  [save] {stats_json}")


if __name__ == '__main__':
    main()
