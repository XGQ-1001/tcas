"""PPO 训练循环 — 在 super-DAG 上训练 GNN 调度策略。

============================================================================
与单任务训练 (gnn_strategy.train.train_policy_real) 的关键区别：

  1. 状态空间变大 K 倍：super-DAG 有 K*N 节点
  2. 奖励信号：我们用「模拟 makespan」而非真实 GPU benchmark
     - 原因：真实 super-DAG 的 CUDA Graph 捕获需要拷贝 K 份模型副本，
       工程复杂度太高；为了快速前期验证，用模拟 makespan 足够证明
       "GNN 能学到比基线更好的调度"
     - 模拟 makespan 的准确度：在单任务上与真实延迟相关系数 > 0.85，
       足够作为 reward signal
  3. 基线：使用 opara_like_order 作为基线（对应"Opara 应用到 super-DAG"）
     - reward = (L_opara_like - L_gnn) / L_opara_like * 100  → 百分比改善
  4. 保持其他所有超参数和网络结构完全不变，复用现有代码

训练完全复用：
  - DynamicActorCritic           (gnn_strategy.policy)
  - SchedulingEnv                (gnn_strategy.env)
  - collect_rollout / ppo_update (gnn_strategy.train)
  - compute_gae / Transition     (gnn_strategy.train)
============================================================================
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_GNN_STRATEGY_DIR = os.path.normpath(
    os.path.join(_THIS_DIR, '..', '..', 'gnn-strategy')
)
_REPO_ROOT = os.path.dirname(_GNN_STRATEGY_DIR)
for _p in (_GNN_STRATEGY_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gnn_strategy.env import SchedulingEnv
from gnn_strategy.graph_state import GraphState, D_STATIC
from gnn_strategy.policy import DynamicActorCritic
from gnn_strategy.train import (
    TrainConfig,
    Transition,
    collect_rollout,
    compute_gae,
    ppo_update,
    scheduled_entropy_coef,
)

from .baselines import opara_like_order, simulate_makespan


# --------------------------------------------------------------------------
# 多任务训练配置（继承自 TrainConfig，额外添加 super-DAG 相关参数）
# --------------------------------------------------------------------------

@dataclass
class MultiTaskTrainConfig(TrainConfig):
    """Multi-task 训练配置，额外字段：

    num_tasks:    K，并发推理任务数（super-DAG 的副本数）
    reward_scale: 奖励缩放因子（默认以百分比形式）
    """
    num_tasks: int = 4
    reward_scale: float = 100.0


def _heuristic_baseline_makespan(
    super_gs: GraphState,
    n_streams: int,
) -> float:
    """用 opara_like_order 作为基线调度，返回其模拟 makespan。"""
    order = opara_like_order(super_gs)
    return simulate_makespan(super_gs, order, n_streams=n_streams)


def train_super_dag(
    super_gs: GraphState,
    cfg: MultiTaskTrainConfig,
    device: Optional[torch.device] = None,
    seed: int = 0,
    save_path: Optional[str] = None,
    init_state_dict: Optional[dict] = None,
    log_prefix: str = 'PPO',
) -> Tuple[DynamicActorCritic, Dict]:
    """在给定的 super-DAG 上训练 GNN 调度策略。

    参数:
        super_gs:        super-DAG GraphState (由 build_super_dag 得到)
        cfg:             MultiTaskTrainConfig
        device:          torch device (默认自动选择 GPU)
        seed:            随机种子
        save_path:       策略保存路径 (可选)
        init_state_dict: 预训练策略权重 (用于 warm start / transfer)
        log_prefix:      日志前缀

    返回:
        (policy, history) 训练完成的策略 + 训练历史
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- 基线 makespan (Opara-like on super-DAG) ----
    L_base = _heuristic_baseline_makespan(super_gs, cfg.n_streams)
    print(f"  Super-DAG baseline (Opara-like): {L_base:.2f}")

    # ---- 策略网络 ----
    policy = DynamicActorCritic(
        static_in_dim=D_STATIC,
        hidden_dim=cfg.hidden_dim,
        emb_dim=cfg.emb_dim,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
    ).to(device)

    if init_state_dict is not None:
        try:
            policy.load_state_dict(init_state_dict, strict=True)
            print(f"  Loaded init weights (strict=True)")
        except Exception as e:
            print(f"  Warning: init weights load failed: {e}")

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy params: {n_params:,}")

    # ---- Optimizer + LR 调度 ----
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    n_updates = max(cfg.episodes // max(cfg.batch_episodes, 1), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_updates, eta_min=cfg.lr * 0.1,
    )

    history: Dict = {
        'episodes': [],
        'grad_steps': [],
        'L_baseline': float(L_base),
        'num_tasks': cfg.num_tasks,
        'num_nodes': len(super_gs.node_names),
    }

    best_L = L_base
    best_speedup = 0.0

    # ---- 批量缓冲 ----
    batch_transitions: List[Transition] = []
    batch_advantages: List[float] = []
    batch_returns: List[float] = []
    pending_recs: List[Dict] = []
    update_count = 0

    for ep in range(cfg.episodes):
        # ---- Rollout ----
        env = SchedulingEnv(
            super_gs, n_streams=cfg.n_streams, device=device,
            reward_weights=cfg.reward_weights,
        )
        transitions, _ = collect_rollout(policy, super_gs, env, device)
        L_gnn = env.current_makespan()

        # ---- Terminal-only reward ----
        real_reward = (L_base - L_gnn) / max(L_base, 1e-9) * cfg.reward_scale
        T_ep = len(transitions)
        for i in range(T_ep):
            transitions[i].reward = real_reward if i == T_ep - 1 else 0.0

        values = [tr.value for tr in transitions]
        rewards = [tr.reward for tr in transitions]
        advantages, returns = compute_gae(
            rewards, values, last_value=0.0,
            gamma=1.0, lam=cfg.gae_lambda,
        )

        batch_transitions.extend(transitions)
        batch_advantages.extend(advantages)
        batch_returns.extend(returns)

        # ---- 追踪最优 ----
        if L_gnn < best_L:
            best_L = L_gnn
            best_speedup = real_reward
            if save_path:
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                torch.save({
                    'state_dict': policy.state_dict(),
                    'config': cfg,
                    'history': history,
                    'best_makespan': float(best_L),
                    'baseline_makespan': float(L_base),
                    'num_tasks': cfg.num_tasks,
                    'hidden_dim': cfg.hidden_dim,
                    'emb_dim': cfg.emb_dim,
                    'n_heads': cfg.n_heads,
                }, save_path)

        pending_recs.append({
            'episode': int(ep),
            'L_gnn': float(L_gnn),
            'L_base': float(L_base),
            'speedup_pct': float(real_reward),
            'best_L': float(best_L),
            'best_speedup_pct': float(best_speedup),
        })

        is_last = (ep == cfg.episodes - 1)
        batch_full = (len(pending_recs) >= cfg.batch_episodes)

        # ---- PPO 更新 ----
        if batch_transitions and (batch_full or is_last):
            ent_sched = scheduled_entropy_coef(cfg, ep, cfg.episodes)
            stats = ppo_update(
                policy, optimizer, batch_transitions, batch_advantages,
                batch_returns, cfg, device, entropy_coef=ent_sched,
            )
            scheduler.step()
            update_count += 1
            cur_lr = optimizer.param_groups[0]['lr']

            per_step = stats.pop('step_stats', [])
            for rec in pending_recs:
                rec.update(stats)
                rec['entropy_coef_used'] = float(ent_sched)
                rec['lr'] = float(cur_lr)
                rec['ppo_update'] = int(update_count)
            history['episodes'].extend(pending_recs)

            base_step = len(history['grad_steps'])
            for i, ss in enumerate(per_step):
                ss['global_step'] = base_step + i
                ss['ppo_update'] = update_count
                ss['episode'] = int(ep)
            history['grad_steps'].extend(per_step)

            last = pending_recs[-1]
            print(
                f"[{log_prefix} ep {ep:04d}] L_gnn={L_gnn:.1f} "
                f"speedup={real_reward:+.2f}% "
                f"best_speedup={best_speedup:+.2f}% "
                f"pg={stats['pg_loss']:+.4f} v={stats['v_loss']:.4f} "
                f"ent={stats['entropy']:.3f} lr={cur_lr:.2e}"
            )

            batch_transitions.clear()
            batch_advantages.clear()
            batch_returns.clear()
            pending_recs.clear()
        else:
            # 未触发 PPO 更新，仅打印 rollout 信息
            if ep % max(cfg.batch_episodes // 2, 1) == 0:
                print(
                    f"[{log_prefix} ep {ep:04d}] L_gnn={L_gnn:.1f} "
                    f"speedup={real_reward:+.2f}% (buffering...)"
                )

    history['best_L'] = float(best_L)
    history['best_speedup_pct'] = float(best_speedup)

    return policy, history


def greedy_evaluate(
    policy: DynamicActorCritic,
    super_gs: GraphState,
    n_streams: int = 8,
    device: Optional[torch.device] = None,
) -> Tuple[float, List[int]]:
    """贪心推理：每步 argmax，返回最终 makespan + 调度顺序。"""
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    env = SchedulingEnv(super_gs, n_streams=n_streams, device=device)
    env.reset()

    x = super_gs.x.to(device)
    with torch.no_grad():
        h_static = policy.encode_static(
            x, parents=super_gs.parents, children=super_gs.children,
        )

    while not env.is_done():
        dyn_node = env.dynamic_node_features().to(device)
        glob = env.global_features().to(device)
        ready_mask = env.ready_mask().to(device)

        with torch.no_grad():
            dist, _ = policy.act(h_static, dyn_node, glob, ready_mask)

        action = int(dist.probs.argmax().item())
        env.step(action)

    return float(env.current_makespan()), list(env.scheduled_order())
