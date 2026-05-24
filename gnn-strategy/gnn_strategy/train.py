"""PPO 训练器 — 整个 GNN 调度策略的训练核心。

============================================================================
整体训练流程（real-latency 模式，推荐）：

  ┌─────────────────────────────────────────────────────────────────┐
  │  一次性准备（只做一次，约6秒）：                                   │
  │    1. torch._dynamo.explain(model) → FX 计算图                   │
  │    2. OperatorLauncher.recompile   → 算子 profiling 数据          │
  │    3. build_graph_state            → GraphState（静态特征+拓扑）  │
  │    4. GraphCapturer.capturer       → Opara 基线延迟              │
  └─────────────────────────────────────────────────────────────────┘
                              ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │  Stage 1: 行为克隆预训练 BC（可选，默认30轮）                     │
  │    - 用 TCAS 启发式算法的调度顺序作为"老师"                       │
  │    - 策略网络模仿老师的选择，快速学会合理的调度策略                  │
  │    - 好处：避免 PPO 从随机策略开始（太慢）                        │
  └─────────────────────────────────────────────────────────────────┘
                              ↓
  ┌─────────────────────────────────────────────────────────────────┐
  │  Stage 2: PPO + 真实 GPU 延迟（核心阶段）                        │
  │    每轮 episode：                                                │
  │    ┌────────────────────────────────────────────────────────┐    │
  │    │ 2a. Rollout：策略网络在模拟环境中逐步选择节点           │    │
  │    │     → 得到完整的调度顺序 + 每步的动态特征              │    │
  │    │                                                       │    │
  │    │ 2b. 真实评估：用调度顺序 capture CUDA Graph            │    │
  │    │     → 在 H20 上 benchmark → 得到真实延迟 L_gnn         │    │
  │    │                                                       │    │
  │    │ 2c. 计算奖励：R = (L_opara - L_gnn) / L_opara × 100  │    │
  │    │     → 正值 = 比 Opara 快，负值 = 比 Opara 慢           │    │
  │    │                                                       │    │
  │    │ 2d. GAE 计算优势函数 + PPO Clipped 更新策略            │    │
  │    └────────────────────────────────────────────────────────┘    │
  │    重复 500 轮，保存最优策略                                      │
  └─────────────────────────────────────────────────────────────────┘

PPO (Proximal Policy Optimization) 核心原理：
  - 目标：最大化 E[R]，即找到让延迟最低的调度策略
  - ★ 关键：每 batch_episodes(默认8) 个 episode 做一次 PPO 更新
    而非每 episode 更新一次，跨 episode 归一化优势才有足够信号
  - 使用向量化 mini-batch SGD，每梯度步处理 mini_batch_size 个 transition
  - Clipped Surrogate Objective:
      L = min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    其中 ratio = π_new(a|s) / π_old(a|s)，A = 优势函数
    clip 防止策略更新过大导致崩溃
  - 学习率余弦退火：从 lr 平滑衰减到 lr*0.1

GAE (Generalized Advantage Estimation)：
  - 估计每一步的"优势"：选择这个动作比平均好多少
  - A_t = Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}
  - δ_t = r_t + γ·V(s_{t+1}) - V(s_t)  (TD 误差)
  - λ 控制偏差-方差权衡：λ→0 低方差高偏差, λ→1 高方差低偏差

============================================================================
可调超参数一览：

  超参数            默认值   说明                          调节建议
  ────────────────────────────────────────────────────────────────────
  episodes         500     PPO 训练总轮数                 500-2000, 越多越充分
  batch_episodes     8     ★ 每次 PPO 更新收集的 ep 数    4-16, 越大梯度越稳
  mini_batch_size  256     ★ mini-batch 大小             128-512
  bc_episodes       30     BC 预训练轮数                  20-50, 太多会过拟合教师
  lr              1e-4     学习率                        1e-4 ~ 1e-3
  ppo_epochs         4     每批数据重复训练次数            2-8
  clip_eps         0.2     PPO 裁剪范围                  0.1-0.3
  gamma            1.0     折扣因子（real模式固定1.0）     不需调
  gae_lambda       0.95    GAE 的 λ 参数                 0.9-0.99
  entropy_coef     0.02    熵正则化系数                   0.01-0.05, 越大探索越多
  value_coef       0.5     价值损失权重                   0.25-1.0
  max_grad_norm    0.5     梯度裁剪                      0.5-1.0
  hidden_dim       128     网络隐藏层维度                  64/128/256
  emb_dim          128     节点嵌入维度                   64/128/256
  n_heads            4     注意力头数                    2/4/8
  n_streams          8     CUDA 流数量                   4/8/16（与目标部署一致）
  bench_iters       20     每次评估的 benchmark 次数       10-50, 越多越精确但更慢
  bench_warmups      5     benchmark 预热次数             3-10
============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

_GNN_STRATEGY_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from Opara import GraphCapturer
from Opara import OperatorLauncher

from .capturer import capturer_gnn, fast_eval_latency, measure_latency_ms
from .env import SchedulingEnv
from .graph_state import build_graph_state, D_STATIC
from .plot_training import plot_training_curves
from .policy import DynamicActorCritic
from .utils import extract_first_fx_graph


# ======================================================================
# 训练配置（所有可调超参数都在这里）
# ======================================================================

@dataclass
class TrainConfig:
    # ---- PPO 超参数 ----
    episodes: int = 200            # PPO 训练总轮数 [可调]
    batch_episodes: int = 8        # ★ 每次 PPO 更新前收集的 episode 数 [可调] 越大梯度越稳
    mini_batch_size: int = 256     # ★ PPO mini-batch 大小 [可调] 每次梯度步用的 transition 数
    ppo_epochs: int = 4            # 每批数据重复训练次数 [可调] 增大→更充分利用数据
    clip_eps: float = 0.2          # PPO 裁剪范围 ε [可调] 控制策略更新幅度
    lr: float = 1e-4               # Adam 学习率 [可调]
    gamma: float = 0.99            # 折扣因子 [可调] real模式下会覆盖为1.0
    gae_lambda: float = 1.0        # GAE 的 λ [可调] terminal-only reward 必须=1.0，否则前面的步骤信号衰减到 0 
    #gae_lambda原先是0.95
    entropy_coef: float = 0.01     # 熵奖励系数 [可调] 越大→探索越多，避免过早收敛
    entropy_coef_end: Optional[float] = None  # 若设置，训练过程中熵系数线性衰减到此值
    value_coef: float = 0.5        # 价值损失权重 [可调]
    max_grad_norm: float = 0.5     # 梯度裁剪上限 [可调] 防止梯度爆炸

    # ---- 网络结构超参数 ----
    hidden_dim: int = 64           # GAT 和 MLP 的隐藏维度 [可调]
    emb_dim: int = 64              # 节点嵌入维度 [可调]
    n_heads: int = 4               # 注意力头数 [可调]
    dropout: float = 0.1           # Dropout 比例 [可调]

    # ---- 环境配置 ----
    n_streams: int = 8             # 模拟的 CUDA 流数量 [可调] 建议与实际一致

    # ---- BC 预训练 ----
    bc_episodes: int = 0           # BC 预训练轮数 [可调] 0=跳过
    bc_lr: float = 1e-3            # BC 学习率 [可调]

    # ---- 真实延迟评估 ----
    real_finetune_episodes: int = 0  # surrogate 模式下的微调轮数
    bench_iters: int = 30          # benchmark 重复次数 [可调] 越多越精确
    bench_warmups: int = 10        # benchmark 预热次数 [可调]
    autosave_interval: int = 1     # real训练每多少次PPO update保存一次_latest.pt，1=每次都保存

    # ---- surrogate 模式的奖励权重 ----
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'makespan': 1.0,           # makespan 改善权重
        'contention': 0.1,         # 资源竞争惩罚
        'overlap': 0.05,           # 并行重叠奖励
        'idle': 0.05,              # 流空闲惩罚
    })


def scheduled_entropy_coef(cfg: TrainConfig, ep: int, total_episodes: int) -> float:
    """PPO 熵系数调度：从 entropy_coef 线性插值到 entropy_coef_end（若未设置则恒定）。"""
    end = cfg.entropy_coef_end
    if end is None:
        return float(cfg.entropy_coef)
    if total_episodes <= 1:
        return float(cfg.entropy_coef)
    t = ep / float(total_episodes - 1)
    return float(cfg.entropy_coef + (end - cfg.entropy_coef) * t)


def _checkpoint_path(save_path: str, suffix: str) -> str:
    """Return a checkpoint path with suffix inserted before .pt."""
    root, ext = os.path.splitext(save_path)
    if ext:
        return f"{root}_{suffix}{ext}"
    return f"{save_path}_{suffix}.pt"


def _save_training_checkpoint(
    path: str,
    policy: DynamicActorCritic,
    cfg: TrainConfig,
    history: Dict,
    best_L: float,
    L_opara: float,
    model_class_name: str,
    update_count: int,
    status: str,
) -> None:
    """Atomically save a training checkpoint that can be replotted later."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save({
        'state_dict': policy.state_dict(),
        'config': cfg,
        'history': history,
        'best_latency_ms': float(best_L),
        'opara_latency_ms': float(L_opara),
        'hidden_dim': cfg.hidden_dim,
        'emb_dim': cfg.emb_dim,
        'n_heads': cfg.n_heads,
        'update_count': int(update_count),
        'status': status,
        'model': model_class_name,
    }, tmp_path)
    os.replace(tmp_path, path)


# ======================================================================
# Rollout 存储 — 记录每步的观测、动作、奖励
# ======================================================================

@dataclass
class Transition:
    """一步交互的记录。rollout 过程中收集，PPO 更新时使用。"""
    h_static: torch.Tensor     # [N, emb]   静态嵌入（detached）
    dyn_node: torch.Tensor     # [N, D_DYN] 动态节点特征
    glob: torch.Tensor         # [D_GLOBAL] 全局特征
    ready_mask: torch.Tensor   # [N]        ready 掩码
    action: int                # 选择的节点 id
    log_prob: float            # 该动作的对数概率 log π(a|s)
    value: float               # Critic 输出的 V(s)
    reward: float              # 该步的奖励（real模式下会被覆盖）


def compute_gae(
    rewards: List[float],
    values: List[float],
    last_value: float,
    gamma: float,
    lam: float,
) -> Tuple[List[float], List[float]]:
    """广义优势估计 (GAE)。

    从后往前递推计算每一步的优势 A_t 和回报 G_t：
      δ_t = r_t + γ·V(s_{t+1}) - V(s_t)    (TD 误差)
      A_t = δ_t + γ·λ·A_{t+1}               (GAE 递推)
      G_t = A_t + V(s_t)                     (回报 = 优势 + 基线)

    参数:
        rewards:    每步奖励列表 [r_0, r_1, ..., r_{T-1}]
        values:     每步价值估计 [V(s_0), V(s_1), ..., V(s_{T-1})]
        last_value: 最后一步的 V(s_T)（episode 结束时为 0）
        gamma:      折扣因子（real 模式下 = 1.0）
        lam:        GAE 的 λ 参数

    返回:
        (advantages, returns) 每步的优势和回报
    """
    T = len(rewards)
    advantages = [0.0] * T
    returns = [0.0] * T
    gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1] if t + 1 < T else last_value
        delta = rewards[t] + gamma * next_val - values[t]  # TD 误差
        gae = delta + gamma * lam * gae                     # GAE 递推
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    return advantages, returns


# ======================================================================
# Rollout 采集 — 策略网络与环境交互，收集一轮数据
# ======================================================================

def collect_rollout(
    policy: DynamicActorCritic,
    gs,
    env: SchedulingEnv,
    device: torch.device,
) -> Tuple[List[Transition], float]:
    """执行一个完整 episode，收集所有步的 transition。

    流程：
      1. 用 StaticEncoder 编码图的静态特征（只算一次）
      2. 循环直到所有节点调度完：
         - 获取动态特征 + 全局特征 + ready 掩码
         - 策略网络输出 (动作分布, 价值估计)
         - 从分布中采样动作 → 环境执行 → 记录 transition
      3. 返回所有 transition 和总奖励

    一个 episode 的步数 = 可调度节点数（如 GoogLeNet = 197 步）。
    """
    env.reset()
    x = gs.x.to(device)

    # 静态编码 — 整个 episode 只算一次（GATv2 两层前向传播）
    with torch.no_grad():
        h_static = policy.encode_static(x, parents=gs.parents, children=gs.children)

    transitions: List[Transition] = []
    total_reward = 0.0

    while not env.is_done():
        # 获取当前步的动态特征
        dyn_node = env.dynamic_node_features().to(device)   # [N, 10]
        glob = env.global_features().to(device)             # [12]
        ready_mask = env.ready_mask().to(device)             # [N]

        # 策略网络做决策（不计算梯度，因为 PPO 用的是 old_log_prob）
        with torch.no_grad():
            dist, value = policy.act(h_static, dyn_node, glob, ready_mask)
        action = dist.sample()                               # 从概率分布中采样
        log_prob = dist.log_prob(action)                     # 记录采样概率

        # 环境执行动作
        result = env.step(int(action.item()))

        # 记录这一步的所有信息
        transitions.append(Transition(
            h_static=h_static.detach(),
            dyn_node=dyn_node.detach(),
            glob=glob.detach(),
            ready_mask=ready_mask.detach(),
            action=int(action.item()),
            log_prob=float(log_prob.item()),
            value=float(value.item()),
            reward=result.reward,
        ))
        total_reward += result.reward

    return transitions, total_reward


# ======================================================================
# PPO 更新 — 用收集的数据更新策略网络
# ======================================================================

def ppo_update(
    policy: DynamicActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: List[Transition],
    advantages: List[float],
    returns: List[float],
    cfg: TrainConfig,
    device: torch.device,
    entropy_coef: Optional[float] = None,
) -> Dict[str, float]:
    """PPO Clipped Surrogate 更新 — 向量化 mini-batch 版本。

    核心改进（相比原逐步版本）：
      1. 将所有 transition 预先 stack 为张量，避免 Python 循环
      2. 随机打乱 + 切分 mini-batch，每个梯度步只用一部分数据
      3. 调用 policy.batch_evaluate_actions() 一次前向传播处理整个 mini-batch

    公式不变：
      ratio = π_new(a|s) / π_old(a|s)
      L_clip = min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)
      L_total = -L_clip + c1·L_value - c2·L_entropy
    """
    T = len(transitions)
    if T == 0:
        return {'pg_loss': 0.0, 'v_loss': 0.0, 'entropy': 0.0}

    adv_t = torch.tensor(advantages, dtype=torch.float32, device=device)
    ret_t = torch.tensor(returns, dtype=torch.float32, device=device)
    old_lp = torch.tensor([tr.log_prob for tr in transitions], dtype=torch.float32, device=device)
    actions = torch.tensor([tr.action for tr in transitions], dtype=torch.long, device=device)

    # 预先 stack 为 [T, ...] 张量，用于向量化索引
    h_all = torch.stack([tr.h_static for tr in transitions])       # [T, N, emb]
    dyn_all = torch.stack([tr.dyn_node for tr in transitions])     # [T, N, D_DYN]
    glob_all = torch.stack([tr.glob for tr in transitions])        # [T, D_GLOBAL]
    mask_all = torch.stack([tr.ready_mask for tr in transitions])  # [T, N]

    # ★ 优势归一化：跨所有 episode 的全 batch 归一化，这是收敛的关键
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    ec = float(entropy_coef) if entropy_coef is not None else float(cfg.entropy_coef)
    mbs = min(cfg.mini_batch_size, T)

    total_pg = 0.0
    total_v = 0.0
    total_ent = 0.0
    n_grad_steps = 0

    step_stats: List[Dict[str, float]] = []

    for _ in range(cfg.ppo_epochs):
        perm = np.random.permutation(T)
        for start in range(0, T, mbs):
            idx = torch.tensor(perm[start:start + mbs], dtype=torch.long, device=device)

            lp, val, ent = policy.batch_evaluate_actions(
                h_all[idx], dyn_all[idx], glob_all[idx], mask_all[idx],
                actions[idx],
            )

            ratio = torch.exp(lp - old_lp[idx])
            surr1 = ratio * adv_t[idx]
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_t[idx]

            pg_loss = -torch.min(surr1, surr2).mean()
            v_loss = (val - ret_t[idx]).pow(2).mean()
            ent_mean = ent.mean()

            loss = pg_loss + cfg.value_coef * v_loss - ec * ent_mean

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            pg_val = pg_loss.item()
            v_val = v_loss.item()
            ent_val = ent_mean.item()

            total_pg += pg_val
            total_v += v_val
            total_ent += ent_val
            n_grad_steps += 1

            step_stats.append({'pg_loss': pg_val, 'v_loss': v_val, 'entropy': ent_val})

    del h_all, dyn_all, glob_all, mask_all

    return {
        'pg_loss': total_pg / max(n_grad_steps, 1),
        'v_loss': total_v / max(n_grad_steps, 1),
        'entropy': total_ent / max(n_grad_steps, 1),
        'step_stats': step_stats,
    }


# ======================================================================
# BC 预训练 — 行为克隆，模仿启发式教师
# ======================================================================

def _tcas_teacher_order(gs) -> List[int]:
    """用 TCAS 启发式算法生成教师调度顺序。

    TCAS (Time-Constrained Scheduler) 基于依赖值排序，
    是一个已知效果不错的启发式方法。
    """
    from Opara.TimeConstrainedScheduler import DepValueScheduler

    dag_edges = {
        gs.node_names[i]: [gs.node_names[c] for c in gs.children[i]]
        for i in range(len(gs.node_names))
    }
    sched = DepValueScheduler()
    sched.build_graph(dag_edges)
    order_names = sched.schedule()

    name_to_id = {n: i for i, n in enumerate(gs.node_names)}
    return [name_to_id[n] for n in order_names if name_to_id.get(n) is not None]


def bc_pretrain(
    policy: DynamicActorCritic,
    gs,
    cfg: TrainConfig,
    device: torch.device,
) -> List[float]:
    """行为克隆预训练：让策略网络模仿 TCAS 教师的调度顺序。

    原理：
      对于教师选择的每一步动作 a*，最大化 log π(a*|s)。
      相当于监督学习的交叉熵损失。

    好处：
      - PPO 从随机策略开始太慢（197 步的组合空间太大）
      - BC 让策略快速学到一个"及格"的初始策略
      - 之后 PPO 在此基础上进一步优化，超越教师
    """
    teacher_order = _tcas_teacher_order(gs)
    movable_set = set(i for i in range(len(gs.node_names)) if gs.movable_mask[i].item() == 1.0)
    teacher_movable = [v for v in teacher_order if v in movable_set]

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.bc_lr)
    losses: List[float] = []

    x = gs.x.to(device)

    for ep in range(cfg.bc_episodes):
        env = SchedulingEnv(gs, n_streams=cfg.n_streams, device=device, reward_weights=cfg.reward_weights)
        env.reset()

        h_static = policy.encode_static(x, parents=gs.parents, children=gs.children)
        ep_loss = 0.0
        steps = 0

        # 按教师顺序逐步训练
        for target_action in teacher_movable:
            if env.is_done():
                break

            ready_mask = env.ready_mask().to(device)
            if ready_mask[target_action].item() != 1.0:
                continue  # 教师的选择当前不 ready，跳过

            dyn_node = env.dynamic_node_features().to(device)
            glob = env.global_features().to(device)

            dist, _ = policy.act(h_static, dyn_node, glob, ready_mask)
            # 损失 = -log π(教师动作)，即交叉熵
            loss = -dist.log_prob(torch.tensor(target_action, device=device))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

            h_static = h_static.detach()  # 阻断梯度传播到 StaticEncoder

            env.step(target_action)
            ep_loss += float(loss.item())
            steps += 1

        avg_loss = ep_loss / max(steps, 1)
        losses.append(avg_loss)
        print(f"[BC ep {ep:03d}] loss={avg_loss:.4f} steps={steps}")

    return losses


# ======================================================================
# 真实延迟评估 — 连接 RL 策略与 GPU 硬件
# ======================================================================

def _opara_baseline_latency(model, inputs, bench_iters, bench_warmups) -> float:
    """测量 Opara 启发式算法的延迟（基线，只测一次）。"""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message=r'Trying to prepend a node to itself\..*', category=UserWarning)
        run = GraphCapturer.capturer(inputs, model, use_tcas=False)
    return measure_latency_ms(run, inputs=inputs, iterations=bench_iters, warmups=bench_warmups)


def _real_latency_for_order(model, inputs, order_names, bench_iters, bench_warmups) -> float:
    """测量给定调度顺序的真实延迟（从头编译，较慢）。"""
    runner, _, _ = capturer_gnn(inputs=inputs, model=model, schedule_order=order_names, copy_outputs=False)
    return measure_latency_ms(runner, inputs=inputs, iterations=bench_iters, warmups=bench_warmups)


# ======================================================================
# Surrogate 模式训练（备用，用模拟奖励）
# ======================================================================

def train_policy(
    model_factory: Callable[[], Tuple[object, Tuple[torch.Tensor, ...]]],
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
    seed: int = 0,
    save_path: Optional[str] = None,
) -> Dict:
    """Surrogate 模式训练: BC预训练 → PPO(模拟奖励) → 可选真实微调。

    这是备用模式，reward 来自环境模拟而非真实 GPU。
    推荐使用 train_policy_real() 代替。
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, inputs = model_factory()
    assert isinstance(inputs, (tuple, list))

    fx_module = extract_first_fx_graph(model, inputs)
    fx_module.cuda()

    model_class_name = model.__class__.__name__
    node_profiles, device_props = OperatorLauncher.recompile(
        model_class_name, fx_module, inputs, apply_opara_schedule=False,
    )

    gs = build_graph_state(fx_module.graph, node_profiles=node_profiles, device_props=device_props)

    policy = DynamicActorCritic(
        static_in_dim=D_STATIC,
        hidden_dim=cfg.hidden_dim,
        emb_dim=cfg.emb_dim,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
    ).to(device)

    history: Dict = {'bc_losses': [], 'episodes': []}

    # Stage 1: BC 预训练
    if cfg.bc_episodes > 0:
        print(f"\n{'='*60}\nStage 1: Behaviour Cloning ({cfg.bc_episodes} episodes)\n{'='*60}")
        bc_losses = bc_pretrain(policy, gs, cfg, device)
        history['bc_losses'] = bc_losses

    # Stage 2: PPO（模拟环境）
    print(f"\n{'='*60}\nStage 2: PPO ({cfg.episodes} episodes)\n{'='*60}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)

    for ep in range(cfg.episodes):
        env = SchedulingEnv(gs, n_streams=cfg.n_streams, device=device, reward_weights=cfg.reward_weights)

        transitions, total_reward = collect_rollout(policy, gs, env, device)

        values = [tr.value for tr in transitions]
        rewards = [tr.reward for tr in transitions]
        advantages, returns = compute_gae(
            rewards, values, last_value=0.0,
            gamma=cfg.gamma, lam=cfg.gae_lambda,
        )

        ent_sched = scheduled_entropy_coef(cfg, ep, cfg.episodes)
        stats = ppo_update(
            policy, optimizer, transitions, advantages, returns, cfg, device,
            entropy_coef=ent_sched,
        )

        mk = env.current_makespan()
        stats.pop('step_stats', None)
        rec = {
            'episode': int(ep),
            'total_reward': float(total_reward),
            'makespan_sim': float(mk),
            'entropy_coef_used': float(ent_sched),
            **stats,
        }
        history['episodes'].append(rec)
        print(
            f"[PPO ep {ep:04d}] reward={total_reward:+.4f} makespan={mk:.2f} "
            f"pg={stats['pg_loss']:.4f} v={stats['v_loss']:.4f} ent={stats['entropy']:.3f}"
        )

    # Stage 3: 真实延迟微调（可选）
    if cfg.real_finetune_episodes > 0:
        print(f"\n{'='*60}\nStage 3: Real Latency Fine-tune ({cfg.real_finetune_episodes} episodes)\n{'='*60}")

        L_opara = _opara_baseline_latency(model, inputs, cfg.bench_iters, cfg.bench_warmups)
        history['L_opara_ms'] = float(L_opara)

        ft_optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr * 0.1)

        for ep in range(cfg.real_finetune_episodes):
            env = SchedulingEnv(gs, n_streams=cfg.n_streams, device=device, reward_weights=cfg.reward_weights)

            transitions, _ = collect_rollout(policy, gs, env, device)
            order_ids = env.scheduled_order()
            order_names = [gs.node_names[i] for i in order_ids if gs.movable_mask[i].item() == 1.0]

            L_gnn = _real_latency_for_order(model, inputs, order_names, cfg.bench_iters, cfg.bench_warmups)

            real_reward = (L_opara - L_gnn) / max(L_opara, 1e-9)

            log_probs = []
            x = gs.x.to(device)
            h_static = policy.encode_static(x, parents=gs.parents, children=gs.children)
            env2 = SchedulingEnv(gs, n_streams=cfg.n_streams, device=device, reward_weights=cfg.reward_weights)
            env2.reset()
            for tr in transitions:
                dyn_node = env2.dynamic_node_features().to(device)
                glob = env2.global_features().to(device)
                ready_mask = env2.ready_mask().to(device)
                dist, _ = policy.act(h_static, dyn_node, glob, ready_mask)
                log_probs.append(dist.log_prob(torch.tensor(tr.action, device=device)))
                env2.step(tr.action)

            loss = sum(-lp * real_reward for lp in log_probs)
            ft_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            ft_optimizer.step()

            print(f"[FT ep {ep:03d}] L_opara={L_opara:.3f}ms L_gnn={L_gnn:.3f}ms real_reward={real_reward:+.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save({'state_dict': policy.state_dict(), 'config': cfg, 'history': history}, save_path)
        print(f"\nSaved policy to: {save_path}")

        plots_dir = os.path.join(os.path.dirname(save_path) or '.', 'plots', model_class_name)
        plot_training_curves(history, save_dir=plots_dir, model_name=model_class_name)

    return history


# ======================================================================
# ★ 真实延迟 PPO 训练（推荐模式，直接优化真实 GPU 延迟）
# ======================================================================

def train_policy_real(
    model_factory: Callable[[], Tuple[object, Tuple[torch.Tensor, ...]]],
    cfg: TrainConfig,
    device: Optional[torch.device] = None,
    seed: int = 0,
    save_path: Optional[str] = None,
) -> Dict:
    """★ 推荐的训练模式：PPO + 真实 GPU 延迟作为奖励信号。

    与 surrogate 模式的关键区别：
    - 环境仍用于提供动态特征（ready 集合、流状态等）
    - 但 reward 完全来自真实 GPU：capture CUDA Graph → benchmark
    - 不存在 sim-to-real 的 gap，直接优化真实延迟

    奖励设计（terminal-only）：
      只在 episode 最后一步给奖励，中间步骤 reward=0
      R = (L_opara - L_gnn) / L_opara × 100  （百分比改善）
      正值 = 比 Opara 快，负值 = 比 Opara 慢

      配合 gamma=1.0，GAE 会将终端奖励正确传播到每一步：
      所有步的 return ≈ R，优势 ≈ R - V(s_t)

    典型性能（H20 GPU, GoogLeNet ~200 节点）：
      每 episode: rollout 0.05s + capture+bench 0.25s + PPO 0.10s ≈ 0.4s
      500 episodes ≈ 3.5 分钟
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, inputs = model_factory()
    assert isinstance(inputs, (tuple, list))

    # ---- 一次性准备：编译 FX 图 + 算子 profiling ----
    print(f"\n{'='*60}")
    print("Compiling FX graph and operator profiles (one-time cost) ...")
    print(f"{'='*60}")
    fx_module = extract_first_fx_graph(model, inputs)
    fx_module.cuda()

    model_class_name = model.__class__.__name__
    node_profiles, device_props = OperatorLauncher.recompile(
        model_class_name, fx_module, inputs, apply_opara_schedule=False,
    )

    gs = build_graph_state(
        fx_module.graph, node_profiles=node_profiles, device_props=device_props,
    )
    n_movable = int(gs.movable_mask.sum().item())
    print(f"  Graph: {len(gs.node_names)} nodes, {n_movable} movable")

    # ---- 一次性：测量 Opara 基线延迟 ----
    print("Measuring Opara baseline latency ...")
    L_opara = _opara_baseline_latency(model, inputs, cfg.bench_iters, cfg.bench_warmups)
    print(f"  Opara baseline: {L_opara:.4f} ms")

    # ---- 初始化策略网络 ----
    policy = DynamicActorCritic(
        static_in_dim=D_STATIC,
        hidden_dim=cfg.hidden_dim,
        emb_dim=cfg.emb_dim,
        n_heads=cfg.n_heads,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy params: {n_params:,}")

    history: Dict = {
        'bc_losses': [],
        'episodes': [],
        'L_opara_ms': float(L_opara),
        'model': model_class_name,
    }

    # ------------------------------------------------------------------
    # Stage 1: BC 预训练（可选，推荐用于加速收敛）
    # ------------------------------------------------------------------
    if cfg.bc_episodes > 0:
        print(f"\n{'='*60}")
        print(f"Stage 1: Behaviour Cloning ({cfg.bc_episodes} episodes)")
        print(f"{'='*60}")
        bc_losses = bc_pretrain(policy, gs, cfg, device)
        history['bc_losses'] = bc_losses

    # ------------------------------------------------------------------
    # Stage 2: PPO + 真实 GPU 延迟（核心训练循环）
    #   ★ 关键改进：每 batch_episodes 个 episode 做一次 PPO 更新
    #     而非每个 episode 更新一次，显著提升梯度信号质量
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Stage 2: PPO with Real GPU Latency")
    print(f"  episodes={cfg.episodes}  batch={cfg.batch_episodes}  "
          f"mini_batch={cfg.mini_batch_size}")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    n_updates = max(cfg.episodes // max(cfg.batch_episodes, 1), 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_updates, eta_min=cfg.lr * 0.1,
    )

    best_L = L_opara
    best_speedup = 0.0

    # 批量缓冲区
    batch_transitions: List[Transition] = []
    batch_advantages: List[float] = []
    batch_returns: List[float] = []
    pending_recs: List[Dict] = []
    update_count = 0
    history['grad_steps'] = []

    for ep in range(cfg.episodes):
        # ---- 2a. Rollout ----
        env = SchedulingEnv(
            gs, n_streams=cfg.n_streams, device=device,
            reward_weights=cfg.reward_weights,
        )
        transitions, _ = collect_rollout(policy, gs, env, device)

        # ---- 2b. 提取调度顺序 ----
        order_ids = env.scheduled_order()
        order_names = [
            gs.node_names[i] for i in order_ids
            if gs.movable_mask[i].item() == 1.0
        ]

        # ---- 2c. 真实 GPU 评估 ----
        try:
            L_gnn = fast_eval_latency(
                fx_module, inputs, order_names,
                iterations=cfg.bench_iters, warmups=cfg.bench_warmups,
            )
        except Exception as e:
            print(f"  [ep {ep:04d}] CUDA capture failed: {e}")
            torch.cuda.empty_cache()
            continue

        # ---- 2d. 计算真实奖励 + GAE (但先不更新策略) ----
        real_reward = (L_opara - L_gnn) / max(L_opara, 1e-9) * 100.0
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
        speedup = real_reward
        if L_gnn < best_L:
            best_L = L_gnn
            best_speedup = speedup
            if save_path:
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                torch.save({
                    'state_dict': policy.state_dict(),
                    'config': cfg,
                    'history': history,
                    'best_latency_ms': float(best_L),
                    'opara_latency_ms': float(L_opara),
                    'hidden_dim': cfg.hidden_dim,
                    'emb_dim': cfg.emb_dim,
                    'n_heads': cfg.n_heads,
                }, save_path)

        pending_recs.append({
            'episode': int(ep),
            'L_gnn_ms': float(L_gnn),
            'L_opara_ms': float(L_opara),
            'speedup_pct': float(speedup),
            'best_speedup_pct': float(best_speedup),
        })

        beat_opara = " ★ BEAT" if L_gnn < L_opara else ""
        print(
            f"  [ep {ep:04d}] L_gnn={L_gnn:.4f}ms  Opara={L_opara:.4f}ms  "
            f"Δ={speedup:+.2f}%  best={best_speedup:+.2f}%{beat_opara}"
        )

        # ---- 2e. 收集满 batch_episodes 后做一次 PPO 更新 ----
        batch_full = len(pending_recs) >= cfg.batch_episodes
        is_last = (ep == cfg.episodes - 1)

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
            history['episodes'].extend(pending_recs)

            base_step = len(history['grad_steps'])
            for i, ss in enumerate(per_step):
                ss['global_step'] = base_step + i
                ss['ppo_update'] = update_count
                ss['episode'] = int(ep)
            history['grad_steps'].extend(per_step)

            print(
                f"  >>> PPO update #{update_count} "
                f"({len(pending_recs)} eps, {len(batch_transitions)} steps) "
                f"pg={stats['pg_loss']:.4f} v={stats['v_loss']:.4f} "
                f"ent={stats['entropy']:.3f} lr={cur_lr:.2e}"
            )

            # ---- 自动落盘：即使训练被中途停止，也能保留历史并重新画图 ----
            # 文件名示例：inception_v3_real_latest.pt
            # 使用 examples/replot.py --checkpoint *_latest.pt 可重新生成曲线。
            if save_path and cfg.autosave_interval > 0 and update_count % cfg.autosave_interval == 0:
                latest_path = _checkpoint_path(save_path, 'latest')
                _save_training_checkpoint(
                    latest_path,
                    policy=policy,
                    cfg=cfg,
                    history=history,
                    best_L=best_L,
                    L_opara=L_opara,
                    model_class_name=model_class_name,
                    update_count=update_count,
                    status='running',
                )
                print(f"  >>> autosaved latest history → {latest_path}")

            batch_transitions = []
            batch_advantages = []
            batch_returns = []
            pending_recs = []

    # ------------------------------------------------------------------
    # 训练总结
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Training complete")
    print(f"{'='*60}")
    print(f"  Opara latency:   {L_opara:.4f} ms")
    print(f"  Best GNN latency: {best_L:.4f} ms")
    print(f"  Best speedup:    {best_speedup:+.2f}%")
    if best_L < L_opara:
        print(f"  *** GNN policy BEATS Opara! ***")
    else:
        print(f"  Gap to Opara:    {L_opara - best_L:.4f} ms")

    if save_path:
        final_path = save_path.replace('.pt', '_final.pt')
        _save_training_checkpoint(
            final_path,
            policy=policy,
            cfg=cfg,
            history=history,
            best_L=best_L,
            L_opara=L_opara,
            model_class_name=model_class_name,
            update_count=update_count,
            status='complete',
        )
        print(f"\n  Best  policy → {save_path}")
        print(f"  Final policy → {final_path}")

    # ------------------------------------------------------------------
    # 绘制训练曲线（保存到 save_path 同目录下的 plots/ 子目录）
    # ------------------------------------------------------------------
    if save_path:
        plots_dir = os.path.join(os.path.dirname(save_path) or '.', 'plots', model_class_name)
    else:
        plots_dir = os.path.join(_GNN_STRATEGY_ROOT, 'artifacts', 'plots', 'default')
    plot_training_curves(history, save_dir=plots_dir, model_name=model_class_name)

    return history
