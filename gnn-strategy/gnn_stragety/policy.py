"""动态 Actor-Critic 策略网络 — 强化学习的「大脑」。

============================================================================
整体架构：

  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │   静态特征 x [N,16]                                              │
  │        ↓                                                         │
  │   ┌─────────────┐     每个 episode 只算一次                       │
  │   │StaticEncoder│     2层 GATv2 图注意力                         │
  │   │ (GATv2×2)   │     第1层: 聚合父节点信息                      │
  │   └─────┬───────┘     第2层: 聚合子节点信息                      │
  │         ↓                                                        │
  │   h_static [N, emb]   静态嵌入（整个episode不变）                 │
  │         ↓                                                        │
  │   ╔═════════════════════════════════════════╗                    │
  │   ║  以下每一步（step）都会重新计算          ║                    │
  │   ╚═════════════════════════════════════════╝                    │
  │   ┌───────────────┐                                              │
  │   │DynamicFusion  │  融合: [h_static, dyn_node, global]          │
  │   │ (2层 MLP)     │  输入: emb + 10 + 12 = emb+22 维            │
  │   └──────┬────────┘  输出: h_dyn [N, emb]                       │
  │          ↓                                                       │
  │    ┌─────┴──────┐                                                │
  │    ↓            ↓                                                │
  │  ┌──────┐  ┌──────────┐                                         │
  │  │Actor │  │  Critic  │                                         │
  │  │ Head │  │   Head   │                                         │
  │  └──┬───┘  └────┬─────┘                                         │
  │     ↓           ↓                                                │
  │  动作分布     状态价值                                             │
  │  π(a|s)       V(s)                                               │
  │  (在 ready    (标量,                                              │
  │   节点上的     评估当前                                            │
  │   概率分布)    状态好坏)                                           │
  └──────────────────────────────────────────────────────────────────┘

关键设计点：
  1. StaticEncoder 每 episode 只跑一次（图拓扑不变），减少计算开销
  2. DynamicFusion 每步都跑，融合实时调度状态
  3. Actor 输出的是在 ready 集合上的概率分布（非 ready 节点 mask 为 -∞）
  4. Critic 输出 V(s)，用于 PPO 的 advantage 计算

超参数：
  hidden_dim:  GAT 隐藏层维度 [可调超参，默认128]
  emb_dim:     节点嵌入维度 [可调超参，默认128]
  n_heads:     GAT 注意力头数 [可调超参，默认4]
  dropout:     Dropout 比例 [可调超参，默认0.1]
============================================================================
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import D_DYN, D_GLOBAL
from .gat import GATv2Layer
from .graph_state import D_STATIC


class StaticEncoder(nn.Module):
    """静态图编码器：两层 GATv2 注意力网络。

    第 1 层（direction='parents'）：每个节点聚合其父节点（上游依赖）的信息
    第 2 层（direction='children'）：每个节点聚合其子节点（下游影响）的信息

    经过两层后，每个节点的嵌入同时包含了上游和下游的结构信息。
    """

    def __init__(self, in_dim: int, hidden_dim: int, emb_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat1 = GATv2Layer(in_dim, hidden_dim, direction='parents', n_heads=n_heads, dropout=dropout)
        self.gat2 = GATv2Layer(hidden_dim, emb_dim, direction='children', n_heads=n_heads, dropout=dropout)

    def forward(self, x: torch.Tensor, parents: List[List[int]], children: List[List[int]]) -> torch.Tensor:
        h = self.gat1(x, parents=parents, children=children)  # [N, hidden_dim]
        h = self.gat2(h, parents=parents, children=children)  # [N, emb_dim]
        return h


class DynamicFusion(nn.Module):
    """动态融合层：将静态嵌入 + 动态节点特征 + 全局特征融合为决策特征。

    输入维度：emb_dim + D_DYN(10) + D_GLOBAL(12)
    输出维度：emb_dim

    全局特征通过 broadcast 扩展到每个节点，让每个节点都能感知全局状态。
    """

    def __init__(self, emb_dim: int, dyn_dim: int, global_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + dyn_dim + global_dim, out_dim),  # 拼接 → 降维
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, h_static: torch.Tensor, dyn: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        N = h_static.shape[0]
        glob_expanded = glob.unsqueeze(0).expand(N, -1)  # [D_GLOBAL] → [N, D_GLOBAL]
        cat = torch.cat([h_static, dyn, glob_expanded], dim=-1)  # [N, emb+10+12]
        return self.mlp(cat)                                       # [N, emb]


class ActorHead(nn.Module):
    """Actor（策略头）：输出在 ready 节点上的动作概率分布。

    对每个节点打分 → 非 ready 节点 mask 为 -∞ → softmax → Categorical 分布。
    训练时从该分布采样（探索），推理时可以取 argmax（贪心）。
    """

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),          # 输出每个节点的标量得分
        )

    def forward(self, h: torch.Tensor, ready_mask: torch.Tensor) -> torch.distributions.Categorical:
        scores = self.net(h).squeeze(-1)       # [N] 每个节点的得分
        neg_inf = torch.finfo(scores.dtype).min

        # 将非 ready 节点的得分设为 -∞，确保只从 ready 集合中选择
        masked = torch.where(ready_mask > 0.0, scores, torch.full_like(scores, neg_inf))

        if torch.isneginf(masked).all():
            # 极端情况：没有 ready 节点，均匀分布
            probs = torch.ones_like(scores) / float(scores.numel())
        else:
            probs = F.softmax(masked, dim=0)   # 归一化为概率分布

        return torch.distributions.Categorical(probs=probs)


class CriticHead(nn.Module):
    """Critic（价值头）：输出当前状态的价值估计 V(s)。

    将所有 ready 节点的特征加权池化（attention pooling），
    拼接全局特征，通过 MLP 输出标量值。
    """

    def __init__(self, node_dim: int, global_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),          # 输出标量 V(s)
        )

    def forward(self, h_dyn: torch.Tensor, glob: torch.Tensor, ready_mask: torch.Tensor) -> torch.Tensor:
        # 用 ready_mask 做加权平均池化（只关注 ready 节点）
        weights = ready_mask / ready_mask.sum().clamp(min=1.0)
        pooled = (h_dyn * weights.unsqueeze(-1)).sum(dim=0)   # [emb]
        return self.net(torch.cat([pooled, glob], dim=-1)).squeeze(-1)  # 标量


class DynamicActorCritic(nn.Module):
    """完整策略网络：StaticEncoder + DynamicFusion + Actor + Critic。

    这是整个 RL 系统的「大脑」，决定每一步调度哪个算子。

    超参数：
        static_in_dim: 静态特征维度 (D_STATIC=16, 不需调)
        hidden_dim:    隐藏层维度 [可调，默认128，增大→容量更大但更慢]
        emb_dim:       嵌入维度 [可调，默认128，与 hidden_dim 保持一致即可]
        n_heads:       注意力头数 [可调，默认4，试过 2/4/8]
        dropout:       Dropout [可调，默认0.1，过拟合时增大]
    """

    def __init__(
        self,
        static_in_dim: int = D_STATIC,
        hidden_dim: int = 64,
        emb_dim: int = 64,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = StaticEncoder(static_in_dim, hidden_dim, emb_dim, n_heads=n_heads, dropout=dropout)
        self.fusion = DynamicFusion(emb_dim, D_DYN, D_GLOBAL, emb_dim, dropout=dropout)
        self.actor = ActorHead(emb_dim, hidden_dim)
        self.critic = CriticHead(emb_dim, D_GLOBAL, hidden_dim)

        self._emb_dim = emb_dim

    def encode_static(self, x: torch.Tensor, parents: List[List[int]], children: List[List[int]]) -> torch.Tensor:
        """编码静态特征 → 静态嵌入。每 episode 只调用一次。"""
        return self.encoder(x, parents, children)

    def act(
        self,
        h_static: torch.Tensor,       # [N, emb] 静态嵌入（episode 开头算好的）
        dyn_node: torch.Tensor,        # [N, 10]  动态节点特征（每步更新）
        glob: torch.Tensor,            # [12]     全局特征（每步更新）
        ready_mask: torch.Tensor,      # [N]      ready 掩码（每步更新）
    ) -> Tuple[torch.distributions.Categorical, torch.Tensor]:
        """每步决策：返回 (动作分布, 状态价值)。"""
        h_dyn = self.fusion(h_static, dyn_node, glob)   # 融合动态信息
        dist = self.actor(h_dyn, ready_mask)              # Actor: 动作概率
        value = self.critic(h_dyn, glob, ready_mask)      # Critic: V(s)
        return dist, value

    def evaluate_actions(
        self,
        h_static: torch.Tensor,
        dyn_node: torch.Tensor,
        glob: torch.Tensor,
        ready_mask: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PPO 更新时使用：给定动作，返回 (log_prob, value, entropy)。

        与 act() 的区别：act() 是采样动作，这里是评估已知动作的概率。
        PPO 需要比较「采集数据时的 log_prob」和「当前策略的 log_prob」。
        """
        h_dyn = self.fusion(h_static, dyn_node, glob)
        dist = self.actor(h_dyn, ready_mask)
        value = self.critic(h_dyn, glob, ready_mask)
        log_prob = dist.log_prob(actions)      # 动作的对数概率
        entropy = dist.entropy()               # 分布的熵（衡量探索程度）
        return log_prob, value, entropy

    # ------------------------------------------------------------------
    # 向量化批量评估 — PPO mini-batch 更新时使用，比逐步循环快一个量级
    # ------------------------------------------------------------------

    def batch_evaluate_actions(
        self,
        h_static_batch: torch.Tensor,     # [B, N, emb]
        dyn_node_batch: torch.Tensor,     # [B, N, D_DYN]
        glob_batch: torch.Tensor,         # [B, D_GLOBAL]
        ready_mask_batch: torch.Tensor,   # [B, N]
        actions_batch: torch.Tensor,      # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """向量化版本的 evaluate_actions，一次处理整个 mini-batch。

        所有 nn.Linear / nn.LayerNorm 天然支持 batch 维度广播，
        因此直接将 [B, N, dim] 喂入 MLP 即可并行计算。
        """
        B, N, _ = h_static_batch.shape

        # ---- Batched Fusion ----
        glob_expanded = glob_batch.unsqueeze(1).expand(B, N, -1)       # [B, N, D_GLOBAL]
        cat = torch.cat([h_static_batch, dyn_node_batch, glob_expanded], dim=-1)
        h_dyn = self.fusion.mlp(cat)                                    # [B, N, emb]

        # ---- Batched Actor ----
        scores = self.actor.net(h_dyn).squeeze(-1)                      # [B, N]
        neg_inf = torch.finfo(scores.dtype).min
        masked = torch.where(
            ready_mask_batch > 0.0, scores, torch.full_like(scores, neg_inf),
        )
        probs = F.softmax(masked, dim=-1)                               # [B, N]
        dist = torch.distributions.Categorical(probs=probs)

        log_prob = dist.log_prob(actions_batch)                         # [B]
        entropy = dist.entropy()                                        # [B]

        # ---- Batched Critic: attention-weighted pool per sample ----
        w = ready_mask_batch / ready_mask_batch.sum(dim=-1, keepdim=True).clamp(min=1.0)
        pooled = (h_dyn * w.unsqueeze(-1)).sum(dim=1)                   # [B, emb]
        value = self.critic.net(
            torch.cat([pooled, glob_batch], dim=-1),
        ).squeeze(-1)                                                   # [B]

        return log_prob, value, entropy
