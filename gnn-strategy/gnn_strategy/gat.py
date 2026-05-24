"""GATv2 图注意力网络层 — 不依赖 torch_geometric，纯手写实现。

============================================================================
核心思想：
  GATv2 (Brody et al., 2022) 相比原始 GAT 的改进在于：
  先对特征做线性变换并拼接，再过 LeakyReLU，最后用注意力向量打分。
  这样注意力系数的排序可以依赖于「query 节点」的特征，表达力更强。

在本项目中的作用：
  将 DAG 中每个算子节点的静态特征（16维）编码为高维嵌入（emb_dim维），
  同时通过注意力机制聚合父/子节点信息，让每个节点"知道"它上下游的情况。

多头注意力：
  使用 n_heads 个独立的注意力头，各自学习不同的聚合模式：
  - 某些头可能关注耗时最长的邻居（关键路径信息）
  - 某些头可能关注资源竞争（共享内存/线程数）
  最终将各头的输出 concat 得到 out_dim 维的向量。
============================================================================

超参数说明：
  in_dim      : 输入特征维度 (第一层=D_STATIC=16, 第二层=hidden_dim)
  out_dim     : 输出特征维度 (第一层=hidden_dim, 第二层=emb_dim)
  n_heads     : 注意力头数，默认4。增大→更丰富的聚合模式，但计算量增加
  dropout     : 注意力权重的 dropout 比例，防止过拟合
  negative_slope : LeakyReLU 负半轴斜率
"""

from __future__ import annotations

from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATv2Layer(nn.Module):
    """多头 GATv2 注意力层，适用于中小规模 DAG（几百个节点）。

    ┌─────────────────────────────────────────────────────────────┐
    │  对于节点 v，聚合其邻居 u ∈ N(v) 的信息：                     │
    │                                                             │
    │  每个注意力头 k 的计算：                                      │
    │    e^k_{vu} = (a^k)^T · LeakyReLU( W_l^k·x_v + W_r^k·x_u ) │
    │    α^k_{vu} = softmax_u( e^k_{vu} )                        │
    │    msg^k_v  = Σ_u  α^k_{vu} · W_r^k · x_u                  │
    │                                                             │
    │  多头拼接：                                                   │
    │    h_v = [msg^1_v ‖ msg^2_v ‖ ... ‖ msg^K_v] + W_self·x_v  │
    └─────────────────────────────────────────────────────────────┘

    参数:
        in_dim:       输入节点特征维度
        out_dim:      输出节点特征维度（= n_heads × head_dim）
        direction:    'parents' 聚合父节点 / 'children' 聚合子节点
        n_heads:      注意力头数 [可调超参]
        concat_heads: True=拼接各头输出, False=取平均
        dropout:      注意力 dropout 比例 [可调超参]
        negative_slope: LeakyReLU 斜率
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        direction: Literal['parents', 'children'],
        n_heads: int = 4,
        concat_heads: bool = True,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        assert out_dim % n_heads == 0 or not concat_heads

        self.direction = direction      # 聚合方向：从父节点聚合 or 从子节点聚合
        self.n_heads = n_heads          # 注意力头的数量
        self.concat_heads = concat_heads
        self.negative_slope = negative_slope

        # 每个头的特征维度
        head_dim = out_dim // n_heads if concat_heads else out_dim

        # 三组可学习的线性变换：
        self.W_l = nn.Linear(in_dim, n_heads * head_dim, bias=False)   # 变换 query 节点 v
        self.W_r = nn.Linear(in_dim, n_heads * head_dim, bias=False)   # 变换 key/value 邻居 u
        self.W_self = nn.Linear(in_dim, out_dim, bias=False)           # 自环（残差连接）

        # 注意力打分向量 a^k，shape=[n_heads, head_dim]
        self.attn_vec = nn.Parameter(torch.empty(n_heads, head_dim))
        nn.init.xavier_uniform_(self.attn_vec.unsqueeze(0))

        self.activation = activation if activation is not None else nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.head_dim = head_dim

    def forward(
        self,
        x: torch.Tensor,                  # [N, in_dim]  所有节点的输入特征
        parents: List[List[int]],          # parents[v] = [v 的父节点 id 列表]
        children: List[List[int]],         # children[v] = [v 的子节点 id 列表]
    ) -> torch.Tensor:                     # [N, out_dim] 输出特征
        N = x.shape[0]                     # 节点总数（包括 placeholder 和 output）
        H, D = self.n_heads, self.head_dim

        # ---- 对所有节点做线性变换（一次性矩阵乘法，高效）----
        hl = self.W_l(x).view(N, H, D)    # [N, H, D] query 变换
        hr = self.W_r(x).view(N, H, D)    # [N, H, D] key/value 变换
        h_self = self.W_self(x)            # [N, out_dim] 自环变换

        out_list: list[torch.Tensor] = []

        # ---- 逐节点聚合邻居信息 ----
        for v in range(N):
            # 根据聚合方向选择邻居：第一层聚合父节点，第二层聚合子节点
            neigh = parents[v] if self.direction == 'parents' else children[v]

            if not neigh:
                # 没有邻居（如根节点/叶节点），直接用自环输出
                out_list.append(h_self[v])
                continue

            idx = torch.tensor(neigh, device=x.device, dtype=torch.long)
            hr_neigh = hr[idx]                                   # [deg, H, D] 邻居的 key/value
            hl_v = hl[v].unsqueeze(0).expand(len(neigh), -1, -1) # [deg, H, D] 当前节点的 query

            # GATv2 核心：先加再激活（区别于 GAT 的先激活再加）
            e = F.leaky_relu(hl_v + hr_neigh, self.negative_slope)  # [deg, H, D]
            e = (e * self.attn_vec.unsqueeze(0)).sum(dim=-1)        # [deg, H] 注意力得分

            # softmax 归一化 → 注意力权重 α
            alpha = torch.softmax(e, dim=0)                         # [deg, H]
            alpha = self.dropout(alpha)                             # 训练时随机丢弃部分注意力

            # 加权聚合邻居信息
            msg = (alpha.unsqueeze(-1) * hr_neigh).sum(dim=0)       # [H, D]

            if self.concat_heads:
                msg = msg.reshape(-1)          # [H*D = out_dim] 拼接所有头
            else:
                msg = msg.mean(dim=0)          # [D = out_dim]   取平均

            # 加上自环（残差连接），保留节点自身信息
            out_list.append(msg + h_self[v])

        # 激活函数（默认 ELU）后输出
        return self.activation(torch.stack(out_list, dim=0))
