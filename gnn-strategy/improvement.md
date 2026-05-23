# GNN + RL 算子调度改进路线图

> 当前方案：GATv2 静态编码 + 动态融合 MLP + PPO 端到端训练，在单模型推理图上学习算子调度顺序。
>
> 核心瓶颈：(1) GATv2 表达力有限，难以捕获长距离依赖；(2) 标准 PPO 在大规模组合优化问题上探索效率低；(3) 每个模型需要从零训练策略，缺乏跨模型泛化能力。
>
> 本文档梳理 5 个可行改进维度，每个维度包含具体方案、参考文献、预估工作量和优先级，供后续逐步实施。

---

## 一、GNN 架构升级

### 1.1 Graph Transformer 替换 GATv2

**动机**：GAT/GATv2 的注意力权重只在一阶邻域内计算，无法直接建模跨越多跳的依赖关系（如关键路径上相距很远的两个算子）。Graph Transformer 通过全局自注意力 + 图结构位置编码，能捕获任意两个节点之间的关系。

**具体方案**：

```
输入: 节点特征 X ∈ R^{N×d}, 邻接矩阵 A
          ↓
 图结构位置编码（Laplacian PE / Random Walk PE）
          ↓
 Multi-Head Self-Attention（全局） + 图结构偏置（边存在时加 bias）
          ↓
 FFN + Residual + LayerNorm
          ↓
 重复 L 层
          ↓
 输出: h_static ∈ R^{N×d}
```

- 保留当前 16 维静态特征 + 10 维动态特征 + 12 维全局特征的设计
- 将 GATv2 encoder 替换为 Graph Transformer encoder
- 位置编码使用 **Laplacian Eigenvector PE**（捕获拓扑距离）或 **Random Walk PE**（计算更快）
- 在 self-attention 中注入 **图结构偏置**：若 (i,j) 有边则 `attn_bias[i,j] += learnable_edge_bias`

**参考**：
- Rampášek et al., "Recipe for a General, Powerful, Scalable Graph Transformer" (NeurIPS 2022)
- Ying et al., "Do Transformers Really Perform Bad for Graph Representation?" — Graphormer (NeurIPS 2021)
- Gensor (2025)：用图结构建模张量编译空间，Markov 分析图遍历

**工作量**：中等（约 1-2 周）。主要改 `gat.py` → `graph_transformer.py`，其他模块接口不变。

**预期收益**：
- 长距离依赖建模能力提升 → 关键路径感知更强
- 对大图（NASNet、BERT）上的调度质量有明显改善
- 消融实验（GAT vs Graph Transformer）即为论文一个 ablation 点

---

### 1.2 异构图神经网络（Heterogeneous GNN）

**动机**：当前所有算子节点使用相同的消息传递函数，但计算图中的节点和边天然是异构的：
- **节点异构**：conv2d / matmul / layernorm / embedding / relu / concat 等算子类型差异极大
- **边异构**：数据流边（tensor 传递）vs 控制流边（执行依赖）vs 资源竞争边（共享 SM）

**具体方案**：

```
算子类型 → type embedding t_i ∈ R^{d_type}
边类型 → relation embedding r_{ij} ∈ R^{d_rel}

异构消息传递:
  m_{ij} = W_{type(j)} · h_j + r_{ij}
  h_i' = Aggregate({m_{ij} | j ∈ N(i)})

或使用 Relational Graph Transformer:
  Attn(i,j) = softmax( (W_Q^{type(i)} h_i)^T (W_K^{type(j)} h_j + r_{ij}) / √d )
```

- 定义算子类型分类：`{compute_heavy, memory_bound, elementwise, reduction, embedding, normalization, reshape}`
- 定义边类型：`{data_flow, skip_connection, sibling}`
- 每种类型有独立的 W_Q, W_K, W_V 参数矩阵

**参考**：
- ADE-HGNN (2024)：异构图注意力网络加速，利用注意力差异性剪枝
- HiHGNN (2023)：异构图阶段融合方法论
- Hector (2023)：关系图神经网络编译框架，优化算子间调度

**工作量**：中等偏高（约 2-3 周）。需修改特征构建、消息传递层、编码器。

**预期收益**：
- 对 BERT（attention 和 MLP 算子差异大）、DeepFM（embedding 和 MLP 混合）效果显著
- 能区分"这个 matmul 很重要"和"这个 reshape 可以随便放"
- 论文创新点：**首次在 GPU 算子调度中引入异构图建模**

---

### 1.3 层次化图表示（Hierarchical Graph Pooling）

**动机**：当前 GNN 对 N 个节点做平等的消息传递。对 NASNet（~1000+ 节点）或 BERT（~384 节点），每一步 GNN forward 都要处理全图，效率低且信息淹没。

**具体方案**：
- **第一层**：在原始算子粒度做局部消息传递（2-3 层 GATv2/GT）
- **图池化**：用 DiffPool / TopKPool / SAGPool 将图压缩为"算子组"（类似 IOS 的 stage）
- **第二层**：在粗粒度"组图"上做全局消息传递
- **解池化**：将粗粒度信息广播回原始节点
- Actor 在原始节点粒度选择动作

**参考**：
- HetRL (2024)：多级搜索框架（task grouping → GPU grouping → fine mapping）
- GrapheonRL (2025)：GNN+RL 多级工作流调度
- AllReduce Scheduling (2025)：分层 DRL，higher-level 选拓扑 + lower-level 做流调度

**工作量**：高（约 3-4 周）。需要实现图池化层、双层编码器、解池化。

**预期收益**：
- 对大图（NASNet/BERT）的扩展性显著提升
- 自动学习"哪些算子应该一起考虑" → 隐式学习 operator fusion 的分组策略
- 论文创新点：**层次化图编码用于算子调度，桥接 stage 分区和细粒度调度**

---

## 二、强化学习算法改进

### 2.1 Decision Transformer 替代 PPO

**动机**：PPO 在大组合空间中探索效率低（尤其 DeepFM/NASNet），而且需要在线交互——每采一个 schedule 就要真实 benchmark，非常昂贵。Decision Transformer (DT) 将 RL 问题重新表述为序列建模问题，可以从离线数据中学习。

**具体方案**：

```
Phase 1: 数据收集
  - 用 TCAS / Opara / 随机策略 / 已训练的 PPO 策略采集大量 (schedule, latency) 对
  - 构建离线数据集 D = {(s_0, a_0, R_0, s_1, a_1, R_1, ...)}

Phase 2: Decision Transformer 训练
  - 输入: (R̂, s_t, a_t, R̂, s_{t+1}, a_{t+1}, ...)
  - R̂ 是目标 return（期望总延迟的负值）
  - 用 Transformer 自回归预测下一个 action
  - 推理时设置 R̂ = "best known return" 来引导生成高质量调度

Phase 3: 可选的在线微调
  - 用 DT 生成的 schedule 做真实 benchmark
  - 将新数据加入 D，继续训练
```

**参考**：
- GNN-DT (2025)：GNN + Decision Transformer 联合架构，处理动态环境优化
- Decision Transformer for JSSP (2025)：在 Job Shop Scheduling 上超越在线 RL 方法
- GOAL (ICLR 2025)：单一通用模型解决多种组合优化问题

**工作量**：高（约 3-4 周）。需要离线数据收集管线 + Transformer 架构实现。

**预期收益**：
- 完全避免"每个 episode 都要真实 benchmark"的瓶颈 → 训练速度数量级提升
- 天然支持利用多种来源的数据（专家、随机、历史 PPO）
- 在推理时通过调节目标 return 控制"探索-利用"平衡
- 论文创新点：**首次将 Decision Transformer 应用于 GPU 算子调度**

---

### 2.2 分层强化学习（Hierarchical RL）

**动机**：当前的 MDP 是"每步选一个算子"，对 384 节点的 BERT 来说，一个 episode 就是 384 步决策，策略空间巨大。分层 RL 将问题分解为两层决策。

**具体方案**：

```
High-Level Policy (Stage Planner):
  - 输入: 全图嵌入
  - 输出: 将算子分成 K 个 stage（stage 内算子可以并行/乱序）
  - 类似 IOS 的 stage partition，但用 RL 学习而非 DP

Low-Level Policy (Intra-Stage Scheduler):
  - 输入: 单个 stage 内的子图嵌入
  - 输出: stage 内算子的执行顺序
  - 搜索空间大幅缩小

联合训练:
  - High-level reward = 总 makespan
  - Low-level reward = 单 stage latency
  - 用 HIRO / HAM / Option-Critic 等分层 RL 框架
```

**参考**：
- HetRL (2024)：5 级搜索框架（task grouping → GPU grouping → task-GPU mapping → parallelization → fine mapping）
- AllReduce Scheduling (2025)：两级 DRL（higher-level 选树 + lower-level 调度流）
- IOS：DP stage partition（我们用 RL 学习替代 DP，使其更灵活）

**工作量**：高（约 4 周）。需要实现双层策略、子图提取、联合训练。

**预期收益**：
- 搜索空间指数级缩小
- 自然结合了 stage partition（粗粒度）和算子调度（细粒度）
- 对大图（NASNet/BERT）的可扩展性显著提升
- 论文创新点：**将 operator scheduling 分解为 stage partition + intra-stage ordering 的两层 MDP**

---

### 2.3 Mask-PPO：动作空间自适应裁剪

**动机**：当前 PPO 在每一步从所有 ready 算子中采样，但很多 ready 算子实际上是"无关紧要的"（如 reshape、view 等零耗时算子）。Mask-PPO 可以学习动态裁剪动作空间。

**具体方案**：
- 训练一个轻量的"动作评估网络" M(s) → mask ∈ {0,1}^N
- M 预测哪些 ready 算子"值得认真考虑"，其余直接按拓扑序放置
- PPO 只在 mask 内的算子中做策略优化
- 双阶段训练：标准 PPO 探索 + Mask-PPO 精炼

**参考**：
- Preference-Based Mask-PPO (2025)：PBMP 算法，在 FJSP 上优于 OR-Tools 和其他 DRL 方法
- Multi-PPO with Graph Networks (2025)：双编码器-解码器定义操作和机器策略

**工作量**：低至中等（约 1-2 周）。在现有 PPO 基础上增加 mask 网络。

**预期收益**：
- 减少无效探索，加速收敛（尤其 DeepFM 大量 embedding 互换无意义）
- entropy 自然下降（因为有效动作空间变小）
- 实现简单，兼容现有代码

---

## 三、问题建模创新

### 3.1 GNN 代理成本模型（Learned Cost Model）

**动机**：当前每个 episode 都要真实 benchmark（300-500 次 CUDA 执行），这是训练速度的最大瓶颈。如果能用一个 GNN 预测"给定 schedule 的预期延迟"，就可以避免大量真实执行。

**具体方案**：

```
Phase 1: 数据收集
  - 跑 N 个随机/启发式 schedule，记录 (schedule_order, real_latency)
  - 这些数据只需收集一次

Phase 2: 训练代理模型
  - 输入: graph features + schedule order (as node ordering features)
  - 输出: predicted latency
  - 用 GNN regression 或 Transformer regression

Phase 3: RL 训练
  - PPO/DT 的 reward 来自代理模型（不需要真实执行）
  - 训练速度提升 100-1000 倍

Phase 4: 真实微调
  - 用代理模型训好的策略，在真实 GPU 上做少量 fine-tuning
  - Sim-to-Real 迁移
```

**参考**：
- CALO-GNN (2024)：首个带校准不确定性的 GNN 成本模型，用于 TVM Meta-Schedule
- NeuSight (2024)：tile 粒度的 GPU 性能预测，误差从 121% 降至 2.3%
- GraphPerf-RT (2024)：图驱动的实时性能建模 + 校准不确定性
- TVM GNN Cost Model：GNN 嵌入 TensorIR AST 预测执行代价

**工作量**：中等（约 2-3 周）。需要数据收集管线 + regression GNN。

**预期收益**：
- 训练速度数量级提升（从小时级降到分钟级）
- 可以大幅增加探索量 → 更好的最终策略
- 代理模型本身也是一个 contribution（可迁移到新模型/新硬件）
- 论文创新点：**Surrogate-guided RL + Real fine-tuning 的两阶段训练**

---

### 3.2 联合优化：算子融合 + 调度

**动机**：当前工作假设融合决策已经由 PyTorch/编译器做好，GNN 只负责排序。但融合决策和调度顺序是强耦合的：不同的融合方案产生不同的算子图，调度空间也随之改变。

**具体方案**：
- 将 MDP 扩展为两类动作：
  - **融合动作**：选择将哪些相邻算子融合为一个 kernel
  - **调度动作**：选择下一个执行的（可能已融合的）算子
- 交替决策或联合决策
- 参考 IOS 的 stage partition 思路，但用 RL 替代 DP

**参考**：
- Neptune (2025)：高级算子融合（包括 reduction 和 attention），通过代数校正打破依赖
- IOS：DP stage partition + operator fusion
- Operator Fusion Scheduling (IEEE 2023)：TVM 深度学习编译器中的融合调度优化

**工作量**：非常高（约 4-6 周）。需要重新设计环境、动作空间、融合验证。

**预期收益**：
- 突破"固定图上排序"的局限，进入"图结构优化"领域
- 与 IOS 的 DP 方法形成直接对比
- 论文创新点最强：**首次用 RL 端到端联合优化融合与调度**

---

### 3.3 跨模型迁移学习 / 元学习

**动机**：当前每个模型（GoogLeNet/BERT/DeepFM）都要从零训练策略。如果能训练一个通用策略，在新模型上 zero-shot 或 few-shot 迁移，就能大幅扩展方法的实用性。

**具体方案**：

```
方案 A: 多模型联合训练
  - 在 {GoogLeNet, ResNet50, Inception_v3, DeepFM, BERT} 的计算图上混合训练
  - 共享 GNN encoder，策略头也共享
  - 测试：在未见过的模型（如 EfficientNet、T5）上 zero-shot 评估

方案 B: 元学习（MAML / Reptile）
  - 每个模型作为一个 task
  - 学习一组好的初始化参数 θ*
  - 新模型只需 few-shot fine-tuning（10-50 episode）

方案 C: GraphLoRA 式迁移
  - 训练一个大的"基座 GNN"
  - 对新模型只训练低秩适配器（类似 LoRA）
```

**参考**：
- GraphBridge (2025)：跨域 GNN 迁移，桥接网络连接输入输出层
- GraphLoRA (2024)：结构感知的低秩适配用于跨图迁移
- GOAL (ICLR 2025)：单一通用模型解决多种组合优化问题
- MiNT (2024)：多网络预训练提升时序图迁移

**工作量**：中等至高（约 2-4 周）。

**预期收益**：
- 从"model-specific offline tuning"升级为"generalizable scheduling policy"
- 可在论文中做 zero-shot / few-shot 迁移实验
- 论文创新点：**通用算子调度策略，跨模型架构泛化**

---

## 四、系统级工程创新

### 4.1 不确定性感知调度

- 在 GNN 中引入 **Monte Carlo Dropout** 或 **Evidential GNN**
- 对调度决策的置信度进行估计
- 高不确定性时回退到安全的启发式策略（TCAS/Opara）
- 参考 CALO-GNN 的不确定性校准思路

### 4.2 多目标优化

- 当前只优化 latency，可扩展为 **latency + GPU 内存峰值 + 能耗**
- 使用 Pareto-PPO 或 MORL（Multi-Objective RL）
- 生成 Pareto 前沿，用户按需选择

### 4.3 动态批处理感知调度

- 当前只考虑 batch=1 的静态图
- 扩展到 **动态 batch size 的多请求并发调度**
- 与 super-DAG 多任务调度结合

---

## 五、推荐优先级排序

综合考虑**创新性、工作量可控性、与当前代码的兼容性、论文贡献度**，推荐以下实施顺序：

### 第一优先级（最易产出 + 创新性好）

| 方案 | 工作量 | 创新性 | 与现有代码兼容 |
|---|---|---|---|
| **1.1 Graph Transformer** | 1-2 周 | ★★★★ | 高（只换 encoder） |
| **2.3 Mask-PPO** | 1-2 周 | ★★★ | 高（增量改动） |
| **3.1 Learned Cost Model** | 2-3 周 | ★★★★★ | 中（新增数据管线） |

**推荐组合**：Graph Transformer + Learned Cost Model + Mask-PPO

这三个可以构成一篇完整的论文：
1. **Graph Transformer** 替换 GATv2 → 图编码能力提升（网络架构创新）
2. **Learned Cost Model** → Surrogate-guided training + Real fine-tuning（训练方法创新）
3. **Mask-PPO** → 动作空间自适应裁剪（RL 算法创新）
4. 在 5 个模型上的消融实验和对比实验

### 第二优先级（工作量大但创新性很强）

| 方案 | 工作量 | 创新性 |
|---|---|---|
| **1.2 异构 GNN** | 2-3 周 | ★★★★★ |
| **2.1 Decision Transformer** | 3-4 周 | ★★★★★ |
| **3.3 跨模型迁移** | 2-4 周 | ★★★★ |

### 第三优先级（需要大量工程 + 研究）

| 方案 | 工作量 | 创新性 |
|---|---|---|
| **1.3 层次化图表示** | 3-4 周 | ★★★★ |
| **2.2 分层 RL** | 4 周 | ★★★★★ |
| **3.2 联合融合+调度** | 4-6 周 | ★★★★★ |

---

## 六、论文故事线建议

如果目标是一区论文，建议围绕以下故事线构建：

> **标题方向**：Hardware-Aware Graph Transformer with Surrogate-Guided RL for Dynamic Operator Scheduling
>
> **核心叙事**：
> 1. 现有启发式方法（Opara/TCAS）无法适应异构算子图的复杂依赖
> 2. 我们提出 **Graph Transformer** 捕获全局依赖 + **异构消息传递** 区分算子类型
> 3. 用 **Learned Cost Model** 构建高效代理环境，先 surrogate 训练后 real fine-tuning
> 4. **Mask-PPO** 裁剪无效动作空间，加速收敛
> 5. 在 CNN（ResNet/GoogLeNet/Inception）、推荐模型（DeepFM）、Transformer（BERT）上全面验证
> 6. 消融实验证明每个组件的贡献

**实验表格设计**：
| Method | ResNet50 | GoogLeNet | Inception_v3 | DeepFM | BERT |
|---|---|---|---|---|---|
| Opara | baseline | baseline | baseline | baseline | baseline |
| TCAS | ... | ... | ... | ... | ... |
| GATv2 + PPO (ours-base) | ... | ... | ... | ... | ... |
| + Graph Transformer | ... | ... | ... | ... | ... |
| + Heterogeneous msg | ... | ... | ... | ... | ... |
| + Learned Cost Model | ... | ... | ... | ... | ... |
| + Mask-PPO | ... | ... | ... | ... | ... |
| **Full model (ours)** | **...** | **...** | **...** | **...** | **...** |

---

## 参考文献索引

- [Graphormer] Ying et al., NeurIPS 2021
- [GPS] Rampášek et al., NeurIPS 2022
- [Gensor] 2025, graph-based tensor compilation
- [Neptune] 2025, advanced operator fusion
- [GNN-DT] 2025, GNN + Decision Transformer
- [Decision Transformer for JSSP] 2025
- [GOAL] ICLR 2025, generalist combinatorial optimization
- [RESCHED] ICLR 2026 under review, Transformer for FJSP
- [Mask-PPO / PBMP] Complex & Intelligent Systems, 2025
- [Multi-PPO + MPGN] CIMS, 2025
- [HetRL] 2024, multi-level hierarchical scheduling
- [AllReduce HRL] 2025, hierarchical DRL scheduling
- [GrapheonRL] 2025, GNN+RL for heterogeneous HPC
- [CALO-GNN] 2024, uncertainty-aware cost model for TVM
- [NeuSight] 2024, GPU performance forecasting
- [GraphBridge] 2025, arbitrary GNN transfer learning
- [GraphLoRA] 2024, structure-aware low-rank GNN adaptation
- [ADE-HGNN] 2024, heterogeneous GNN acceleration
- [HiHGNN] 2023, bound-aware stage-fusion for HGNN
- [Hector] 2023, relational GNN compilation framework
- [IOS] Inter-Operator Scheduler, DP-based stage partition
