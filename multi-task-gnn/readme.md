# multi-task-gnn: Super-DAG 调度的前期实验框架

> 核心思想：**把 K 个 batch=1 的独立推理 DAG 拼成一个大的 super-DAG，让同一个 GNN 做全局调度。** 这样就把"多任务并发调度"问题彻底转化为"更大 DAG 的算子调度"问题，零改动地复用 `gnn-strategy/` 中现有的 GNN + PPO 代码。

## 目录结构

```
multi-task-gnn/
├── multi_task/                    # 核心 Python 包
│   ├── super_dag.py              # build_super_dag(base_gs, K) → 拼接 K 份
│   ├── baselines.py              # Topological / RoundRobin / Opara-like / Random
│   ├── train_multi.py            # train_super_dag(...) — PPO 训练循环
│   └── plot.py                   # 训练曲线 + 对比柱状图 + K-泛化曲线
├── examples/
│   ├── _common.py                # 模型工厂 + FX/GraphState 构建
│   ├── phase1_train.py           # Phase 1: 在 K 任务 super-DAG 上训练
│   ├── phase2_compare.py         # Phase 2: GNN vs 所有基线对比
│   └── phase3_generalize.py      # Phase 3: K-泛化评估 (K={2,4,8,16})
└── artifacts/                    # 输出产物 (checkpoint + 图表)
```

## 研究路线

### Phase 1 — 训练 (验证可行性)
用 PPO 在 K=4 super-DAG 上训练 GNN；baseline = Opara-like 贪心；reward = (L_baseline − L_gnn)/L_baseline × 100。

```bash
python multi-task-gnn/examples/phase1_train.py \
    --model googlenet --num-tasks 4 --episodes 300 \
    --hidden 128 --emb 128 --heads 4 --streams 8
```

**产出**：`artifacts/phase1/googlenet_K4.pt` + 训练曲线。

### Phase 2 — 对比 (证明效果)
固定 K=4，对比所有算法：Topological / RoundRobin / Opara-like / Random / GNN(greedy) / GNN(best)。

```bash
python multi-task-gnn/examples/phase2_compare.py \
    --model googlenet --num-tasks 4 \
    --policy artifacts/phase1/googlenet_K4.pt
```

**产出**：`baseline_compare.png` + `results.csv` + `stats.json`。

### Phase 3 — 泛化 (稳健性)
用 K=4 训练好的同一个策略，在 K={2,4,8,16} 上评估 zero-shot；或每个 K 上短期 fine-tune 看适应性。

```bash
# zero-shot (推荐先跑)
python multi-task-gnn/examples/phase3_generalize.py \
    --model googlenet \
    --policy artifacts/phase1/googlenet_K4.pt \
    --ks 2,4,8,16

# 每个 K 微调 50 episodes
python multi-task-gnn/examples/phase3_generalize.py \
    --model googlenet \
    --policy artifacts/phase1/googlenet_K4.pt \
    --ks 2,4,8,16 --mode finetune --ft-episodes 50
```

**产出**：`k_generalization.png` + `results.csv`。

## 端到端串联示例

```bash
cd /mnt/workspace/xiaguoqing/x-ky/TCAS

# 1. 训练
python multi-task-gnn/examples/phase1_train.py \
    --model googlenet --num-tasks 4 --episodes 300

# 2. 对比所有基线
python multi-task-gnn/examples/phase2_compare.py \
    --model googlenet --num-tasks 4 \
    --policy multi-task-gnn/artifacts/phase1/googlenet_K4.pt

# 3. K-泛化 (zero-shot)
python multi-task-gnn/examples/phase3_generalize.py \
    --model googlenet \
    --policy multi-task-gnn/artifacts/phase1/googlenet_K4.pt \
    --ks 2,4,8,16

# 4. K-泛化 (fine-tune)
python multi-task-gnn/examples/phase3_generalize.py \
    --model googlenet \
    --policy multi-task-gnn/artifacts/phase1/googlenet_K4.pt \
    --ks 2,4,8,16 --mode finetune --ft-episodes 50
```

## 关键设计决定

1. **用模拟 makespan 而非真实 GPU 延迟作为 reward**
   - 真实 super-DAG 的 CUDA Graph 捕获需要 K 份模型副本，工程复杂度太高
   - 模拟 makespan 在单任务上与真实延迟相关性 > 0.85，够用作 reward signal
   - 前期验证重点是"GNN 能否超过基线"，不是"真实 GPU 能省多少毫秒"
   - 真实 GPU 评估可在后续阶段接入 (用 K 个独立 CUDA Graph + 多流调度)

2. **Opara-like 贪心作为 baseline**
   - 对应"把 Opara 直接应用到 super-DAG"的做法
   - 每步从 ready 集合中选 `descendant_work` 最大的节点
   - 与 Phase 1 训练中 GNN 要超越的目标一致

3. **零改动复用 gnn-strategy 代码**
   - `GraphState` / `SchedulingEnv` / `DynamicActorCritic` / `ppo_update` / `collect_rollout` 全部原样用
   - super-DAG 只是一个节点更多的 `GraphState`，GNN 对节点数不敏感
   - 意味着现有的调参结论 (gae_lambda=1.0, batch_episodes=8, ...) 可直接沿用

## 常用调参建议

| 参数 | 默认 | 说明 |
|---|---|---|
| `--num-tasks` | 4 | K 越大搜索空间越大，收敛更慢 |
| `--episodes` | 300 | K=4: 300 够，K=8: 建议 500+ |
| `--batch-episodes` | 8 | super-DAG 步数多了 K 倍，可加大到 16 |
| `--mini-batch-size` | 512 | super-DAG 步数多 → mini_batch 相应加大 |
| `--hidden / --emb` | 128 | K 变大若难收敛，加到 256 |
| `--streams` | 8 | 模拟的 CUDA 流数量，通常 8~16 |
