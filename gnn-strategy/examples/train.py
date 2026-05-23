"""训练入口脚本 — 启动 GNN 调度策略的训练。

============================================================================
使用方法：

  # ★ 推荐：真实延迟 PPO（直接优化真实 GPU makespan）
  python gnn-strategy/examples/train.py \\
      --mode real --model googlenet --episodes 500

  # BC 预训练 + 真实 PPO
  python gnn-strategy/examples/train.py \\
      --mode real --model googlenet --bc-episodes 30 --episodes 500

  # Surrogate PPO（快速原型验证，不需要 GPU benchmark）
  python gnn-strategy/examples/train.py \\
      --mode surrogate --model googlenet --episodes 200

  # 跑其他模型
  python gnn-strategy/examples/train.py \\
      --mode real --model inception_v3 --episodes 500
  python gnn-strategy/examples/train.py \\
      --mode real --model resnet50 --episodes 500
  python gnn-strategy/examples/train.py \\
      --mode real --model deepfm --episodes 500
  python gnn-strategy/examples/train.py \\
      --mode real --model bert_base --episodes 500

支持的模型：googlenet, inception_v3, resnet50, resnet152, deepfm, bert_base
============================================================================
"""

import os
import sys
import argparse

_GNN_STRATEGY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_GNN_STRATEGY_DIR)
_TCAS_EXAMPLES_DIR = os.path.join(_REPO_ROOT, 'examples')
for _p in (_GNN_STRATEGY_DIR, _REPO_ROOT, _TCAS_EXAMPLES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torchvision
import torch.nn as nn

from gnn_strategy.train import TrainConfig, train_policy, train_policy_real


def _project_path(path: str) -> str:
    """Resolve default artifact paths against the gnn-strategy root."""
    if os.path.isabs(path):
        return path
    norm = os.path.normpath(path)
    if norm == 'artifacts' or norm.startswith(f'artifacts{os.sep}'):
        return os.path.join(_GNN_STRATEGY_DIR, norm)
    if norm == 'gnn-strategy' or norm.startswith(f'gnn-strategy{os.sep}'):
        prefix = f'gnn-strategy{os.sep}'
        suffix = norm[len(prefix):] if norm.startswith(prefix) else ''
        return os.path.join(_GNN_STRATEGY_DIR, suffix)
    return os.path.abspath(path)


# ======================================================================
# 模型工厂 — 创建待优化的 PyTorch 模型 + 输入张量
# ======================================================================

def make_googlenet(device: str):
    """GoogLeNet: 经典 Inception 架构，DAG 分支多，调度优化空间大。"""
    x = torch.randint(low=0, high=256, size=(1, 3, 224, 224), dtype=torch.float32, device=device)
    model = torchvision.models.googlenet().to(device=device).eval()
    return model, (x,)


def make_inception_v3(device: str):
    """Inception v3: 更深的 Inception 架构，节点更多。"""
    x = torch.randint(low=0, high=256, size=(1, 3, 299, 299), dtype=torch.float32, device=device)
    model = torchvision.models.inception_v3(aux_logits=False).to(device=device).eval()
    return model, (x,)


def make_resnet50(device: str):
    """ResNet-50: 残差网络，DAG 较简单（主要是链式 + 跳跃连接）。"""
    x = torch.randint(low=0, high=256, size=(1, 3, 224, 224), dtype=torch.float32, device=device)
    model = torchvision.models.resnet50().to(device=device).eval()
    return model, (x,)


def make_resnet152(device: str):
    """ResNet-152: 更深的残差网络。"""
    x = torch.randint(low=0, high=256, size=(1, 3, 224, 224), dtype=torch.float32, device=device)
    model = torchvision.models.resnet152().to(device=device).eval()
    return model, (x,)


def make_deepfm(device: str):
    """DeepFM: 推荐系统模型，embedding + elementwise + MLP，作为轻量非 CNN 参考。"""
    from NCF import DeepFM

    cate_fea_nuniqs = [100 * (i + 1) for i in range(32)]
    nume_fea_size = 16
    model = DeepFM(
        cate_fea_nuniqs,
        nume_fea_size,
        emb_size=8,
        hid_dims=[256, 128],
        num_classes=1,
        dropout=[0.2, 0.2],
    ).to(device=device).eval()

    batch_size = int(os.environ.get('GNN_DEEPFM_BATCH', '1'))
    x_sparse = torch.randint(
        low=0, high=100, size=(batch_size, len(cate_fea_nuniqs)),
        dtype=torch.long, device=device,
    )
    x_dense = torch.rand(batch_size, nume_fea_size, device=device)
    return model, (x_sparse, x_dense)


class BertLastHiddenState(nn.Module):
    """Wrap HuggingFace BERT so FX/CUDA Graph sees a plain Tensor output."""

    def __init__(self, local_path: str):
        super().__init__()
        from transformers import BertModel

        self.bert = BertModel.from_pretrained(local_path, local_files_only=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )[0]


def make_bert_base(device: str):
    """BERT-base: Transformer 代表模型，用于验证非 CNN DAG 的可迁移性。"""
    local_path = os.environ.get(
        'GNN_BERT_BASE_PATH',
        '/mnt/workspace/xiaguoqing/models/bert-base-uncased',
    )
    if not os.path.exists(local_path):
        raise FileNotFoundError(
            f"BERT local path not found: {local_path}. "
            "Set GNN_BERT_BASE_PATH to a local HuggingFace BertModel directory."
        )

    batch_size = int(os.environ.get('GNN_BERT_BATCH', '1'))
    seq_len = int(os.environ.get('GNN_BERT_SEQ_LEN', '256'))
    input_ids = torch.randint(
        low=0, high=30522, size=(batch_size, seq_len),
        dtype=torch.long, device=device,
    )
    attention_mask = torch.ones_like(input_ids, device=device)
    model = BertLastHiddenState(local_path).to(device=device).eval()
    return model, (input_ids, attention_mask)


MODEL_FACTORIES = {
    'googlenet': make_googlenet,
    'inception_v3': make_inception_v3,
    'resnet50': make_resnet50,
    'resnet152': make_resnet152,
    'deepfm': make_deepfm,
    'bert_base': make_bert_base,
}


def main():
    p = argparse.ArgumentParser(description='Train dynamic GNN scheduling policy')

    # ---- 训练模式 ----
    p.add_argument('--mode', type=str, default='real', choices=['real', 'surrogate'],
                   help='训练模式: "real" 用真实 GPU 延迟作为奖励（推荐）; '
                        '"surrogate" 仅用模拟环境')

    # ---- 模型与轮数 ----
    p.add_argument('--model', type=str, default='googlenet', choices=sorted(MODEL_FACTORIES.keys()),
                   help='要优化调度的 PyTorch 模型')
    p.add_argument('--episodes', type=int, default=500,
                   help='PPO 训练轮数 [可调] 建议 200-2000')
    p.add_argument('--bc-episodes', type=int, default=30,
                   help='BC 预训练轮数 [可调] 0=跳过, 建议 20-50')
    p.add_argument('--real-ft-episodes', type=int, default=0,
                   help='真实延迟微调轮数（仅 surrogate 模式）')

    # ---- PPO 超参数 ----
    p.add_argument('--batch-episodes', type=int, default=8,
                   help='★ 每次 PPO 更新前收集的 episode 数 [可调] '
                        '越大梯度越稳, 建议 4-16')
    p.add_argument('--mini-batch-size', type=int, default=256,
                   help='★ PPO mini-batch 大小 [可调] 建议 128-512')
    p.add_argument('--lr', type=float, default=3e-4,
                   help='学习率 [可调] 建议 1e-4 ~ 1e-3')
    p.add_argument('--ppo-epochs', type=int, default=4,
                   help='每批数据重复训练次数 [可调] 建议 2-8')
    p.add_argument('--clip-eps', type=float, default=0.2,
                   help='PPO 裁剪范围 [可调] 建议 0.1-0.3')
    p.add_argument('--gamma', type=float, default=0.99,
                   help='折扣因子 [可调] real 模式自动用 1.0')
    p.add_argument('--gae-lambda', type=float, default=1.0,
                   help='GAE λ [可调] terminal-only reward 下建议 1.0')
                   #gae-lambda原先是0.95
    p.add_argument('--entropy-coef', type=float, default=0.02,
                   help='熵系数起点 [可调] 越大探索越多, 建议 0.01-0.05')
    p.add_argument('--entropy-coef-end', type=float, default=None,
                   help='熵系数终点（可选）: 若设置，训练中线性衰减到该值')

    # ---- 网络结构 ----
    p.add_argument('--hidden', type=int, default=128,
                   help='隐藏层维度 [可调] 建议 64/128/256')
    p.add_argument('--emb', type=int, default=128,
                   help='嵌入维度 [可调] 建议与 hidden 一致')
    p.add_argument('--heads', type=int, default=4,
                   help='注意力头数 [可调] 建议 2/4/8')

    # ---- 环境与评估 ----
    p.add_argument('--streams', type=int, default=8,
                   help='CUDA 流数量 [可调] 建议 4/8/16')
    p.add_argument('--iters', type=int, default=20,
                   help='每次评估的 benchmark 次数 [可调]')
    p.add_argument('--warmups', type=int, default=5,
                   help='benchmark 预热次数 [可调]')
    p.add_argument('--autosave-interval', type=int, default=1,
                   help='real训练每多少次PPO update保存一次 *_latest.pt；1=每次更新都保存')

    # ---- 其他 ----
    p.add_argument('--seed', type=int, default=0, help='随机种子')
    p.add_argument('--save', type=str, default=None,
                   help='模型保存路径（默认自动生成）')
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.mode == 'real' and device != 'cuda':
        print("ERROR: --mode real requires a CUDA GPU")
        sys.exit(1)

    if args.save is None:
        args.save = os.path.join('artifacts', f'{args.model}_{args.mode}.pt')
    args.save = _project_path(args.save)

    def factory():
        return MODEL_FACTORIES[args.model](device)

    # 构建训练配置
    cfg = TrainConfig(
        episodes=args.episodes,
        batch_episodes=args.batch_episodes,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        entropy_coef=args.entropy_coef,
        entropy_coef_end=args.entropy_coef_end,
        hidden_dim=args.hidden,
        emb_dim=args.emb,
        n_heads=args.heads,
        n_streams=args.streams,
        bc_episodes=args.bc_episodes,
        bc_lr=args.lr * 3,
        real_finetune_episodes=args.real_ft_episodes,
        bench_iters=args.iters,
        bench_warmups=args.warmups,
        autosave_interval=args.autosave_interval,
    )

    print(f"Mode: {args.mode.upper()}")
    print(f"Model: {args.model}")
    print(f"Episodes: {args.episodes} (BC: {args.bc_episodes})")
    print(f"PPO batch: {args.batch_episodes} eps/update, "
          f"mini_batch={args.mini_batch_size}")
    print(f"Network: hidden={args.hidden} emb={args.emb} heads={args.heads}")
    print(f"Save to: {args.save}")
    print()

    # 选择训练函数
    train_fn = train_policy_real if args.mode == 'real' else train_policy
    train_fn(
        model_factory=factory,
        cfg=cfg,
        device=torch.device(device),
        seed=args.seed,
        save_path=args.save,
    )


if __name__ == '__main__':
    main()
