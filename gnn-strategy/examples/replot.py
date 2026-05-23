"""从已保存的 checkpoint 文件重新生成训练曲线图。

不需要重新训练，直接读取 checkpoint 中的 history 数据画图。

用法：
  python gnn-strategy/examples/replot.py \
      --checkpoint artifacts/googlenet_real.pt \
      --output artifacts/plots/GoogLeNet

  # 也可以从 _final.pt 画图
  python gnn-strategy/examples/replot.py \
      --checkpoint artifacts/googlenet_real_final.pt
"""

import os
import sys
import argparse

_GNN_STRATEGY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_GNN_STRATEGY_DIR)
for _p in (_GNN_STRATEGY_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from gnn_strategy.plot_training import plot_training_curves


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


def main():
    p = argparse.ArgumentParser(description='Plot training curves from a saved checkpoint')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to .pt checkpoint file (contains history)')
    p.add_argument('--output', type=str, default=None,
                   help='Output directory for plots (default: same dir as checkpoint)')
    p.add_argument('--dpi', type=int, default=300, help='Output DPI')
    args = p.parse_args()
    args.checkpoint = _project_path(args.checkpoint)
    if args.output is not None:
        args.output = _project_path(args.output)

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    history = ckpt.get('history')
    if history is None:
        print("ERROR: checkpoint does not contain 'history' key")
        print(f"  Available keys: {list(ckpt.keys())}")
        sys.exit(1)

    model_name = history.get('model', '')
    if not model_name:
        model_name = os.path.basename(args.checkpoint).split('_')[0]

    output_dir = args.output
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(args.checkpoint), 'plots', model_name)

    print(f"Model: {model_name}")
    print(f"Episodes: {len(history.get('episodes', []))}")
    print(f"Output: {output_dir}")

    saved = plot_training_curves(history, save_dir=output_dir, model_name=model_name, dpi=args.dpi)

    if saved:
        print(f"\nDone! {len(saved)} files saved.")
    else:
        print("\nNo plots generated.")


if __name__ == '__main__':
    main()
