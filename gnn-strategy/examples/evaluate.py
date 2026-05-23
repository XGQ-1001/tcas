"""Evaluate a trained dynamic GNN policy against Opara / TCAS baselines.

Usage:
  python gnn-strategy/examples/evaluate.py \\
      --models googlenet,inception_v3,resnet50,deepfm,bert_base \\
      --policy artifacts/policy_dynamic.pt \\
      --out artifacts/eval_results_dynamic.csv \\
      --plot --overwrite
"""

import os
import sys
import argparse
import csv
import warnings
from typing import Callable, Dict, List, Tuple

_GNN_STRATEGY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_GNN_STRATEGY_DIR)
_TCAS_EXAMPLES_DIR = os.path.join(_REPO_ROOT, 'examples')
for _p in (_GNN_STRATEGY_DIR, _REPO_ROOT, _TCAS_EXAMPLES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torchvision

from Opara import GraphCapturer
from Opara import OperatorLauncher

from gnn_strategy.capturer import benchmark_runner, capturer_gnn_from_fx
from gnn_strategy.env import SchedulingEnv
from gnn_strategy.graph_state import build_graph_state, D_STATIC
from gnn_strategy.policy import DynamicActorCritic
from gnn_strategy.plot_eval import plot_latency_and_speedup
from gnn_strategy.utils import extract_first_fx_graph

warnings.filterwarnings('ignore', message=r'Trying to prepend a node to itself\..*', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning, module=r'torchvision\..*')


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


# ------------------------------------------------------------------
# Model factories
# ------------------------------------------------------------------

def make_googlenet(device: str):
    x = torch.randint(low=0, high=256, size=(1, 3, 224, 224), dtype=torch.float32, device=device)
    return torchvision.models.googlenet().to(device=device).eval(), (x,)


def make_inception_v3(device: str):
    x = torch.randint(low=0, high=256, size=(1, 3, 299, 299), dtype=torch.float32, device=device)
    return torchvision.models.inception_v3(aux_logits=False).to(device=device).eval(), (x,)


def make_resnet50(device: str):
    x = torch.randint(low=0, high=256, size=(1, 3, 224, 224), dtype=torch.float32, device=device)
    return torchvision.models.resnet50().to(device=device).eval(), (x,)


def make_resnet152(device: str):
    x = torch.randint(low=0, high=256, size=(1, 3, 224, 224), dtype=torch.float32, device=device)
    return torchvision.models.resnet152().to(device=device).eval(), (x,)


def make_deepfm(device: str):
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
    return BertLastHiddenState(local_path).to(device=device).eval(), (input_ids, attention_mask)


MODEL_FACTORIES = {
    'googlenet': make_googlenet,
    'inception_v3': make_inception_v3,
    'resnet50': make_resnet50,
    'resnet152': make_resnet152,
    'deepfm': make_deepfm,
    'bert_base': make_bert_base,
}


# ------------------------------------------------------------------
# Policy loading & greedy inference
# ------------------------------------------------------------------

def load_policy(path: str, device: torch.device) -> DynamicActorCritic:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    saved_cfg = ckpt.get('config', None) if isinstance(ckpt, dict) else None

    hidden = 256
    emb = 256
    n_heads = 8
    if saved_cfg is not None:
        hidden = getattr(saved_cfg, 'hidden_dim', 256)
        emb = getattr(saved_cfg, 'emb_dim', 256)
        n_heads = getattr(saved_cfg, 'n_heads', 8)

    policy = DynamicActorCritic(
        static_in_dim=D_STATIC, hidden_dim=hidden, emb_dim=emb, n_heads=n_heads, dropout=0.0,
    ).to(device)
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()
    return policy


def greedy_schedule_order(
    policy: DynamicActorCritic,
    gs,
    device: torch.device,
    n_streams: int = 8,
) -> List[str]:
    """Run greedy inference: at each step pick the highest-scoring ready node."""
    env = SchedulingEnv(gs, n_streams=n_streams, device=device)
    env.reset()

    x = gs.x.to(device)
    with torch.no_grad():
        h_static = policy.encode_static(x, parents=gs.parents, children=gs.children)

    while not env.is_done():
        dyn_node = env.dynamic_node_features().to(device)
        glob = env.global_features().to(device)
        ready_mask = env.ready_mask().to(device)

        with torch.no_grad():
            dist, _ = policy.act(h_static, dyn_node, glob, ready_mask)

        action = int(dist.probs.argmax().item())
        env.step(action)

    order_ids = env.scheduled_order()
    return [gs.node_names[i] for i in order_ids if gs.movable_mask[i].item() == 1.0]


# ------------------------------------------------------------------
# Build FX + profiles
# ------------------------------------------------------------------

def build_fx_and_profiles(model, inputs):
    fx_module = extract_first_fx_graph(model, inputs)
    fx_module.cuda()
    model_class_name = model.__class__.__name__
    node_profiles, device_props = OperatorLauncher.recompile(
        model_class_name, fx_module, inputs, apply_opara_schedule=False,
    )
    return fx_module, node_profiles, device_props


# ------------------------------------------------------------------
# Evaluate one model
# ------------------------------------------------------------------

def eval_one_model(
    model_name: str,
    model,
    inputs: Tuple[torch.Tensor, ...],
    policy_path: str,
    trials: int,
    iterations: int,
    warmups: int,
    n_streams: int,
    out_csv: str,
):
    device = torch.device('cuda')
    policy = load_policy(policy_path, device=device)

    # Baselines
    runner_opara = GraphCapturer.capturer(inputs, model, use_tcas=False)
    runner_tcas = GraphCapturer.capturer(inputs, model, use_tcas=True)

    # GNN policy runner
    fx_module, node_profiles, device_props = build_fx_and_profiles(model, inputs)
    gs = build_graph_state(fx_module.graph, node_profiles=node_profiles, device_props=device_props)
    order_names = greedy_schedule_order(policy, gs=gs, device=device, n_streams=n_streams)

    runner_gnn = capturer_gnn_from_fx(
        fx_module=fx_module, inputs=inputs, schedule_order=order_names, copy_outputs=False,
    )

    algos: List[Tuple[str, Callable]] = [
        ('Opara', runner_opara),
        ('TCAS', runner_tcas),
        ('GNN-Dynamic(Greedy)', runner_gnn),
    ]

    rows = []
    for algo_name, runner in algos:
        for t in range(trials):
            br = benchmark_runner(runner, inputs=inputs, iterations=iterations, warmups=warmups)
            meta = getattr(runner, '_opara_meta', {}) if hasattr(runner, '_opara_meta') else {}
            rows.append({
                'model': model_name,
                'algo': algo_name,
                'trial': int(t),
                'mean_ms': float(br.mean_ms),
                'std_ms': float(br.std_ms),
                'num_nodes': int(meta.get('num_nodes', -1)) if isinstance(meta, dict) else -1,
                'num_streams': int(meta.get('num_streams', -1)) if isinstance(meta, dict) else -1,
            })

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    write_header = not os.path.exists(out_csv)
    with open(out_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)

    def _mean_of(algo: str) -> float:
        vals = [r['mean_ms'] for r in rows if r['algo'] == algo]
        return float(np.mean(vals)) if vals else float('nan')

    m_opara = _mean_of('Opara')
    m_tcas = _mean_of('TCAS')
    m_gnn = _mean_of('GNN-Dynamic(Greedy)')

    print(f"\n[{model_name}] Opara={m_opara:.4f}ms  TCAS={m_tcas:.4f}ms  GNN-Dynamic={m_gnn:.4f}ms")
    if m_opara > 1e-9:
        print(f"  speedup vs Opara: {(m_opara - m_gnn)/m_opara * 100:.2f}%")
    if m_tcas > 1e-9:
        print(f"  speedup vs TCAS:  {(m_tcas - m_gnn)/m_tcas * 100:.2f}%")


def main():
    p = argparse.ArgumentParser(description='Evaluate dynamic GNN policy')
    p.add_argument('--models', type=str, default='googlenet,inception_v3,resnet50,deepfm,bert_base')
    p.add_argument('--policy', type=str, default='artifacts/policy_dynamic.pt')
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--iterations', type=int, default=500)
    p.add_argument('--warmups', type=int, default=20)
    p.add_argument('--streams', type=int, default=8)
    p.add_argument('--out', type=str, default='artifacts/eval_results_dynamic.csv')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--latency-fig', type=str, default='')
    p.add_argument('--speedup-fig', type=str, default='')
    args = p.parse_args()
    args.policy = _project_path(args.policy)
    args.out = _project_path(args.out)
    if args.latency_fig:
        args.latency_fig = _project_path(args.latency_fig)
    if args.speedup_fig:
        args.speedup_fig = _project_path(args.speedup_fig)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = [m.strip() for m in args.models.split(',') if m.strip()]
    for m in models:
        if m not in MODEL_FACTORIES:
            raise SystemExit(f"Unknown model: {m}. Choices: {sorted(MODEL_FACTORIES.keys())}")

    if args.overwrite and os.path.exists(args.out):
        os.remove(args.out)

    for m in models:
        model, inputs = MODEL_FACTORIES[m](device)
        eval_one_model(
            model_name=m, model=model, inputs=inputs,
            policy_path=args.policy, trials=args.trials,
            iterations=args.iterations, warmups=args.warmups,
            n_streams=args.streams, out_csv=args.out,
        )

    if args.plot:
        out_base, _ = os.path.splitext(args.out)
        latency_fig = args.latency_fig or (out_base + '_latency.png')
        speedup_fig = args.speedup_fig or (out_base + '_speedup.png')
        algo_order = ['Opara', 'TCAS', 'GNN-Dynamic(Greedy)']
        plot_latency_and_speedup(
            csv_path=args.out,
            out_latency_png=latency_fig,
            out_speedup_png=speedup_fig,
            model_order=models,
            algo_order=algo_order,
            baseline_algo='Opara',
        )
        print(f"\nSaved latency fig: {latency_fig}")
        print(f"Saved speedup fig: {speedup_fig}")


if __name__ == '__main__':
    main()
