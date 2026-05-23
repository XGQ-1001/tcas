"""Example 脚本共用的工具：模型工厂 + FX/GraphState 构建 + 路径解析。"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
import torchvision

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
_MT_ROOT = os.path.dirname(_EXAMPLES_DIR)
_GNN_STRATEGY_DIR = os.path.normpath(
    os.path.join(_MT_ROOT, '..', 'gnn-strategy')
)
_REPO_ROOT = os.path.dirname(_GNN_STRATEGY_DIR)
_TCAS_EXAMPLES_DIR = os.path.join(_REPO_ROOT, 'examples')

for _p in (_MT_ROOT, _GNN_STRATEGY_DIR, _REPO_ROOT, _TCAS_EXAMPLES_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from Opara import OperatorLauncher
from gnn_strategy.graph_state import build_graph_state
from gnn_strategy.utils import extract_first_fx_graph


def project_path(path: str) -> str:
    """相对路径默认挂到 multi-task-gnn/artifacts 下。"""
    if os.path.isabs(path):
        return path
    norm = os.path.normpath(path)
    if norm == 'artifacts' or norm.startswith(f'artifacts{os.sep}'):
        return os.path.join(_MT_ROOT, norm)
    return os.path.abspath(path)


# --------------------------------------------------------------------------
# 模型工厂 (与 gnn-strategy/examples 保持一致)
# --------------------------------------------------------------------------

def make_googlenet(device: str):
    x = torch.randint(low=0, high=256, size=(1, 3, 224, 224),
                      dtype=torch.float32, device=device)
    return torchvision.models.googlenet().to(device=device).eval(), (x,)


def make_inception_v3(device: str):
    x = torch.randint(low=0, high=256, size=(1, 3, 299, 299),
                      dtype=torch.float32, device=device)
    return torchvision.models.inception_v3(aux_logits=False).to(
        device=device).eval(), (x,)


def make_resnet50(device: str):
    x = torch.randint(low=0, high=256, size=(1, 3, 224, 224),
                      dtype=torch.float32, device=device)
    return torchvision.models.resnet50().to(device=device).eval(), (x,)


def make_resnet152(device: str):
    x = torch.randint(low=0, high=256, size=(1, 3, 224, 224),
                      dtype=torch.float32, device=device)
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


def build_base_graph_state(model_name: str, device: str):
    """给定模型名，返回 (base_gs, fx_module, inputs, model_class_name)."""
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choices: {sorted(MODEL_FACTORIES.keys())}")
    model, inputs = MODEL_FACTORIES[model_name](device)
    fx_module = extract_first_fx_graph(model, inputs)
    fx_module.cuda()
    model_class_name = model.__class__.__name__
    node_profiles, device_props = OperatorLauncher.recompile(
        model_class_name, fx_module, inputs, apply_opara_schedule=False,
    )
    base_gs = build_graph_state(
        fx_module.graph, node_profiles=node_profiles, device_props=device_props,
    )
    return base_gs, fx_module, inputs, model_class_name
