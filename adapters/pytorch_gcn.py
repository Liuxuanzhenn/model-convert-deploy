"""PyTorch GCN 适配器"""
import os
import logging
from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from .base import ModelAdapter
from .registry import register

logger = logging.getLogger(__name__)


class SimpleGCN(nn.Module):
    """简化版GCN结构，便于state_dict重建"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.input_dim = in_dim
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, node_features, edge_index=None):
        x = torch.relu(self.lin1(node_features))
        return self.lin2(x)


def _infer_gcn_config(state_dict: dict) -> dict:
    """从state_dict推断GCN配置"""
    tensors = [v for v in state_dict.values() if isinstance(v, torch.Tensor) and v.ndim == 2]
    if len(tensors) < 2:
        return {"in_dim": 64, "hidden_dim": 64, "out_dim": 1}
    
    in_dim = int(tensors[0].shape[1])
    hidden_dim = int(tensors[0].shape[0])
    out_dim = int(tensors[-1].shape[0]) if len(tensors) > 1 else 1
    return {"in_dim": in_dim, "hidden_dim": hidden_dim, "out_dim": out_dim}


def _build_gcn_from_state_dict(state_dict: dict) -> Optional[nn.Module]:
    """从state_dict重建GCN模型"""
    config = _infer_gcn_config(state_dict)
    model = SimpleGCN(**config)
    try:
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Rebuilt SimpleGCN with config={config}")
        return model
    except Exception as e:
        logger.warning(f"Could not fully load GCN state_dict: {e}")
        return model


@register("pytorch", "gcn")
class PytorchGCNAdapter(ModelAdapter):
    """GCN适配器 - 图神经网络模型"""

    def load(self) -> None:
        """加载GCN模型"""
        weight = self._find_weight()
        if not weight:
            self.model = None
            return

        try:
            try:
                obj = torch.load(weight, map_location="cpu", weights_only=False)
            except TypeError:
                obj = torch.load(weight, map_location="cpu")

            if hasattr(obj, "forward"):
                self.model = obj
                return

            if isinstance(obj, dict):
                candidates = []
                if all(isinstance(v, torch.Tensor) for v in obj.values()):
                    candidates.append(obj)

                for key in ("state_dict", "model_state_dict", "model"):
                    val = obj.get(key)
                    if val is None:
                        continue
                    if hasattr(val, "forward"):
                        self.model = val
                        logger.info(f"Loaded GCN model from '{key}' in {weight}")
                        return
                    if isinstance(val, dict):
                        candidates.append(val)

                for sd in candidates:
                    self.model = _build_gcn_from_state_dict(sd)
                    if self.model is not None:
                        break

                if self.model is None:
                    logger.warning(f"Failed to build GCN from dict in {weight}; keys={list(obj.keys())[:10]}")
            else:
                logger.warning(f"Unsupported object type from {weight}: {type(obj)}")
                self.model = None
        except Exception as e:
            logger.error(f"GCN model loading failed: {e}", exc_info=True)
            self.model = None

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出GCN模型"""
        out = []
        if self.model is None or not hasattr(self.model, "forward"):
            return out

        try:
            import torch
            fmts = [str(x).lower() for x in formats]
            input_dim = getattr(self.model, "input_dim", 64)
            num_nodes = 128
            num_edges = 256
            node_features = torch.randn(num_nodes, input_dim)
            edge_index = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long)

            if "pt" in fmts or "pytorch" in fmts:
                try:
                    path = os.path.join(self.artifacts_dir, "gcn.pt")
                    torch.save(self.model.state_dict() if hasattr(self.model, "state_dict") else self.model, path)
                    out.append(path)
                except Exception as e:
                    logger.error(f"PyTorch export failed: {e}")

            if "torchscript" in fmts:
                try:
                    scripted = torch.jit.trace(self.model, (node_features, edge_index))
                    path = os.path.join(self.artifacts_dir, "gcn.torchscript.pt")
                    scripted.save(path)
                    out.append(path)
                except Exception as e:
                    logger.error(f"TorchScript export failed: {e}")

            if "onnx" in fmts:
                try:
                    path = os.path.join(self.artifacts_dir, "gcn.onnx")
                    torch.onnx.export(
                        self.model,
                        (node_features, edge_index),
                        path,
                        input_names=["node_features", "edge_index"],
                        output_names=["output"],
                        opset_version=12,
                        export_params=True,
                        do_constant_folding=True
                    )
                    out.append(path)
                except Exception as e:
                    logger.error(f"ONNX export failed: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"GCN export failed: {e}")

        return out

