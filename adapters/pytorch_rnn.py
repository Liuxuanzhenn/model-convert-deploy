"""PyTorch RNN 适配器"""
import os
import logging
from typing import Iterable, List

import torch
import torch.nn as nn

from .base import ModelAdapter
from .registry import register

logger = logging.getLogger(__name__)


class ForecastRNN(nn.Module):
    """还原 run_rnn.py 中的结构，便于 state_dict 重建"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


def _infer_rnn_config(state_dict: dict) -> dict:
    tensors = [(k, v) for k, v in state_dict.items() if isinstance(v, torch.Tensor)]
    if not tensors:
        return {"input_dim": 7, "hidden_dim": 64, "output_dim": 1, "num_layers": 1, "dropout": 0.0}

    ih_weights = [v for k, v in tensors if "weight_ih" in k]
    base = ih_weights[0] if ih_weights else tensors[0][1]
    input_dim = int(base.shape[1]) if base.ndim >= 2 else 7
    hidden_dim = int(base.shape[0]) if base.ndim >= 2 else 64
    output_dim = next((v.shape[0] for k, v in tensors if "fc.weight" in k), 1)
    layer_indices = {k.split("weight_ih_l")[-1].split("_")[0] for k, _ in tensors if "weight_ih_l" in k}
    num_layers = max(len(layer_indices), 1)
    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "num_layers": num_layers,
        "dropout": 0.0,
    }


def _build_rnn_from_state_dict(state_dict: dict):
    cfg = _infer_rnn_config(state_dict)
    model = ForecastRNN(**cfg)
    try:
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Built RNN: {cfg}")
    except Exception as e:
        logger.warning(f"Could not fully load RNN state_dict: {e}")
    return model


@register("pytorch", "rnn")
class PytorchRNNAdapter(ModelAdapter):
    """RNN适配器 - 时间序列预测模型"""

    def load(self) -> None:
        """加载 RNN 模型（支持完整模型或多种 state_dict 包装形式）。"""
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
                        logger.info(f"Loaded RNN model from '{key}' in {weight}")
                        return
                    if isinstance(val, dict):
                        candidates.append(val)

                for sd in candidates:
                    self.model = _build_rnn_from_state_dict(sd)
                    if self.model is not None:
                        break

                if self.model is None:
                    logger.warning(f"Failed to build RNN from dict in {weight}; keys={list(obj.keys())[:10]}")
            else:
                logger.warning(f"Unsupported object type from {weight}: {type(obj)}")
                self.model = None
        except Exception as e:
            logger.error(f"RNN model loading failed: {e}", exc_info=True)
            self.model = None

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出RNN模型"""
        out = []
        if self.model is None or not hasattr(self.model, "forward"):
            return out

        try:
            import torch
            fmts = [str(x).lower() for x in formats]
            example_input = torch.randn(1, 10, self._get_input_dim())

            if "pt" in fmts or "pytorch" in fmts:
                try:
                    path = os.path.join(self.artifacts_dir, f"{self._get_base_name()}.pt")
                    torch.save(self.model.state_dict() if hasattr(self.model, "state_dict") else self.model, path)
                    out.append(path)
                except Exception as e:
                    logger.error(f"PyTorch export failed: {e}")

            if "torchscript" in fmts:
                path = self._export_torchscript(example_input, f"{self._get_base_name()}.torchscript.pt")
                if path:
                    out.append(path)

            if "onnx" in fmts:
                path = self._export_onnx(
                    example_input=example_input,
                    filename=f"{self._get_base_name()}.onnx",
                    input_names=["input_sequence"],
                    output_names=["output"]
                )
                if path:
                    out.append(path)

        except Exception as e:
            logger.error(f"RNN export failed: {e}")

        return out

