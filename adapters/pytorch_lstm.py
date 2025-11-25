"""PyTorch LSTM 适配器"""
import os
import logging
from typing import Iterable, List

import torch
import torch.nn as nn

from .base import ModelAdapter
from .registry import register

logger = logging.getLogger(__name__)


class ForecastLSTM(nn.Module):
    """复用 run_lstm.py 中的 LSTM 结构"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def _infer_config_from_state_dict(state_dict: dict) -> dict:
    tensors = [(k, v) for k, v in state_dict.items() if isinstance(v, torch.Tensor)]
    if not tensors:
        return {"input_dim": 7, "hidden_dim": 64, "output_dim": 1, "num_layers": 1, "dropout": 0.0}

    ih_weights = [v for k, v in tensors if "weight_ih" in k]
    base = ih_weights[0] if ih_weights else tensors[0][1]
    input_dim = int(base.shape[1]) if base.ndim >= 2 else 7
    hidden_dim = int(base.shape[0] // 4) if base.ndim >= 2 else 64
    layer_indices = {k.split("weight_ih_l")[-1].split("_")[0] for k, _ in tensors if "weight_ih_l" in k}
    num_layers = max(len(layer_indices), 1)
    return {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": 1,
        "num_layers": num_layers,
        "dropout": 0.0,
    }


def _build_lstm_from_state_dict(state_dict: dict):
    """从常见 LSTM state_dict 推断一个简单 LSTM 结构并加载权重。"""
    if not state_dict:
        logger.warning("Empty state_dict")
        return None

    tensors = [(k, v) for k, v in state_dict.items() if isinstance(v, torch.Tensor) and v.ndim >= 2]
    if not tensors:
        logger.warning("No weight tensor found in state_dict")
        return None

    first_weight = tensors[0][1]
    ih_weights = [v for k, v in tensors if "weight_ih" in k]

    base = ih_weights[0] if ih_weights else first_weight
    input_size = int(base.shape[1]) if base.ndim >= 2 else 7
    hidden_size = int(base.shape[0] // 4) if base.ndim >= 2 else 64
    num_layers = max(1, len(ih_weights))

    model = ForecastLSTM(input_size, hidden_size, 1, num_layers, 0.0)
    try:
        model.load_state_dict(state_dict, strict=False)
        logger.info(f"Built LSTM: input={input_size}, hidden={hidden_size}, layers={num_layers}")
    except Exception as e:
        logger.warning(f"Could not fully load state_dict: {e}; using default-initialized weights")
    return model


@register("pytorch", "lstm")
class PytorchLSTMAdapter(ModelAdapter):
    """LSTM适配器 - 时间序列预测模型"""

    def load(self) -> None:
        """加载 LSTM 模型（支持保存完整模型或多种 state_dict 包装形式）。"""
        weight = self._find_weight()
        if not weight:
            logger.warning("No weight file found in model_dir")
            self.model = None
            return

        try:
            try:
                obj = torch.load(weight, map_location="cpu", weights_only=False)
            except TypeError:
                obj = torch.load(weight, map_location="cpu")

            if hasattr(obj, "forward"):
                self.model = obj
                logger.info(f"Loaded LSTM model object from {weight}")
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
                        logger.info(f"Loaded LSTM model from '{key}' in {weight}")
                        return
                    if isinstance(val, dict):
                        candidates.append(val)

                for sd in candidates:
                    self.model = _build_lstm_from_state_dict(sd)
                    if self.model is not None:
                        break

                if self.model is None:
                    logger.warning(f"Failed to build LSTM from dict in {weight}; keys={list(obj.keys())[:10]}")
            else:
                logger.warning(f"Unsupported object type from {weight}: {type(obj)}")
                self.model = None
        except Exception as e:
            logger.error(f"LSTM model loading failed: {e}", exc_info=True)
            self.model = None

    def _get_input_dim(self) -> int:
        """获取LSTM输入维度"""
        if hasattr(self.model, "lstm") and hasattr(self.model.lstm, "input_size"):
            return self.model.lstm.input_size
        return getattr(self.model, "input_size", 7)

    def _get_base_name(self) -> str:
        """从原始文件名生成基础名称"""
        weight_file = self._find_weight()
        if not weight_file:
            return "lstm"
        file_basename = os.path.splitext(os.path.basename(weight_file))[0]
        if file_basename.startswith("model_"):
            return file_basename.replace("model_", "").replace("_quantized", "").replace("_pruned", "").replace("_distilled", "")
        return file_basename

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出LSTM模型"""
        out = []
        if self.model is None or not hasattr(self.model, "forward"):
            return out

        try:
            import torch
            fmts = [str(x).lower() for x in formats]
            input_dim = self._get_input_dim()
            example_input = torch.randn(1, 10, input_dim)

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
            logger.error(f"LSTM export failed: {e}")

        return out

