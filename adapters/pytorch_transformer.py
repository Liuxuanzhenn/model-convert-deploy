"""PyTorch Transformer 适配器"""

import io
import math
import os
import pickle
import logging
from typing import Iterable, List

import torch
import torch.nn as nn

from .base import ModelAdapter
from .registry import register

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    """Transformer时间序列预测模型（与run_transformer.py结构一致）"""
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1, max_seq_len=1000):
        super().__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_projection(x).transpose(0, 1)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        out = self.output_projection(x[:, -1, :])
        return out.squeeze(-1)


def _get_param_dtype(model: nn.Module):
    if model is None or not hasattr(model, "parameters"):
        return None
    for param in model.parameters():
        if param is not None:
            return param.dtype
    return None


class _TransformerPickleModule:
    """自定义pickle模块，将__main__中的Transformer相关类映射到当前定义"""

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == "TransformerModel":
                return TransformerModel
            if name == "PositionalEncoding":
                return PositionalEncoding
            return super().find_class(module, name)

    @staticmethod
    def load(file_obj, **kwargs):
        return _TransformerPickleModule.Unpickler(file_obj, **kwargs).load()

    @staticmethod
    def loads(data, **kwargs):
        return _TransformerPickleModule.Unpickler(io.BytesIO(data), **kwargs).load()


def _safe_torch_load(weight: str):
    """兼容不同版本Torch保存格式，必要时注入自定义pickle"""
    load_kwargs = {"map_location": "cpu", "weights_only": False}
    for _ in range(2):
        try:
            return torch.load(weight, **load_kwargs)
        except TypeError:
            load_kwargs.pop("weights_only", None)
        except AttributeError as attr_err:
            if "TransformerModel" in str(attr_err):
                return torch.load(weight, pickle_module=_TransformerPickleModule, **load_kwargs)
            raise
    return torch.load(weight, pickle_module=_TransformerPickleModule, **load_kwargs)


def _infer_config_from_state_dict(state_dict: dict) -> dict:
    """从state_dict推断模型配置"""
    if not state_dict:
        return {"input_dim": 14, "d_model": 64, "nhead": 2, "num_encoder_layers": 2, 
                "dim_feedforward": 256, "dropout": 0.2, "max_seq_len": 720}
    
    tensors = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
    
    # 从input_projection推断input_dim和d_model
    if "input_projection.weight" in tensors:
        w = tensors["input_projection.weight"]
        input_dim, d_model = int(w.shape[1]), int(w.shape[0])
    else:
        input_dim, d_model = 14, 64
    
    # 从transformer_encoder层推断层数和配置
    encoder_keys = [k for k in tensors.keys() if "transformer_encoder.layers" in k]
    layer_indices = {int(k.split("layers.")[1].split(".")[0]) for k in encoder_keys if "layers." in k}
    num_encoder_layers = max(len(layer_indices), 2) if layer_indices else 2
    
    dim_feedforward = 256
    nhead = 2
    for key in encoder_keys:
        if "linear1.weight" in key:
            dim_feedforward = int(tensors[key].shape[0])
        if "self_attn.in_proj_weight" in key:
            inferred_d_model = int(tensors[key].shape[1])
            if inferred_d_model == d_model:
                nhead = next((n for n in [2, 4, 8] if d_model % n == 0), 2)
    
    return {"input_dim": input_dim, "d_model": d_model, "nhead": nhead,
            "num_encoder_layers": num_encoder_layers, "dim_feedforward": dim_feedforward,
            "dropout": 0.2, "max_seq_len": 720}


def _rebuild_transformer(state_dict: dict):
    """从state_dict重建Transformer模型"""
    if not state_dict:
        return None
    try:
        config = _infer_config_from_state_dict(state_dict)
        model = TransformerModel(**config)
        model.load_state_dict(state_dict, strict=False)
        return model
    except Exception:
        return None


@register("pytorch", "transformer")
class PytorchTransformerAdapter(ModelAdapter):
    """Transformer适配器 - 时间序列预测模型"""

    def load(self) -> None:
        """加载Transformer模型（支持完整模型对象或从state_dict重建）"""
        weight = self._find_weight()
        if not weight:
            self.model = None
            return

        try:
            obj = _safe_torch_load(weight)

            if hasattr(obj, "forward"):
                self.model = obj
                return

            if isinstance(obj, dict):
                for key in ("model", "transformer_model", "transformer"):
                    val = obj.get(key)
                    if val and hasattr(val, "forward"):
                        self.model = val
                        return

                candidates = []
                for key in ("state_dict", "model_state_dict"):
                    val = obj.get(key)
                    if isinstance(val, dict) and all(isinstance(v, torch.Tensor) for v in val.values()):
                        candidates.append(val)
                
                if not candidates and all(isinstance(v, torch.Tensor) for v in obj.values()):
                    candidates.append(obj)

                for sd in candidates:
                    self.model = _rebuild_transformer(sd)
                    if self.model is not None:
                        break
        except Exception as e:
            logger.error(f"Transformer loading failed: {e}")
            self.model = None

    def _get_input_dim(self) -> int:
        """获取Transformer输入维度"""
        if hasattr(self.model, "input_projection") and hasattr(self.model.input_projection, "in_features"):
            return self.model.input_projection.in_features
        return 14

    def _get_base_name(self) -> str:
        weight = self._find_weight()
        if not weight:
            return "transformer"
        base = os.path.splitext(os.path.basename(weight))[0]
        if base.startswith("model_"):
            base = base.replace("model_", "", 1)
        return base or "transformer"

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出Transformer模型"""
        out = []
        if self.model is None or not hasattr(self.model, "forward"):
            return out

        try:
            import torch
            fmts = [str(x).lower() for x in formats]
            
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            input_dim = self._get_input_dim()
            example_input = torch.randn(1, 128, input_dim)
            model_dtype = _get_param_dtype(self.model)
            if model_dtype is not None and model_dtype != torch.float32:
                example_input = example_input.to(dtype=model_dtype)
                try:
                    self.model.to(dtype=model_dtype)
                except Exception:
                    pass

            base_name = self._get_base_name()

            if "pt" in fmts or "pth" in fmts:
                path = os.path.join(self.artifacts_dir, f"{base_name}.pt")
                torch.save(self.model, path, _use_new_zipfile_serialization=False)
                out.append(path)

            if "torchscript" in fmts:
                path = self._export_torchscript(example_input, f"{base_name}.torchscript.pt")
                if path:
                    out.append(path)

            if "onnx" in fmts:
                try:
                    onnx_path = os.path.join(self.artifacts_dir, f"{base_name}.onnx")
                    torch.onnx.export(
                        self.model,
                        example_input,
                        onnx_path,
                        export_params=True,
                        opset_version=14,
                        do_constant_folding=True,
                        input_names=["input"],
                        output_names=["output"],
                        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
                    )
                    if os.path.exists(onnx_path):
                        out.append(onnx_path)
                        logger.info(f"ONNX export successful: {onnx_path}")
                except Exception as onnx_err:
                    logger.error(f"ONNX export failed: {onnx_err}", exc_info=True)
        except Exception as e:
            logger.error(f"Transformer export failed: {e}")

        return out
