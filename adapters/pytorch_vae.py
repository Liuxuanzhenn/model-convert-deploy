"""PyTorch VAE 适配器"""
import os
import logging
from typing import Iterable, List

import torch
import torch.nn as nn

from .base import ModelAdapter
from .registry import register

logger = logging.getLogger(__name__)


class SimpleVAE(nn.Module):
    """用于 state_dict 推断的简易 VAE 结构（MNIST 尺寸）。"""

    def __init__(self, in_dim: int = 784, hid: int = 400, latent: int = 20):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(), nn.Linear(hid, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, hid), nn.ReLU(), nn.Linear(hid, in_dim), nn.Sigmoid())

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        return self.decoder(z), z[:, : z.size(1) // 2], z[:, z.size(1) // 2 :]


def _build_vae_from_state_dict(state_dict: dict):
    """从常见 VAE state_dict 构建 SimpleVAE。"""
    model = SimpleVAE()
    try:
        model.load_state_dict(state_dict, strict=False)
        logger.info("Built VAE from state_dict")
    except Exception as e:
        logger.warning(f"Could not fully load VAE state_dict: {e}")
    return model


@register("pytorch", "vae")
class PytorchVAEAdapter(ModelAdapter):
    """VAE适配器 - 变分自编码器"""

    def load(self) -> None:
        """加载 VAE 模型（完整模型或常见 state_dict 包装）。"""
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
                        logger.info(f"Loaded VAE model from '{key}' in {weight}")
                        return
                    if isinstance(val, dict):
                        candidates.append(val)

                for sd in candidates:
                    self.model = _build_vae_from_state_dict(sd)
                    if self.model is not None:
                        break

                if self.model is None:
                    logger.warning(f"Failed to build VAE from dict in {weight}; keys={list(obj.keys())[:10]}")
            else:
                logger.warning(f"Unsupported object type from {weight}: {type(obj)}")
                self.model = None
        except Exception as e:
            logger.error(f"VAE model loading failed: {e}", exc_info=True)
            self.model = None

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出VAE模型"""
        out = []
        if self.model is None:
            return out

        fmts = [str(x).lower() for x in formats]

        # VAE模型包含编码器和解码器，需要手动处理ONNX导出
        if "onnx" in fmts:
            try:
                import torch
                if hasattr(self.model, "forward"):
                    example_input = torch.randn(1, 1, 28, 28)
                    
                    # 检测并匹配模型dtype
                    model_dtype = None
                    if hasattr(self.model, 'parameters'):
                        for param in self.model.parameters():
                            if param is not None:
                                model_dtype = param.dtype
                                break
                    
                    if model_dtype == torch.float16:
                        example_input = example_input.half()
                    elif model_dtype is not None and model_dtype != torch.float32:
                        example_input = example_input.to(dtype=model_dtype)
                    
                    # 构造文件名（包含操作后缀）
                    ops = self._operations if self._operations else []
                    filename = "vae.onnx"
                    if ops:
                        filename = f"vae_{'_'.join(ops)}.onnx"
                    
                    path = os.path.join(self.artifacts_dir, filename)
                    self.model.eval()
                    
                    torch.onnx.export(
                        self.model,
                        example_input,
                        path,
                        input_names=["input_image"],
                        output_names=["reconstructed", "mu", "logvar"],
                        opset_version=11,
                        export_params=True,
                        do_constant_folding=True
                    )
                    out.append(path)
                    logger.info(f"Successfully exported VAE to ONNX: {path}")
            except Exception as e:
                logger.error(f"ONNX export failed: {e}", exc_info=True)

        if "torchscript" in fmts:
            try:
                import torch
                if hasattr(self.model, "forward"):
                    example_input = torch.randn(1, 1, 28, 28)

                    try:
                        scripted = torch.jit.trace(self.model, example_input)
                        path = os.path.join(self.artifacts_dir, "vae.torchscript.pt")
                        scripted.save(path)
                        out.append(path)
                    except Exception as e:
                        logger.error(f"TorchScript export failed: {e}")
            except Exception as e:
                logger.error(f"TorchScript export failed: {e}")

        if "pytorch" in fmts or "pt" in fmts:
            try:
                import torch
                path = os.path.join(self.artifacts_dir, "vae.pt")
                if hasattr(self.model, "state_dict"):
                    torch.save(self.model.state_dict(), path)
                else:
                    torch.save(self.model, path)
                out.append(path)
            except Exception as e:
                logger.error(f"PyTorch export failed: {e}")

        return out

