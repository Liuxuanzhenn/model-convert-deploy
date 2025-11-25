"""PyTorch VAN 适配器"""
import os
import logging
import math
from typing import Iterable, List

import torch
import torch.nn as nn

from .base import ModelAdapter
from .registry import register

logger = logging.getLogger(__name__)

van_settings = {
    "S": ([2, 2, 4, 2], [64, 128, 320, 512]),
    "B": ([3, 3, 12, 3], [64, 128, 320, 512]),
    "L": ([3, 5, 27, 3], [64, 128, 320, 512]),
}


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x):
        return self.dwconv(x)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None):
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Conv2d(dim, hidden_dim, 1)
        self.dwconv = DWConv(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_dim, out_dim, 1)

    def forward(self, x):
        return self.fc2(self.act(self.dwconv(self.fc1(x))))


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, 1, 9, 3, dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        attn = self.conv1(self.conv_spatial(self.conv0(x)))
        return x * attn


class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(dim)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        shortcut = x
        x = self.proj_2(self.spatial_gating_unit(self.activation(self.proj_1(x))))
        return x + shortcut


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop_path_rate=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_ch=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride, patch_size // 2)
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class VisionAttentionNetwork(nn.Module):
    def __init__(self, variant: str, num_classes: int):
        super().__init__()
        depths, embed_dims = van_settings[variant]
        mlp_ratios = [8, 8, 4, 4]
        drop_rates = torch.linspace(0, 0.0, sum(depths)).tolist()
        cur = 0

        self.patch_embeds = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(4):
            patch_embed = OverlapPatchEmbed(
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_ch=3 if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
            )
            block = nn.Sequential(*[
                Block(embed_dims[i], mlp_ratios[i], drop_rates[cur + j]) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]

            self.patch_embeds.append(patch_embed)
            self.blocks.append(block)
            self.norms.append(norm)

        self.head = nn.Linear(embed_dims[-1], num_classes)

    def forward(self, x):
        B = x.shape[0]
        for i in range(4):
            x = self.patch_embeds[i](x)
            x = self.blocks[i](x)
            x = x.flatten(2).transpose(1, 2)
            x = self.norms[i](x)
            if i != 3:
                H = W = int(math.sqrt(x.shape[1]))
                x = x.permute(0, 2, 1).contiguous().view(B, -1, H, W)
        return self.head(x.mean(dim=1))


def _infer_variant(state_dict: dict) -> str:
    stage_depths = [0, 0, 0, 0]
    for key in state_dict.keys():
        if key.startswith("block"):
            stage = int(key[5]) - 1
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                stage_depths[stage] = max(stage_depths[stage], int(parts[1]) + 1)
    for name, (depths, _) in van_settings.items():
        if stage_depths == depths:
            return name
    return "S"


def _build_van_from_state_dict(state_dict: dict) -> VisionAttentionNetwork:
    num_classes = next((v.shape[0] for k, v in state_dict.items()
                        if k.endswith("head.weight") and isinstance(v, torch.Tensor)), 1000)
    variant = _infer_variant(state_dict)
    model = VisionAttentionNetwork(variant, num_classes)
    model.load_state_dict(state_dict, strict=False)
    logger.info(f"Rebuilt VAN variant={variant}, num_classes={num_classes}")
    return model


@register("pytorch", "van")
class PytorchVANAdapter(ModelAdapter):
    """VAN适配器 - Vision Attention Network"""

    def load(self) -> None:
        weight = self._find_weight()
        if not weight:
            self.model = None
            return

        try:
            try:
                obj = torch.load(weight, map_location="cpu", weights_only=False)
            except (TypeError, AttributeError, RuntimeError):
                obj = torch.load(weight, map_location="cpu", weights_only=True)

            if hasattr(obj, "forward"):
                self.model = obj
                return

            if isinstance(obj, dict):
                self.model = _build_van_from_state_dict(obj)
                return

            logger.warning(f"Unsupported VAN weight format: {type(obj)}")
            self.model = None
        except Exception as e:
            logger.error(f"VAN model loading failed: {e}", exc_info=True)
            self.model = None

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        out: List[str] = []
        if self.model is None:
            return out

        try:
            import torch

            fmts = [str(x).lower() for x in formats]
            example_input = torch.randn(1, 3, 224, 224)

            if "pt" in fmts or "pytorch" in fmts:
                path = os.path.join(self.artifacts_dir, "van.pt")
                torch.save(self.model.state_dict() if hasattr(self.model, "state_dict") else self.model, path)
                out.append(path)

            if "torchscript" in fmts:
                path = self._export_torchscript(example_input, "van.torchscript.pt")
                if path:
                    out.append(path)

            if "onnx" in fmts:
                path = self._export_onnx(example_input, "van.onnx", opset=13, input_names=["input"], output_names=["logits"])
                if path:
                    out.append(path)

        except Exception as e:
            logger.error(f"VAN export failed: {e}")

        return out

