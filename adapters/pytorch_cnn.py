"""PyTorch CNN 适配器"""
import os
import logging
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models

from .base import ModelAdapter
from .registry import register

logger = logging.getLogger(__name__)


class CustomCNN(nn.Module):
    """与 run_cnn.py 中一致的结构"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _infer_num_classes(state_dict: dict) -> int:
    for key in ("fc.6.weight", "fc.4.weight", "classifier.1.weight", "classifier.3.weight"):
        tensor = state_dict.get(key)
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            return tensor.shape[0]
    tensors = [v for v in state_dict.values() if isinstance(v, torch.Tensor) and v.ndim == 2]
    return tensors[-1].shape[0] if tensors else 10


def _rebuild_custom_cnn(state_dict: dict) -> Optional[nn.Module]:
    try:
        model = CustomCNN(_infer_num_classes(state_dict))
        model.load_state_dict(state_dict, strict=False)
        return model
    except Exception as e:
        logger.error(f"Failed to rebuild CustomCNN: {e}")
        return None


@register("pytorch", "cnn")
class PytorchCNNAdapter(ModelAdapter):
    """CNN适配器 - 自定义CNN/AlexNet/SqueezeNet等"""

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
                for builder in (tv_models.alexnet, tv_models.squeezenet1_0, tv_models.squeezenet1_1):
                    try:
                        model = builder(weights=None)
                        model.load_state_dict(obj, strict=False)
                        self.model = model
                        return
                    except Exception:
                        continue

                rebuilt = _rebuild_custom_cnn(obj)
                if rebuilt:
                    self.model = rebuilt
                    return

            logger.warning(f"Unsupported CNN weight format: {weight}")
            self.model = None
        except Exception as e:
            logger.error(f"CNN model loading failed: {e}", exc_info=True)
            self.model = None

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        out: List[str] = []
        if self.model is None:
            return out

        try:
            import torch

            base_name = self._get_base_name()
            fmts = [str(x).lower() for x in formats]
            example_input = torch.randn(1, 3, 224, 224)

            if "pt" in fmts or "pytorch" in fmts:
                path = os.path.join(self.artifacts_dir, f"{base_name}.pt")
                torch.save(self.model.state_dict() if hasattr(self.model, "state_dict") else self.model, path)
                out.append(path)

            if "torchscript" in fmts:
                path = self._export_torchscript(example_input, f"{base_name}.torchscript.pt")
                if path:
                    out.append(path)

            if "onnx" in fmts:
                path = self._export_onnx(example_input, f"{base_name}.onnx")
                if path:
                    out.append(path)

        except Exception as e:
            logger.error(f"CNN export failed: {e}")

        return out

    def _get_base_name(self) -> str:
        weight_file = self._find_weight()
        if not weight_file:
            return "cnn"
        base = os.path.splitext(os.path.basename(weight_file))[0]
        if base.startswith("model_"):
            base = base.replace("model_", "").replace("_quantized", "").replace("_pruned", "").replace("_distilled", "")
        return base or "cnn"

