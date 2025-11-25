"""PyTorch ResNet 适配器"""
import os
import logging
from typing import Iterable, List
from .base import ModelAdapter
from .registry import register

logger = logging.getLogger(__name__)


@register("pytorch", "resnet")
class PytorchResNetAdapter(ModelAdapter):
    """ResNet适配器 - 使用基类通用方法"""

    def load(self) -> None:
        """加载ResNet模型"""
        weight = self._find_weight()
        if not weight:
            self.model = None
            return

        try:
            import torch
            from collections import OrderedDict
            # 设置 weights_only=False 以支持加载完整模型对象
            try:
                obj = torch.load(weight, map_location="cpu", weights_only=False)
            except TypeError:
                # 兼容旧版本 PyTorch（没有 weights_only 参数）
                obj = torch.load(weight, map_location="cpu")

            # 直接是模型
            if hasattr(obj, "forward"):
                self.model = obj
            # 是state_dict，尝试用torchvision构建
            elif isinstance(obj, (dict, OrderedDict)):
                try:
                    import torchvision.models as models
                    for name in ["resnet50", "resnet18", "resnet34", "resnet101"]:
                        try:
                            model = getattr(models, name, lambda **k: None)(weights=None)
                            if model:
                                model.load_state_dict(obj, strict=False)
                                self.model = model
                                break
                        except:
                            continue
                except:
                    self.model = None
        except:
            self.model = None

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出ResNet模型"""
        out = []
        if self.model is None:
            return out

        try:
            import torch
            fmts = [str(f).lower() for f in formats]
            
            if "pt" in fmts or "pth" in fmts:
                if os.path.exists(self.artifacts_dir):
                    for name in os.listdir(self.artifacts_dir):
                        if name.endswith((".pt", ".pth")) and "model_" in name:
                            out.append(os.path.join(self.artifacts_dir, name))
                            break
                if not out:
                    model_path = os.path.join(self.artifacts_dir, "model.pt")
                    if hasattr(self.model, 'state_dict'):
                        torch.save(self.model.state_dict(), model_path, _use_new_zipfile_serialization=False)
                    else:
                        torch.save(self.model, model_path, _use_new_zipfile_serialization=False)
                    if os.path.exists(model_path):
                        out.append(model_path)

            example_input = torch.randn(1, 3, 224, 224)

            if "torchscript" in fmts:
                path = self._export_torchscript(example_input, "resnet.torchscript.pt")
                if path:
                    out.append(path)

            if "onnx" in fmts:
                path = self._export_onnx(example_input, "resnet.onnx")
                if path:
                    out.append(path)
        except Exception as e:
            logger.error(f"ResNet export failed: {e}")

        return out

