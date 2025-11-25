"""PyTorch VGG 适配器"""
import logging
from typing import Iterable, List
from .base import ModelAdapter
from .registry import register

logger = logging.getLogger(__name__)


@register("pytorch", "vgg")
class PyTorchVGGAdapter(ModelAdapter):
    """VGG适配器 - 使用基类通用方法"""

    def load(self) -> None:
        """加载VGG模型"""
        weight = self._find_weight()
        if not weight:
            self.model = None
            return

        try:
            import torch
            try:
                obj = torch.load(weight, map_location='cpu', weights_only=False)
            except TypeError:
                obj = torch.load(weight, map_location='cpu')

            # 直接是模型
            if hasattr(obj, "forward"):
                self.model = obj
            # 是state_dict，尝试用torchvision构建
            elif isinstance(obj, dict):
                try:
                    from torchvision.models import vgg16, vgg19
                    # 根据文件大小判断VGG16还是VGG19
                    import os
                    file_size_mb = os.path.getsize(weight) / (1024 * 1024)
                    model = vgg19(pretrained=False) if file_size_mb > 550 else vgg16(pretrained=False)

                    state_dict = obj.get('state_dict', obj)
                    model.load_state_dict(state_dict, strict=False)
                    model.eval()
                    self.model = model
                except:
                    self.model = None
        except:
            self.model = None

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出VGG模型"""
        out = []
        if self.model is None:
            return out

        try:
            import torch
            example_input = torch.randn(1, 3, 224, 224)

            if "torchscript" in formats:
                path = self._export_torchscript(example_input, "vgg.torchscript.pt")
                if path:
                    out.append(path)

            if "onnx" in formats:
                path = self._export_onnx(example_input, "vgg.onnx")
                if path:
                    out.append(path)
        except Exception as e:
            logger.error(f"VGG export failed: {e}")

        return out
