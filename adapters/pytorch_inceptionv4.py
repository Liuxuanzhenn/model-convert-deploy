"""PyTorch InceptionV4 适配器"""
import logging
from typing import Iterable, List
from .base import ModelAdapter
from .registry import register

logger = logging.getLogger(__name__)


@register("pytorch", "inceptionv4")
class PyTorchInceptionV4Adapter(ModelAdapter):
    """InceptionV4适配器 - 使用基类通用方法"""

    def load(self) -> None:
        """加载InceptionV4模型"""
        weight = self._find_weight()
        if not weight:
            self.model = None
            return

        try:
            import torch
            # 设置 weights_only=False 以支持加载完整模型对象
            try:
                obj = torch.load(weight, map_location='cpu', weights_only=False)
            except TypeError:
                # 兼容旧版本 PyTorch
                obj = torch.load(weight, map_location='cpu')

            # 直接是模型
            if hasattr(obj, "forward"):
                self.model = obj
            # 是state_dict，尝试用torchvision构建
            elif isinstance(obj, dict):
                try:
                    from torchvision.models import inception_v3
                    model = inception_v3(pretrained=False)
                    state_dict = obj.get('state_dict', obj)
                    model.load_state_dict(state_dict, strict=False)
                    model.eval()
                    self.model = model
                except Exception as e:
                    logger.error(f"InceptionV4 model loading failed: {e}")
                    self.model = None
        except Exception as e:
            logger.error(f"Weight file loading failed: {e}")
            self.model = None

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出InceptionV4模型"""
        out = []
        if self.model is None:
            return out

        try:
            import torch
            example_input = torch.randn(1, 3, 299, 299)  # InceptionV4使用299x299

            if "torchscript" in formats:
                path = self._export_torchscript(example_input, "inceptionv4.torchscript.pt")
                if path:
                    out.append(path)

            if "onnx" in formats:
                path = self._export_onnx(example_input, "inceptionv4.onnx")
                if path:
                    out.append(path)
        except Exception as e:
            logger.error(f"InceptionV4 export failed: {e}")

        return out
