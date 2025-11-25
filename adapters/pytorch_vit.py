"""PyTorch ViT 适配器"""
import os
import logging
from typing import Iterable, List
from .base import ModelAdapter
from .registry import register

logger = logging.getLogger(__name__)


@register("pytorch", "vit")
class PyTorchViTAdapter(ModelAdapter):
    """ViT适配器 - 使用基类通用方法"""

    def load(self) -> None:
        """加载ViT模型"""
        weight = self._find_weight()
        if not weight:
            self.model = None
            return

        try:
            import torch
            obj = torch.load(weight, map_location='cpu', weights_only=False)

            if hasattr(obj, "forward"):
                self.model = obj
            elif isinstance(obj, dict):
                try:
                    file_size_mb = os.path.getsize(weight) / (1024 * 1024)

                    try:
                        import timm
                        if file_size_mb > 300:
                            model_name = 'vit_large_patch16_224'
                        elif file_size_mb > 150:
                            model_name = 'vit_base_patch16_224'
                        else:
                            model_name = 'vit_small_patch16_224'
                        model = timm.create_model(model_name, pretrained=False)
                    except Exception:
                        from torchvision.models import vit_b_16
                        model = vit_b_16(pretrained=False)

                    state_dict = obj.get('state_dict', obj)
                    model.load_state_dict(state_dict, strict=False)
                    model.eval()
                    self.model = model
                except Exception as e:
                    logger.error(f"ViT model construction failed: {e}")
                    self.model = None
        except Exception as e:
            logger.error(f"ViT model loading failed: {e}")
            self.model = None

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出ViT模型"""
        out = []
        if self.model is None:
            return out

        try:
            import torch
            fmts = [str(x).lower() for x in formats]
            example_input = torch.randn(1, 3, 224, 224)

            if "pt" in fmts or "pytorch" in fmts:
                try:
                    pt_path = os.path.join(self.artifacts_dir, "vit_base_patch16_224.pt")
                    if hasattr(self.model, "state_dict"):
                        torch.save(self.model.state_dict(), pt_path)
                    else:
                        torch.save(self.model, pt_path)
                    out.append(pt_path)
                except Exception as e:
                    logger.error(f"PyTorch format save failed: {e}")

            if "torchscript" in fmts:
                path = self._export_torchscript(example_input, "vit.torchscript.pt")
                if path:
                    out.append(path)

            if "onnx" in fmts:
                path = self._export_onnx(example_input, "vit_base_patch16_224.onnx", opset=17)
                if path:
                    out.append(path)

        except Exception as e:
            logger.error(f"ViT export failed: {e}")

        return out
