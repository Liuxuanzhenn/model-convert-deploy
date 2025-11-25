"""PyTorch 通用适配器（精简版）"""

import os
from typing import Iterable, List

import torch

from .base import ModelAdapter
from .registry import register


@register("pytorch", "generic")
@register("pytorch", "other")
class PytorchGenericAdapter(ModelAdapter):
    """PyTorch通用适配器 - 尽量加载模型对象，无法解析时回退为原始对象。"""

    def load(self) -> None:
        """加载通用PyTorch模型，兼容 pt/pth/safetensors 及常见字典包装。"""
        weight = self._find_weight()
        if not weight:
            self.model = None
            return

        try:
            # safetensors：通常为纯 state_dict
            if weight.lower().endswith(".safetensors"):
                try:
                    from safetensors.torch import load_file
                    self.model = load_file(weight)
                    return
                except Exception:
                    self.model = None
                    return

            try:
                obj = torch.load(weight, map_location="cpu", weights_only=False)
            except TypeError:
                obj = torch.load(weight, map_location="cpu")

            self.model = obj

            detected_family = None
            if isinstance(obj, dict):
                # YOLO 等模型在 train_args 里带有 model 名称
                train_args = obj.get("train_args")
                if train_args is not None:
                    try:
                        model_name = (
                            train_args.get("model")
                            if isinstance(train_args, dict)
                            else getattr(train_args, "model", "")
                        )
                        if isinstance(model_name, str) and "yolo" in model_name.lower():
                            detected_family = "yolo"
                    except Exception:
                        pass

                model_obj = obj.get("model")
                if hasattr(model_obj, "forward"):
                    self.model = model_obj
                    if detected_family:
                        self.family = detected_family
        except Exception:
            self.model = None

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出通用PyTorch模型（只导出原始格式）"""
        if self.model is None:
            return []

        out = []
        fmts = [str(x).lower() for x in formats]

        try:
            import torch

            # PyTorch格式导出（pt/pth/safetensors）
            if "pt" in fmts or "pth" in fmts:
                pt_path = os.path.join(self.artifacts_dir, "model.pt")
                if hasattr(self.model, 'state_dict'):
                    torch.save(self.model.state_dict(), pt_path, _use_new_zipfile_serialization=False)
                else:
                    torch.save(self.model, pt_path, _use_new_zipfile_serialization=False)
                out.append(pt_path)

            if "safetensors" in fmts:
                try:
                    from safetensors.torch import save_file
                    safetensors_path = os.path.join(self.artifacts_dir, "model.safetensors")
                    if hasattr(self.model, 'state_dict'):
                        save_file(self.model.state_dict(), safetensors_path)
                    else:
                        # 如果模型没有state_dict，尝试转换为字典
                        state_dict = {k: v for k, v in self.model.named_parameters()} if hasattr(self.model, 'named_parameters') else {}
                        if state_dict:
                            save_file(state_dict, safetensors_path)
                    if os.path.exists(safetensors_path):
                        out.append(safetensors_path)
                except ImportError:
                    pass

        except Exception:
            pass

        return out
