"""PyTorch YOLO适配器"""

import os
import glob
import shutil
from typing import Iterable, List, Optional
from .base import ModelAdapter
from .registry import register


@register("pytorch", "yolo")
class PytorchYoloAdapter(ModelAdapter):
    """YOLO适配器"""

    def load(self) -> None:
        """加载YOLO模型"""
        weight = self._find_weight()
        if not weight:
            self.model = None
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(weight)
        except ImportError:
            self.model = None

    def _extract_state_dict(self) -> Optional[object]:
        """提取模型的state_dict"""
        if hasattr(self.model, "model") and hasattr(self.model.model, "state_dict"):
            return self.model.model.state_dict()
        
        if hasattr(self.model, "ckpt"):
            ckpt = self.model.ckpt
            if isinstance(ckpt, dict):
                if "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
                    return ckpt["model"].state_dict()
                if "state_dict" in ckpt:
                    return ckpt["state_dict"]
                return ckpt
            if hasattr(ckpt, "state_dict"):
                return ckpt.state_dict()
            return ckpt
        
        if hasattr(self.model, "state_dict"):
            return self.model.state_dict()
        
        return None

    def _save_pytorch(self) -> Optional[str]:
        """保存PyTorch格式模型"""
        import torch
        state_dict = self._extract_state_dict()
        if state_dict is None:
            return None
        
        pt_path = os.path.join(self.artifacts_dir, "yolov8n.pt")
        torch.save(state_dict, pt_path, _use_new_zipfile_serialization=False)
        return pt_path

    def _export_ultralytics_format(self, fmt: str) -> Optional[str]:
        """使用ultralytics的export方法导出"""
        try:
            self.model.export(format=fmt, half=False, project=self.artifacts_dir, name="")
        except TypeError:
            self.model.export(format=fmt, half=False)
        
        patterns = {"torchscript": "*.torchscript*", "onnx": "*.onnx"}
        pattern = patterns.get(fmt)
        if not pattern:
            return None
        
        search_roots = [(self.artifacts_dir, True), (os.getcwd(), True), (self.model_dir, False)]
        for root, recursive in search_roots:
            if not os.path.isdir(root):
                continue
            
            matches = glob.glob(os.path.join(root, "**" if recursive else "", pattern), recursive=recursive)
            for src in matches:
                if root == self.model_dir and ("raw" in src.replace("\\", "/") or src.startswith(self.model_dir)):
                    continue
                if os.path.isfile(src) and os.path.getsize(src) > 0:
                    if not src.startswith(self.artifacts_dir):
                        dst = os.path.join(self.artifacts_dir, os.path.basename(src))
                        shutil.move(src, dst)
                        return dst
                    return src
        
        return None

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出YOLO模型（只导出原始格式）"""
        if self.model is None:
            return []

        out = []
        fmts = [str(x).lower() for x in formats]

        # PyTorch格式导出
        if "pt" in fmts or "pth" in fmts:
            pt_path = self._save_pytorch()
            if pt_path:
                out.append(pt_path)

        # Safetensors格式导出
        if "safetensors" in fmts:
            try:
                from safetensors.torch import save_file
                safetensors_path = os.path.join(self.artifacts_dir, "model.safetensors")
                state_dict = self._extract_state_dict()
                if state_dict:
                    save_file(state_dict, safetensors_path)
                    out.append(safetensors_path)
            except ImportError:
                pass

        return out
