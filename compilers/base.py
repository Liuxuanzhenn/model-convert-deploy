"""硬件编译器基类"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import logging

logger = logging.getLogger(__name__)


class HardwareCompiler(ABC):
    """硬件编译器基类"""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    @abstractmethod
    def compile(self, model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """编译模型为硬件专用格式"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """检查编译工具是否可用"""
        pass

    def _convert_pytorch_to_onnx(self, pytorch_path: str, input_shape: Optional[tuple] = None) -> str:
        """将PyTorch模型转换为ONNX
        
        Args:
            pytorch_path: PyTorch模型路径
            input_shape: 输入形状，默认(1,3,224,224)
        
        Returns:
            ONNX模型路径
        """
        import torch
        
        try:
            model = torch.load(pytorch_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        if isinstance(model, dict):
            raise ValueError("Input is state_dict, cannot convert to ONNX without model architecture")
        
        if input_shape is None:
            input_shape = (1, 3, 224, 224)
        
        onnx_path = pytorch_path.rsplit(".", 1)[0] + "_temp.onnx"
        dummy_input = torch.randn(*input_shape)
        
        torch.onnx.export(
            model, dummy_input, onnx_path,
            input_names=["input"], output_names=["output"],
            opset_version=13, do_constant_folding=True
        )
        
        return onnx_path

    def _detect_input_format(self, model_path: str) -> str:
        """检测输入模型格式"""
        if model_path.endswith((".pt", ".pth", ".safetensors")):
            return "pytorch"
        elif model_path.endswith(".onnx"):
            return "onnx"
        elif model_path.endswith(".pb"):
            return "tensorflow"
        return "unknown"

    def _prepare_onnx_model(self, model_path: str, config: Dict[str, Any]) -> str:
        """准备ONNX模型（自动转换PyTorch）"""
        input_format = self._detect_input_format(model_path)
        
        if input_format == "pytorch":
            input_shape = config.get("input_shape")
            if isinstance(input_shape, str):
                try:
                    shape_part = input_shape.split(":")[-1]
                    shape_tuple = tuple(map(int, shape_part.split(",")))
                except:
                    shape_tuple = None
            else:
                shape_tuple = input_shape
            
            model_path = self._convert_pytorch_to_onnx(model_path, shape_tuple)
        
        if not model_path.endswith(".onnx"):
            raise ValueError(f"Unsupported input format: {input_format}")
        
        return model_path
