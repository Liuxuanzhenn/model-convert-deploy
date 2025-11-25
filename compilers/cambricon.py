"""寒武纪思元系列编译器"""
import os
import shutil
import subprocess
import logging
from typing import Dict, Any
from .base import HardwareCompiler

logger = logging.getLogger(__name__)


class CambriconCompiler(HardwareCompiler):
    """寒武纪思元系列编译器"""

    SUPPORTED_DEVICES = {
        "MLU220": "mlu220",
        "MLU270": "mlu270",
        "MLU370": "mlu370",
        "MLU370-X4": "mlu370-x4",
        "MLU370-X8": "mlu370-x8",
        "MLU 220": "mlu220",
        "MLU 270": "mlu270",
        "MLU 370": "mlu370",
    }

    def compile(self, model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("CNCC tool not found. Please install Cambricon Neuware SDK")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_path = self._prepare_onnx_model(model_path, config)
        device = config.get("device", "MLU370")
        mlu_arch = self.SUPPORTED_DEVICES.get(device, "mlu370")
        
        output_name = "model_cambricon"
        output_path = os.path.join(self.output_dir, f"{output_name}.cambricon")
        
        cmd = [
            "cncc", f"--model={model_path}", "--framework=onnx",
            f"--output={output_path}", f"--mlu_arch={mlu_arch}"
        ]
        
        optimization = config.get("optimization", {})
        quant_mode = optimization.get("quantization", "int8")
        if quant_mode in ["int8", "int16", "fp16"]:
            cmd.append(f"--quant_mode={quant_mode}")
        
        batch_size = optimization.get("batch_size", 1)
        cmd.append(f"--batch_size={batch_size}")
        
        input_shape = config.get("input_shape")
        if input_shape:
            cmd.append(f"--input_shape={input_shape}")
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"CNCC compilation failed:\n{result.stderr}")
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output file not generated: {output_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("CNCC compilation timeout (>10 minutes)")
        
        input_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        compression_ratio = (1 - output_size_mb / input_size_mb) * 100 if input_size_mb > 0 else 0
        
        return {
            "output_path": output_path,
            "input_format": "onnx",
            "output_format": "cambricon",
            "hardware": "cambricon",
            "device": device,
            "mlu_arch": mlu_arch,
            "quantization": quant_mode,
            "input_size_mb": round(input_size_mb, 2),
            "output_size_mb": round(output_size_mb, 2),
            "compression_ratio": f"{compression_ratio:.1f}%",
            "estimated_time": "3-6分钟",
            "speedup": "~3.0x"
        }

    def is_available(self) -> bool:
        return shutil.which("cncc") is not None
