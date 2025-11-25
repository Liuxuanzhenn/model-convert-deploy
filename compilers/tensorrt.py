"""TensorRT编译器"""
import os
import shutil
import subprocess
import logging
from typing import Dict, Any
from .base import HardwareCompiler
from utils.security import sanitize_input_shape, sanitize_path

logger = logging.getLogger(__name__)


class TensorRTCompiler(HardwareCompiler):
    """TensorRT编译器"""

    def compile(self, model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("trtexec not found. Please install TensorRT")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_path = self._prepare_onnx_model(model_path, config)
        optimization = config.get("optimization", {})
        
        output_name = "model_int8.engine" if optimization.get("int8") else \
                     "model_fp16.engine" if optimization.get("fp16") else "model.engine"
        output_path = os.path.join(self.output_dir, output_name)
        
        cmd = ["trtexec", f"--onnx={model_path}", f"--saveEngine={output_path}"]
        
        if optimization.get("fp16"):
            cmd.append("--fp16")
        if optimization.get("int8"):
            cmd.append("--int8")
            calib_cache = config.get("calib_cache")
            if calib_cache:
                safe_calib = sanitize_path(str(calib_cache), os.path.dirname(calib_cache) or ".")
                if os.path.exists(safe_calib):
                    cmd.append(f"--calib={safe_calib}")
        
        workspace_size = max(1, min(int(optimization.get("workspace_size", 4096)), 32768))
        cmd.append(f"--workspace={workspace_size}")
        
        builder_opt_level = max(0, min(int(optimization.get("builder_optimization_level", 5)), 5))
        cmd.append(f"--builderOptimizationLevel={builder_opt_level}")
        
        input_shape = config.get("input_shape")
        if input_shape:
            cmd.append(f"--shapes={sanitize_input_shape(str(input_shape))}")
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"TensorRT compilation failed:\n{result.stderr}")
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output file not generated: {output_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("TensorRT compilation timeout (>10 minutes)")
        
        input_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        compression_ratio = (1 - output_size_mb / input_size_mb) * 100 if input_size_mb > 0 else 0
        
        return {
            "output_path": output_path,
            "input_format": "onnx",
            "output_format": "engine",
            "hardware": "tensorrt",
            "precision": "int8" if optimization.get("int8") else ("fp16" if optimization.get("fp16") else "fp32"),
            "input_size_mb": round(input_size_mb, 2),
            "output_size_mb": round(output_size_mb, 2),
            "compression_ratio": f"{compression_ratio:.1f}%",
            "estimated_time": "3-8分钟",
            "speedup": "~5.0x"
        }

    def is_available(self) -> bool:
        return shutil.which("trtexec") is not None
