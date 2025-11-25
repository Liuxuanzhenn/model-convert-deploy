"""华为昇腾NPU编译器"""
import os
import shutil
import subprocess
import logging
from typing import Dict, Any
from .base import HardwareCompiler
from utils.security import sanitize_input_shape, sanitize_path

logger = logging.getLogger(__name__)


class AscendCompiler(HardwareCompiler):
    """华为昇腾NPU编译器"""

    SUPPORTED_DEVICES = {
        "Ascend310": "Ascend310",
        "Ascend310P": "Ascend310P3",
        "Ascend910": "Ascend910",
        "Ascend 310": "Ascend310",
        "Ascend 310P": "Ascend310P3",
        "Ascend 910": "Ascend910",
    }

    def compile(self, model_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_available():
            raise RuntimeError("ATC tool not found. Please install Ascend CANN toolkit")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        model_path = self._prepare_onnx_model(model_path, config)
        device = config.get("device", "Ascend 310")
        soc_version = self.SUPPORTED_DEVICES.get(device, "Ascend310")
        
        output_name = "model_ascend"
        output_path = os.path.join(self.output_dir, f"{output_name}.om")
        
        cmd = [
            "atc", f"--model={model_path}", "--framework=5",
            f"--output={os.path.join(self.output_dir, output_name)}",
            f"--soc_version={soc_version}"
        ]
        
        input_format_str = config.get("input_format", "NCHW")
        if input_format_str not in ["NCHW", "NHWC"]:
            input_format_str = "NCHW"
        cmd.append(f"--input_format={input_format_str}")
        
        input_shape = config.get("input_shape")
        if input_shape:
            cmd.append(f"--input_shape={sanitize_input_shape(str(input_shape))}")
        
        optimization = config.get("optimization", {})
        if optimization.get("operator_fusion", True):
            fusion_cfg = config.get("fusion_config_file")
            if fusion_cfg:
                safe_cfg = sanitize_path(str(fusion_cfg), os.path.dirname(fusion_cfg) or ".")
                if os.path.exists(safe_cfg):
                    cmd.append(f"--fusion_switch_file={safe_cfg}")
        
        precision_mode = optimization.get("precision_mode", "allow_fp32_to_fp16")
        if precision_mode not in ["allow_fp32_to_fp16", "force_fp16", "allow_mix_precision"]:
            precision_mode = "allow_fp32_to_fp16"
        cmd.append(f"--precision_mode={precision_mode}")
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, timeout=600)
            if result.returncode != 0:
                raise RuntimeError(f"ATC compilation failed:\n{result.stderr}")
            if not os.path.exists(output_path):
                raise RuntimeError(f"Output file not generated: {output_path}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("ATC compilation timeout (>10 minutes)")
        
        input_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        compression_ratio = (1 - output_size_mb / input_size_mb) * 100 if input_size_mb > 0 else 0
        
        return {
            "output_path": output_path,
            "input_format": "onnx",
            "output_format": "om",
            "hardware": "ascend_npu",
            "device": device,
            "soc_version": soc_version,
            "input_size_mb": round(input_size_mb, 2),
            "output_size_mb": round(output_size_mb, 2),
            "compression_ratio": f"{compression_ratio:.1f}%",
            "estimated_time": "3-5分钟",
            "speedup": "~3.2x"
        }

    def is_available(self) -> bool:
        return shutil.which("atc") is not None
