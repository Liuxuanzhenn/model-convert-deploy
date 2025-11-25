"""ONNX 通用适配器"""

import os
from typing import Iterable, List, Dict, Any, Optional
from .base import ModelAdapter
from .registry import register


@register("onnx", "generic")
class OnnxGenericAdapter(ModelAdapter):
    """ONNX通用适配器"""

    def load(self) -> None:
        """加载ONNX模型"""
        weight = self._find_weight(extensions=(".onnx",))
        if not weight:
            self.model = None
            return

        try:
            import onnx
            self.model = onnx.load(weight)
        except ImportError:
            self.model = None
        except Exception:
            self.model = None

    def apply_quant(self, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """应用量化（支持FP16和INT8）"""
        if self.model is None:
            return {"status": "skipped", "reason": "no model"}

        try:
            import onnx
            prec = self._get_cfg(cfg, "precision")
            auto = self._get_cfg(cfg, "auto", False)
            
            if auto and not prec:
                prec = "fp16"
            
            if prec == "fp16":
                try:
                    from onnxconverter_common import float16
                    model_path = self._find_weight(extensions=(".onnx",))
                    if not model_path:
                        return {"status": "error", "reason": "ONNX model not found"}
                    
                    model_fp32 = onnx.load(model_path)
                    model_fp16 = float16.convert_float_to_float16(model_fp32)
                    op_name = "quantized_auto" if auto else "quantized_fp16"
                    quantized_path = os.path.join(self.artifacts_dir, f"model_{op_name}.onnx")
                    onnx.save(model_fp16, quantized_path)
                    self.model = model_fp16
                    self._operations.append(op_name)
                    return {
                        "precision": "fp16",
                        "status": "success",
                        "onnx_path": quantized_path
                    }
                except ImportError:
                    return {"status": "error", "reason": "onnxconverter-common not installed"}
                except Exception as e:
                    return {"status": "error", "reason": str(e)}
            
            from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
            
            if prec in ["int8", "int8_static"]:
                try:
                    model_path = self._find_weight(extensions=(".onnx",))
                    if not model_path:
                        return {"status": "error", "reason": "ONNX model not found"}
                    
                    op_name = "quantized_auto" if auto else "quantized_int8"
                    quantized_path = os.path.join(self.artifacts_dir, f"model_{op_name}.onnx")
                    quantize_static(
                        model_path,
                        quantized_path,
                        quant_type=QuantType.QInt8
                    )
                    self.model = onnx.load(quantized_path)
                    self._operations.append(op_name)
                    return {
                        "precision": "int8",
                        "status": "success",
                        "onnx_path": quantized_path
                    }
                except ImportError:
                    return {"status": "error", "reason": "onnxruntime not installed"}
                except Exception as e:
                    return {"status": "error", "reason": str(e)}
            
            elif prec == "int8_dynamic":
                try:
                    model_path = self._find_weight(extensions=(".onnx",))
                    if not model_path:
                        return {"status": "error", "reason": "ONNX model not found"}
                    
                    op_name = "quantized_auto" if auto else "quantized_int8_dynamic"
                    quantized_path = os.path.join(self.artifacts_dir, f"model_{op_name}.onnx")
                    quantize_dynamic(
                        model_path,
                        quantized_path,
                        weight_type=QuantType.QInt8
                    )
                    self.model = onnx.load(quantized_path)
                    self._operations.append(op_name)
                    return {
                        "precision": "int8_dynamic",
                        "status": "success",
                        "onnx_path": quantized_path
                    }
                except ImportError:
                    return {"status": "error", "reason": "onnxruntime not installed"}
                except Exception as e:
                    return {"status": "error", "reason": str(e)}
            
            return {"precision": prec, "status": "skipped"}
        except ImportError:
            return {"status": "skipped", "reason": "ONNX libraries not installed"}

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出ONNX模型（只导出原始格式）"""
        if self.model is None:
            return []

        out = []
        fmts = [str(x).lower() for x in formats]

        # ONNX格式导出
        if "onnx" in fmts:
            try:
                import onnx
                onnx_path = os.path.join(self.artifacts_dir, "model.onnx")
                onnx.save(self.model, onnx_path)
                out.append(onnx_path)
            except ImportError:
                pass
            except Exception:
                pass

        return out

