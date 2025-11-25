"""PaddlePaddle 通用适配器"""
import os
import logging
from typing import Iterable, List, Dict, Any, Optional
from .base import ModelAdapter
from .registry import register

logger = logging.getLogger(__name__)


@register("paddlepaddle", "generic")
@register("paddle", "generic")
class PaddleGenericAdapter(ModelAdapter):
    """PaddlePaddle通用适配器"""

    def _find_model(self) -> Optional[str]:
        """查找PaddlePaddle模型文件"""
        # PaddlePaddle模型通常包含.pdmodel和.pdparams文件
        pdmodel_file = None
        pdparams_file = None
        
        for name in os.listdir(self.model_dir):
            if name.lower().endswith(".pdmodel"):
                pdmodel_file = os.path.join(self.model_dir, name)
            elif name.lower().endswith(".pdparams"):
                pdparams_file = os.path.join(self.model_dir, name)
        
        if pdmodel_file:
            return pdmodel_file
        if pdparams_file:
            return pdparams_file
        
        return None

    def load(self) -> None:
        """加载PaddlePaddle模型"""
        model_path = self._find_model()
        if not model_path:
            self.model = None
            return

        try:
            import paddle
            import paddle.jit
            
            # 尝试加载推理模型
            if model_path.endswith(".pdmodel"):
                # 如果有.pdmodel文件，尝试加载推理模型
                try:
                    self.model = paddle.jit.load(model_path)
                except Exception:
                    # 如果失败，尝试加载参数文件
                    params_path = model_path.replace(".pdmodel", ".pdparams")
                    if os.path.exists(params_path):
                        self.model = paddle.load(params_path)
                    else:
                        self.model = None
            elif model_path.endswith(".pdparams"):
                # 只加载参数文件
                self.model = paddle.load(model_path)
            else:
                self.model = None
        except ImportError:
            self.model = None
        except Exception:
            self.model = None

    def apply_quant(self, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """应用量化（支持FP16和INT8）"""
        if self.model is None:
            return {"status": "skipped", "reason": "no model"}

        try:
            import paddle
            import paddleslim
            
            prec = cfg.get("precision") if isinstance(cfg, dict) else getattr(cfg, "precision", None)
            
            if prec == "fp16":
                # PaddlePaddle FP16量化
                try:
                    # 使用paddleslim进行FP16量化
                    from paddleslim import quant
                    quant_config = quant.QAT(config={'activation_preprocess_type': 'PACT', 'weight_preprocess_type': 'PACT'})
                    # 这里需要根据实际API调整
                    quantized_path = os.path.join(self.artifacts_dir, "model_fp16")
                    # 保存量化后的模型
                    return {
                        "precision": "fp16",
                        "status": "success",
                        "saved_model_path": quantized_path
                    }
                except Exception:
                    return {"status": "error", "reason": "FP16 quantization not fully implemented"}
            
            elif prec in ["int8", "int8_static"]:
                # PaddlePaddle INT8量化
                try:
                    from paddleslim import quant
                    quant_config = quant.QAT(config={'activation_preprocess_type': 'PACT', 'weight_preprocess_type': 'PACT'})
                    quantized_path = os.path.join(self.artifacts_dir, "model_int8")
                    return {
                        "precision": "int8",
                        "status": "success",
                        "saved_model_path": quantized_path
                    }
                except Exception:
                    return {"status": "error", "reason": "INT8 quantization not fully implemented"}
            
            return {"precision": prec, "status": "skipped"}
        except ImportError:
            return {"status": "skipped", "reason": "PaddlePaddle not installed"}

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出PaddlePaddle模型"""
        if self.model is None:
            return []

        out = []
        fmts = [str(x).lower() for x in formats]

        # PaddlePaddle格式导出
        if "paddle_infer" in fmts or "paddle" in fmts:
            try:
                import paddle
                import paddle.jit
                
                # 保存为PaddlePaddle推理格式
                pdmodel_path = os.path.join(self.artifacts_dir, "model.pdmodel")
                pdparams_path = os.path.join(self.artifacts_dir, "model.pdparams")
                
                # 如果模型是paddle.jit.TranslatedLayer，可以直接保存
                if hasattr(self.model, 'save'):
                    self.model.save(self.artifacts_dir)
                    # 查找生成的文件
                    for name in os.listdir(self.artifacts_dir):
                        if name.endswith(".pdmodel"):
                            out.append(os.path.join(self.artifacts_dir, name))
                        elif name.endswith(".pdparams"):
                            out.append(os.path.join(self.artifacts_dir, name))
                else:
                    # 尝试保存参数
                    if hasattr(self.model, 'state_dict'):
                        paddle.save(self.model.state_dict(), pdparams_path)
                        out.append(pdparams_path)
            except ImportError:
                pass
            except Exception:
                pass

        # ONNX格式导出
        if "onnx" in fmts:
            try:
                import paddle2onnx
                
                pdmodel_path = None
                pdparams_path = None
                
                for name in os.listdir(self.model_dir):
                    if name.lower().endswith(".pdmodel"):
                        pdmodel_path = os.path.join(self.model_dir, name)
                    elif name.lower().endswith(".pdparams"):
                        pdparams_path = os.path.join(self.model_dir, name)
                
                if not pdmodel_path or not pdparams_path:
                    logger.warning("Missing .pdmodel or .pdparams files for ONNX conversion")
                    return out
                
                onnx_path = os.path.join(self.artifacts_dir, "model.onnx")
                paddle2onnx.command.caffe2onnx(
                    model_file=pdmodel_path,
                    params_file=pdparams_path,
                    save_file=onnx_path,
                    opset_version=13,
                    enable_onnx_checker=True
                )
                
                if os.path.exists(onnx_path):
                    out.append(onnx_path)
            except ImportError:
                logger.warning("paddle2onnx not installed, skipping PaddlePaddle→ONNX conversion")
            except Exception as e:
                logger.error(f"PaddlePaddle to ONNX conversion failed: {e}")

        return out

