"""自动量化策略选择器 - 根据配置和模型类型自动选择最优量化方法"""

from typing import Any, Dict, Tuple, Optional

from strategies.quant.ptq import apply_fp16, apply_int8_dynamic, apply_int8_static
from strategies.quant.qat import apply_qat


def _get_model_size_mb(model: Any) -> float:
    """估算模型大小（MB）"""
    try:
        import torch
        total_params = sum(p.numel() for p in model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    except Exception:
        return 0.0


def decide_and_apply_quant(model: Any, qc: Dict[str, Any], family: Optional[str] = None) -> Tuple[Any, Dict[str, Any]]:
    """自动量化策略选择和应用（支持模型感知）
    
    Args:
        model: 待量化模型
        qc: 量化配置字典（precision/bits/auto/calib_dir/calib_num）
        family: 模型家族（yolo/resnet/lstm/rnn/gcn/vae/transformer等）
    
    Returns:
        (量化后的模型, 信息字典)
    """
    if not isinstance(qc, dict):
        qc = {}
    
    precision = qc.get("precision")
    bits = qc.get("bits")
    auto = bool(qc.get("auto", False))
    calib_dir = qc.get("calib_dir")
    calib_num = qc.get("calib_num")
    
    info: Dict[str, Any] = {}
    family_lower = str(family or "generic").lower()
    
    if precision == "fp16":
        m, i = apply_fp16(model)
        info.update(i)
        return m, info
    if precision == "int8_dynamic":
        m, i = apply_int8_dynamic(model)
        info.update(i)
        return m, info
    if precision == "int8_static":
        m, i = apply_int8_static(model, calib_dir=calib_dir, calib_num=calib_num)
        info.update(i)
        return m, info
    if precision in ["qat", "int8_qat"]:
        m, i = apply_qat(model, qc)
        info.update(i)
        return m, info
    
    if bits == 16:
        m, i = apply_fp16(model)
        info.update(i)
        return m, info
    
    if bits == 8:
        if family_lower in ["lstm", "rnn"]:
            import torch.nn as nn
            m, i = apply_int8_dynamic(model, module_types=(nn.Linear,))
            i.update({"strategy": "linear_only", "preserved_layers": "LSTM/RNN (FP32)"})
            info.update(i)
            return m, info
        elif family_lower == "gcn":
            import torch.nn as nn
            m, i = apply_int8_dynamic(model, module_types=(nn.Linear,))
            i.update({"strategy": "linear_only", "preserved_layers": "GraphConv (FP32)"})
            info.update(i)
            return m, info
        elif family_lower == "vae":
            try:
                import torch.nn as nn
                if hasattr(model, "encoder") and hasattr(model, "decoder"):
                    model.encoder = apply_int8_dynamic(model.encoder, module_types=(nn.Linear,))[0]
                    model.decoder = apply_fp16(model.decoder)[0] if hasattr(model.decoder, "half") else model.decoder
                    info.update({"precision": "mixed_int8_fp16", "strategy": "encoder_quantized_decoder_fp16"})
                    return model, info
            except Exception:
                pass
            m, i = apply_fp16(model)
            info.update(i)
            return m, info
        elif family_lower == "transformer":
            m, i = apply_int8_dynamic(model)
            i["strategy"] = "attention_aware"
            info.update(i)
            return m, info
        else:
            visual_models = ["resnet", "vgg", "cnn", "yolo", "inception", "inceptionv4", "van", "alexnet", "squeezenet", "densenet", "vit"]
            if family_lower in visual_models and (calib_dir or calib_num):
                m, i = apply_int8_static(model, calib_dir=calib_dir, calib_num=calib_num)
            elif auto and (calib_dir or calib_num):
                m, i = apply_int8_static(model, calib_dir=calib_dir, calib_num=calib_num)
            else:
                m, i = apply_int8_dynamic(model)
            info.update(i)
            return m, info
    
    if auto and not precision and not bits:
        model_size_mb = _get_model_size_mb(model)
        
        if calib_dir or calib_num:
            try:
                m, i = apply_int8_static(model, calib_dir=calib_dir, calib_num=calib_num)
                info.update(i)
                return m, info
            except Exception:
                pass
        
        visual_models = ["resnet", "vgg", "cnn", "yolo", "inception", "inceptionv4", "van", "alexnet", "squeezenet", "densenet", "vit"]
        if family_lower in visual_models:
            m, i = apply_fp16(model)
            info.update(i)
            return m, info
        elif family_lower in ["lstm", "rnn"]:
            import torch.nn as nn
            m, i = apply_int8_dynamic(model, module_types=(nn.Linear,))
            i.update({"strategy": "auto_selected", "reason": "LSTM/RNN Linear-only quantization"})
            info.update(i)
            return m, info
        elif family_lower == "transformer":
            m, i = apply_int8_dynamic(model)
            i.update({"strategy": "auto_selected", "reason": "Transformer INT8 dynamic"})
            info.update(i)
            return m, info
        elif family_lower == "gcn":
            import torch.nn as nn
            m, i = apply_int8_dynamic(model, module_types=(nn.Linear,))
            i.update({"strategy": "auto_selected", "reason": "GCN Linear-only quantization"})
            info.update(i)
            return m, info
        elif family_lower == "vae":
            try:
                import torch.nn as nn
                if hasattr(model, "encoder") and hasattr(model, "decoder"):
                    model.encoder = apply_int8_dynamic(model.encoder, module_types=(nn.Linear,))[0]
                    model.decoder = apply_fp16(model.decoder)[0] if hasattr(model.decoder, "half") else model.decoder
                    info.update({"precision": "mixed_int8_fp16", "strategy": "auto_selected", "reason": "VAE encoder INT8 + decoder FP16"})
                    return model, info
            except Exception:
                pass
            m, i = apply_fp16(model)
            info.update(i)
            return m, info
        else:
            if model_size_mb > 500:
                m, i = apply_fp16(model)
                i.update({"strategy": "auto_selected", "reason": f"Large model ({model_size_mb:.1f}MB), using FP16"})
            else:
                m, i = apply_fp16(model)
                i.update({"strategy": "auto_selected", "reason": "Unknown model type, default FP16"})
            info.update(i)
            return m, info
    
    return model, info
