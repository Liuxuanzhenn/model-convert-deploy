"""自动剪枝策略选择器 - 根据模型类型自动选择最优剪枝方法"""

from typing import Any, Dict, Optional, Tuple

from strategies.prune.structured import apply_structured, select_sparsity
from strategies.prune.unstructured import apply_unstructured


def _get_model_size_mb(model: Any) -> float:
    """估算模型大小（MB）"""
    try:
        import torch
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    except Exception:
        return 0.0


def _analyze_model_structure(model: Any) -> Tuple[int, int]:
    """分析模型结构：返回(卷积层数, 线性层数)"""
    try:
        import torch.nn as nn
        conv_count = linear_count = 0
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d)):
                conv_count += 1
            elif isinstance(m, nn.Linear):
                linear_count += 1
        return conv_count, linear_count
    except Exception:
        return 0, 0


def _select_default_sparsity(family: str, model_size_mb: float, constraints: Optional[Dict[str, Any]] = None) -> float:
    """根据模型类型和大小选择默认稀疏度"""
    family_lower = str(family or "generic").lower()
    
    if constraints and constraints.get("max_accuracy_drop", 1.0) < 0.02:
        return 0.1
    
    visual_models = ["resnet", "vgg", "cnn", "yolo", "inception", "inceptionv4", "van", "alexnet", "squeezenet", "densenet"]
    if family_lower in visual_models:
        base_sparsity = 0.35 if model_size_mb > 200 else 0.3
    elif family_lower in ["transformer", "vit", "bert"]:
        base_sparsity = 0.25
    elif family_lower in ["lstm", "rnn", "vae"]:
        base_sparsity = 0.2
    elif family_lower == "gcn":
        base_sparsity = 0.25
    else:
        base_sparsity = 0.3 if model_size_mb > 200 else 0.25
    
    if model_size_mb > 500:
        base_sparsity = min(base_sparsity + 0.1, 0.5)
    elif model_size_mb < 50:
        base_sparsity = max(base_sparsity - 0.1, 0.15)
    
    return max(0.1, min(0.5, base_sparsity))


def decide_and_apply_prune(
    model: Any,
    cfg: Dict[str, Any],
    family: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """自动剪枝策略选择和应用（支持模型感知）
    
    Args:
        model: 待剪枝模型
        cfg: 剪枝配置（type/target_sparsity/search_space/flops_reduction/constraints）
        family: 模型家族
        
    Returns:
        dict: 包含 "target_sparsity"、"chosen_strategy"、"reason" 等执行信息的字典
    """
    if not isinstance(cfg, dict) or not cfg.get("enable", True):
        return None

    ptype = cfg.get("type", "structured")
    family_lower = str(family or "generic").lower()
    constraints = cfg.get("constraints", {})
    model_size_mb = _get_model_size_mb(model)
    conv_count, linear_count = _analyze_model_structure(model)
    
    visual_models = ["resnet", "vgg", "cnn", "yolo", "inception", "inceptionv4", "van", "alexnet", "squeezenet", "densenet"]
    
    if ptype == "auto":
        if family_lower in ["transformer", "vit", "bert"]:
            ptype, reason = "unstructured", "Transformer models prefer unstructured pruning for attention layers"
        elif family_lower in ["lstm", "rnn"]:
            ptype, reason = "unstructured", "LSTM/RNN models use unstructured pruning on Linear layers only"
        elif family_lower in ["gcn", "vae"]:
            ptype, reason = "unstructured", f"{family_lower.upper()} models use unstructured pruning"
        elif family_lower in visual_models or conv_count > linear_count * 2:
            ptype, reason = "structured", f"{family_lower.upper()} visual model, using structured pruning on Conv layers"
        else:
            ptype, reason = "structured", "Default structured pruning for visual models"
    else:
        reason = f"User specified {ptype} pruning"
    
    if cfg.get("search") or cfg.get("flops_reduction"):
        tgt = select_sparsity(
            constraints=constraints if constraints else {"flops_reduction": cfg.get("flops_reduction")},
            search=cfg.get("search"),
            default=cfg.get("target_sparsity")
        )
    elif cfg.get("target_sparsity") is not None:
        tgt = float(cfg.get("target_sparsity"))
    else:
        tgt = _select_default_sparsity(family_lower, model_size_mb, constraints)
    
    tgt = max(0.1, min(0.9, float(tgt)))
    
    result = None
    fallback_reason = None
    
    if ptype == "unstructured":
        module_types = None
        if family_lower in ["lstm", "rnn", "gcn"]:
            try:
                import torch.nn as nn
                module_types = (nn.Linear,)
                reason += ", Linear-only"
            except Exception:
                pass
        result = apply_unstructured(model, target_sparsity=tgt, module_types=module_types)
        if not result:
            result = apply_structured(model, target_sparsity=tgt)
            if result:
                ptype, reason = "structured", "Fallback to structured pruning (BN-based by default)"
    else:
        result = apply_structured(model, target_sparsity=tgt)
        if not result:
            result = apply_unstructured(model, target_sparsity=tgt)
            if result:
                ptype, reason = "unstructured", "Fallback to unstructured pruning"
    
    if result:
        result.update({
            "chosen_strategy": ptype,
            "reason": reason,
            "model_size_mb": round(model_size_mb, 2),
            "conv_layers": conv_count,
            "linear_layers": linear_count
        })
        if fallback_reason:
            result["fallback_reason"] = fallback_reason
        if ptype == "structured":
            result["note"] = "Structured pruning masks parameters. To reduce file size, rebuild model or export to ONNX/TensorRT"
    
    return result
