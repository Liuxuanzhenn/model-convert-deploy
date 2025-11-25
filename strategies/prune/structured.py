from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

try:
    from ..common import clamp
except ImportError:
    from strategies.common import clamp


def select_sparsity(constraints: Optional[Dict[str, Any]] = None, search: Optional[Dict[str, Any]] = None, default: float = 0.5) -> float:
    """基于简单启发式从搜索空间中选择目标稀疏率。

    规则：
    - 若提供 `search["space"]`（如 [0.2,0.4,0.6]），取其中位数作为默认值；
    - 若给定 `constraints.flops_reduction`，优先选择"不超过该值的最大候选"；
    - 否则回退到 `default`，并将结果裁剪到 [0, 0.9]。
    """
    space: Iterable[float] = []
    if search and isinstance(search.get("space"), (list, tuple)):
        space = list(search.get("space"))  # type: ignore
    cand = list(space) if space else [default]
    cand = sorted([clamp(x) for x in cand])
    tgt = cand[len(cand) // 2]
    if constraints and (constraints.get("flops_reduction") is not None):
        try:
            fr = float(constraints.get("flops_reduction"))
            fr = clamp(fr)
            le = [x for x in cand if x <= fr]
            if le:
                tgt = max(le)
        except Exception:
            pass
    return clamp(tgt)


def apply_structured_bn(model: Any, *, target_sparsity: float) -> Optional[Dict[str, float]]:
    """基于BatchNorm权重的结构化通道剪枝"""
    try:
        import torch.nn as nn
        import torch.nn.utils.prune as prune
    except Exception:
        return None
    
    amount = clamp(target_sparsity)
    if amount <= 0:
        return None
    
    try:
        changed = False
        has_bn = False
        modules_list = list(model.named_modules())
        
        for i, (name, m) in enumerate(modules_list):
            if isinstance(m, nn.Conv2d) and hasattr(m, "weight"):
                bn_module = None
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                
                for j in range(i + 1, min(i + 5, len(modules_list))):
                    next_name, next_m = modules_list[j]
                    if isinstance(next_m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        if parent_name and next_name.startswith(parent_name):
                            bn_module = next_m
                            break
                        elif not parent_name or next_name.count(".") <= name.count("."):
                            bn_module = next_m
                            break
                
                if bn_module and hasattr(bn_module, "weight") and bn_module.weight is not None:
                    has_bn = True
                    if bn_module.weight.shape[0] > 0:
                        prune.ln_structured(m, name="weight", amount=amount, n=1, dim=0)
                        prune.remove(m, "weight")
                        changed = True
        
        return {"target_sparsity": amount, "method": "bn_based"} if (has_bn and changed) else None
    except Exception:
        return None


def apply_structured(model: Any, *, target_sparsity: float) -> Optional[Dict[str, float]]:
    """结构化通道剪枝：优先BN权重，无BN则回退Ln范数"""
    amount = clamp(target_sparsity)
    if amount <= 0:
        return None
    
    result = apply_structured_bn(model, target_sparsity=amount)
    if result:
        return result
    
    try:
        import torch.nn.utils.prune as prune
        import torch.nn as nn
    except Exception:
        return None
    
    try:
        changed = False
        for m in getattr(model, "modules", lambda: [])():
            if isinstance(m, nn.Conv2d) and hasattr(m, "weight"):
                try:
                    prune.ln_structured(m, name="weight", amount=amount, n=1, dim=0)
                    prune.remove(m, "weight")
                    changed = True
                except Exception:
                    pass
        return {"target_sparsity": amount, "method": "ln_norm"} if changed else None
    except Exception:
        return None

