"""非结构化剪枝实现 - 全局L1非结构化剪枝"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

try:
    from ..common import clamp
except ImportError:
    from strategies.common import clamp


def apply_unstructured(model: Any, *, target_sparsity: float, module_types: Optional[Tuple[type, ...]] = None) -> Optional[dict]:
    """对指定模块集合按 L1 进行全局非结构化剪枝。

    参数：
      model: 类 nn.Module 的模型
      target_sparsity: 目标稀疏率，范围 [0, 0.9]
      module_types: 参与剪枝的层类型元组（默认：(nn.Conv2d, nn.Linear)）

    返回：成功时返回包含实际稀疏率的字典，否则返回 None。
    """
    try:
        import torch.nn as nn  # type: ignore
        import torch.nn.utils.prune as prune  # type: ignore
    except Exception:
        return None

    amt = clamp(target_sparsity)
    if amt <= 0:
        return None
    mtypes: Tuple[type, ...] = module_types or (nn.Conv2d, nn.Linear)

    try:
        params: list[tuple[Any, str]] = []
        for m in getattr(model, "modules", lambda: [])():
            if isinstance(m, mtypes) and hasattr(m, "weight"):
                params.append((m, "weight"))
        if not params:
            return None
        prune.global_unstructured(
            params,
            pruning_method=prune.L1Unstructured,
            amount=amt,
        )
        # Make pruning permanent
        for m, _ in params:
            try:
                prune.remove(m, "weight")
            except Exception:
                pass
        return {"target_sparsity": amt}
    except Exception:
        return None

