"""策略模块公共工具函数"""

import json
import os
from typing import Any, Dict, Optional


def clamp(value: float, low: float = 0.0, high: float = 0.9) -> float:
    """限制值在指定范围内"""
    return max(low, min(high, float(value)))


def write_report(artifacts_dir: Optional[str], obj: Dict[str, Any], filename: str = "report.json") -> Optional[str]:
    """写入训练报告（统一版本）"""
    if not artifacts_dir:
        return None
    try:
        os.makedirs(artifacts_dir, exist_ok=True)
        p = os.path.join(artifacts_dir, filename)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return p
    except Exception:
        return None


def evaluate_accuracy(model: Any, data_loader: Any) -> float:
    """评估模型精度（统一版本）"""
    try:
        import torch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in data_loader:
                outputs = model(images)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        return correct / total if total > 0 else 0.0
    except Exception:
        return 0.0

