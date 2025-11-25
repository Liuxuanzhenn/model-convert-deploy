"""压缩效果预估器"""
from typing import Dict, Any, List, Optional

from compression.capabilities_v2 import get_registry_v2


class CompressionEstimator:
    """压缩效果预估器"""

    def __init__(self):
        self.registry = get_registry_v2()
        self._load_history()

    def _load_history(self) -> None:
        """加载历史压缩记录"""
        self.history = []

    def estimate(self, model_info: Dict[str, Any], strategy: Dict[str, Any]) -> Dict[str, Any]:
        """预估压缩效果"""
        if not self.registry:
            return self._default_estimate(
                model_info.get("model_size_mb"),
                model_info.get("current_latency_ms")
            )
        
        framework = model_info.get("framework", "pytorch")
        family = model_info.get("family", "generic")
        base_size = model_info.get("model_size_mb")
        base_latency = model_info.get("current_latency_ms")

        cap = self.registry.get(framework, family)
        if not cap:
            return self._default_estimate(base_size, base_latency)

        if base_size is None:
            base_size = cap.get("typical_size_mb", 100)
        if base_latency is None:
            base_latency = cap.get("typical_latency_ms", 100)

        total_size_reduction = 1.0
        total_speedup = 1.0
        total_acc_drop = 0.0
        applied_techs = []

        quant_cfg = strategy.get("quantize", {})
        if quant_cfg.get("enable"):
            precision = quant_cfg.get("precision", "fp16")
            quant_effect = self._estimate_quantize(cap, precision)
            total_size_reduction *= quant_effect["size_reduction"]
            total_speedup *= quant_effect["speedup"]
            total_acc_drop += quant_effect["accuracy_drop"]
            applied_techs.append(f"量化({precision})")

        prune_cfg = strategy.get("prune", {})
        if prune_cfg.get("enable"):
            sparsity = prune_cfg.get("target_sparsity", 0.3)
            prune_effect = self._estimate_prune(cap, sparsity)
            total_size_reduction *= prune_effect["size_reduction"]
            total_speedup *= prune_effect["speedup"]
            total_acc_drop += prune_effect["accuracy_drop"]
            applied_techs.append(f"剪枝({sparsity})")

        distill_cfg = strategy.get("distill", {})
        if distill_cfg.get("enable"):
            distill_effect = self._estimate_distill(cap)
            total_size_reduction *= distill_effect["size_reduction"]
            total_speedup *= distill_effect["speedup"]
            total_acc_drop += distill_effect["accuracy_drop"]
            applied_techs.append("蒸馏")

        if len(applied_techs) > 1:
            total_size_reduction *= 1.05
            total_speedup *= 0.85
            total_acc_drop *= 1.15

        estimated_size = base_size * total_size_reduction
        estimated_latency = base_latency / total_speedup
        size_reduction_ratio = 1 - total_size_reduction

        confidence = self._calculate_confidence(cap, applied_techs)

        return {
            "estimated_size_mb": round(estimated_size, 2),
            "estimated_latency_ms": round(estimated_latency, 1),
            "estimated_accuracy_drop": round(total_acc_drop, 4),
            "size_reduction_ratio": round(size_reduction_ratio, 3),
            "speedup": round(total_speedup, 2),
            "confidence": confidence,
            "applied_techniques": applied_techs,
            "reference_models": [f"{framework}.{family}"],
            "baseline": {
                "size_mb": base_size,
                "latency_ms": base_latency
            }
        }

    def _estimate_quantize(self, cap: Dict[str, Any], precision: str) -> Dict[str, float]:
        """预估量化效果"""
        quant_cap = cap.get("compression", {}).get("quantize", {})
        methods = quant_cap.get("methods", {})
        method_config = methods.get(precision, {})
        expected_effects = method_config.get("expected_effects", {})
        return {
            "size_reduction": expected_effects.get("size_reduction", 0.5),
            "speedup": expected_effects.get("speedup", 1.5),
            "accuracy_drop": expected_effects.get("accuracy_drop", 0.01)
        }

    def _estimate_prune(self, cap: Dict[str, Any], sparsity: float) -> Dict[str, float]:
        """预估剪枝效果"""
        prune_cap = cap.get("compression", {}).get("prune", {})
        methods = prune_cap.get("methods", {})
        # 尝试找到匹配的剪枝方法配置
        # 优先查找structured_pruning或unstructured_pruning
        method_config = None
        for method_name in ["structured_pruning", "unstructured_pruning"]:
            if method_name in methods:
                method_config = methods[method_name]
                break
        
        if not method_config:
            # 如果没有找到，使用第一个可用的方法
            if methods:
                method_config = list(methods.values())[0]
            else:
                return {"size_reduction": 0.7, "speedup": 1.2, "accuracy_drop": 0.02}
        
        # 从configurable中获取sparsity范围，或使用默认值
        expected_effects = method_config.get("expected_effects", {})
        # 根据sparsity调整效果（简单线性插值）
        base_size_reduction = expected_effects.get("size_reduction", 0.7)
        base_speedup = expected_effects.get("speedup", 1.2)
        base_acc_drop = expected_effects.get("accuracy_drop", 0.02)
        
        # 根据sparsity比例调整（假设0.3是基准）
        sparsity_ratio = sparsity / 0.3 if sparsity > 0 else 1.0
        
        return {
            "size_reduction": min(0.95, base_size_reduction * sparsity_ratio),
            "speedup": base_speedup * (1 + sparsity_ratio * 0.2),
            "accuracy_drop": base_acc_drop * sparsity_ratio
        }

    def _estimate_distill(self, cap: Dict[str, Any]) -> Dict[str, float]:
        """预估蒸馏效果"""
        distill_cap = cap.get("compression", {}).get("distill", {})
        methods = distill_cap.get("methods", {})
        # 使用第一个可用的蒸馏方法
        method_config = None
        if methods:
            method_config = list(methods.values())[0]
        
        if not method_config:
            return {"size_reduction": 0.5, "speedup": 1.5, "accuracy_drop": 0.0}
        
        expected_effects = method_config.get("expected_effects", {})
        return {
            "size_reduction": expected_effects.get("size_reduction", 0.5),
            "speedup": expected_effects.get("speedup", 1.5),
            "accuracy_drop": expected_effects.get("accuracy_drop", 0.0)
        }

    def _calculate_confidence(self, cap: Dict[str, Any], applied_techs: List[str]) -> float:
        """计算预估置信度"""
        base_conf = 0.75
        if cap.get("typical_size_mb") and cap.get("typical_latency_ms"):
            base_conf += 0.1
        if len(applied_techs) == 1:
            base_conf += 0.1
        elif len(applied_techs) > 2:
            base_conf -= 0.1
        return round(max(0.5, min(0.95, base_conf)), 2)

    def _default_estimate(self, base_size: Optional[float], base_latency: Optional[float]) -> Dict[str, Any]:
        """默认预估"""
        if base_size is None:
            base_size = 100
        if base_latency is None:
            base_latency = 100
        return {
            "estimated_size_mb": round(base_size * 0.5, 2),
            "estimated_latency_ms": round(base_latency / 1.5, 1),
            "estimated_accuracy_drop": 0.02,
            "size_reduction_ratio": 0.5,
            "speedup": 1.5,
            "confidence": 0.5,
            "applied_techniques": ["未知"],
            "reference_models": [],
            "baseline": {"size_mb": base_size, "latency_ms": base_latency}
        }

