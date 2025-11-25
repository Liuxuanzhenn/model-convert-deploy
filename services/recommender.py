"""压缩策略推荐引擎"""
from typing import Dict, Any, List, Optional, Tuple

from compression.capabilities_v2 import get_registry_v2


class CompressionRecommender:
    """压缩策略推荐引擎"""

    def __init__(self):
        self.registry = get_registry_v2()

    def recommend(self, model_info: Dict[str, Any], constraints: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """推荐压缩策略"""
        if not self.registry:
            return []
        
        framework = model_info.get("framework", "pytorch")
        family = model_info.get("family", "generic")
        constraints = constraints or {}

        cap = self.registry.get(framework, family)
        if not cap:
            return []

        base_size = model_info.get("model_size_mb") or cap.get("typical_size_mb", 100)
        base_latency = model_info.get("current_latency_ms") or cap.get("typical_latency_ms", 100)

        candidates = self._generate_candidates(cap, base_size, base_latency)
        scored = self._score_candidates(candidates, constraints, base_size, base_latency)

        return scored[:3]

    def _generate_candidates(self, cap: Dict[str, Any], base_size: float, base_latency: float) -> List[Dict[str, Any]]:
        """生成候选策略"""
        candidates = []
        compression = cap.get("compression", {})
        
        quant = compression.get("quantize", {})
        if quant.get("enabled"):
            methods = quant.get("methods", {})
            recommended = quant.get("recommended", "fp16")
            for method_name, method_config in methods.items():
                expected_effects = method_config.get("expected_effects", {})
                candidates.append({
                    "name": f"量化 ({method_name})",
                    "strategy": {"quantize": {"enable": True, "precision": method_name}},
                    "estimated_size_mb": base_size * expected_effects.get("size_reduction", 0.5),
                    "estimated_latency_ms": base_latency / expected_effects.get("speedup", 1.5),
                    "estimated_accuracy_drop": expected_effects.get("accuracy_drop", 0.01),
                    "complexity": "low",
                    "requirements": {
                        "required_files": method_config.get("required_extra_files", []),
                        "optional_files": method_config.get("optional_extra_files", [])
                    }
                })
        
        prune = compression.get("prune", {})
        if prune.get("enabled"):
            methods = prune.get("methods", {})
            recommended_method = None
            for method_name in ["structured_pruning", "unstructured_pruning"]:
                if method_name in methods:
                    recommended_method = method_name
                    break
            if not recommended_method and methods:
                recommended_method = list(methods.keys())[0]
            
            if recommended_method:
                method_config = methods[recommended_method]
                expected_effects = method_config.get("expected_effects", {})
                for sparsity in [0.3, 0.5]:
                    prune_type = "structured" if "structured" in recommended_method else "unstructured"
                    candidates.append({
                        "name": f"剪枝 (稀疏度{sparsity})",
                        "strategy": {"prune": {"enable": True, "type": prune_type, "target_sparsity": sparsity}},
                        "estimated_size_mb": base_size * expected_effects.get("size_reduction", 0.7) * sparsity / 0.3,
                        "estimated_latency_ms": base_latency / expected_effects.get("speedup", 1.2),
                        "estimated_accuracy_drop": expected_effects.get("accuracy_drop", 0.02) * sparsity / 0.3,
                        "complexity": "medium",
                        "requirements": {
                            "required_files": method_config.get("required_extra_files", []),
                            "optional_files": method_config.get("optional_extra_files", [])
                        }
                    })
        
        distill = compression.get("distill", {})
        if distill.get("enabled"):
            methods = distill.get("methods", {})
            if methods:
                method_config = list(methods.values())[0]
                expected_effects = method_config.get("expected_effects", {})
                candidates.append({
                    "name": "知识蒸馏",
                    "strategy": {"distill": {"enable": True, "temperature": 4.0, "alpha": 0.5}},
                    "estimated_size_mb": base_size * expected_effects.get("size_reduction", 0.5),
                    "estimated_latency_ms": base_latency / expected_effects.get("speedup", 1.5),
                    "estimated_accuracy_drop": expected_effects.get("accuracy_drop", 0.0),
                    "complexity": "high",
                    "requirements": {
                        "required_files": method_config.get("required_extra_files", []),
                        "optional_files": method_config.get("optional_extra_files", [])
                    }
                })
        
        if quant.get("enabled") and prune.get("enabled"):
            q_method = quant.get("recommended", "fp16")
            q_config = quant.get("methods", {}).get(q_method, {})
            q_effects = q_config.get("expected_effects", {})
            
            prune_methods = prune.get("methods", {})
            p_config = None
            for method_name in ["structured_pruning", "unstructured_pruning"]:
                if method_name in prune_methods:
                    p_config = prune_methods[method_name]
                    break
            if not p_config and prune_methods:
                p_config = list(prune_methods.values())[0]
            
            if p_config:
                p_effects = p_config.get("expected_effects", {})
                size_red = q_effects.get("size_reduction", 0.5) * p_effects.get("size_reduction", 0.7) * 1.1
                speedup = (q_effects.get("speedup", 1.5) + p_effects.get("speedup", 1.2)) * 0.7
                acc_drop = q_effects.get("accuracy_drop", 0.01) + p_effects.get("accuracy_drop", 0.02) * 1.2
                prune_type = "structured" if "structured" in (p_config.get("display_name", "") or "") else "unstructured"
                candidates.append({
                    "name": f"量化+剪枝 ({q_method}+0.3稀疏)",
                    "strategy": {
                        "quantize": {"enable": True, "precision": q_method},
                        "prune": {"enable": True, "type": prune_type, "target_sparsity": 0.3}
                    },
                    "estimated_size_mb": base_size * size_red,
                    "estimated_latency_ms": base_latency / speedup,
                    "estimated_accuracy_drop": acc_drop,
                    "complexity": "medium",
                    "requirements": {
                        "required_files": q_config.get("required_extra_files", []),
                        "optional_files": {}
                    }
                })
        
        return candidates

    def _score_candidates(self, candidates: List[Dict[str, Any]], constraints: Dict[str, Any],
                         base_size: float, base_latency: float) -> List[Dict[str, Any]]:
        """根据约束对候选策略评分并排序"""
        scored = []
        for candidate in candidates:
            score, reasons = self._score_single_candidate(candidate, constraints, base_size, base_latency)
            candidate["score"] = score
            candidate["reasons"] = reasons
            candidate["confidence"] = self._calculate_confidence(candidate)
            scored.append(candidate)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored
    
    def _score_single_candidate(self, candidate: Dict[str, Any], constraints: Dict[str, Any],
                                base_size: float, base_latency: float) -> Tuple[float, List[str]]:
        """评分单个候选策略"""
        score = 0.0
        reasons = []
        est_size = candidate["estimated_size_mb"]
        est_latency = candidate["estimated_latency_ms"]
        est_acc_drop = candidate["estimated_accuracy_drop"]
        
        target_size = constraints.get("target_size_mb")
        if target_size:
            if est_size <= target_size:
                score += 30 * (1 - est_size / target_size)
                reasons.append(f"满足大小要求({est_size:.1f}MB ≤ {target_size}MB)")
            else:
                score -= 50
                reasons.append(f"超出大小限制({est_size:.1f}MB > {target_size}MB)")
        else:
            score += ((base_size - est_size) / base_size) * 20
            reasons.append(f"压缩率{(base_size - est_size) / base_size * 100:.1f}%")
        
        max_latency = constraints.get("max_latency_ms")
        if max_latency:
            if est_latency <= max_latency:
                score += 30 * (1 - est_latency / max_latency)
                reasons.append(f"满足延迟要求({est_latency:.1f}ms ≤ {max_latency}ms)")
            else:
                score -= 50
                reasons.append(f"超出延迟限制({est_latency:.1f}ms > {max_latency}ms)")
        else:
            speedup = base_latency / est_latency
            score += (speedup - 1) * 15
            reasons.append(f"加速{speedup:.1f}x")
        
        min_accuracy = constraints.get("min_accuracy")
        if min_accuracy:
            if est_acc_drop <= (1 - min_accuracy):
                score += 25 * (1 - est_acc_drop / (1 - min_accuracy))
                reasons.append(f"精度下降{est_acc_drop*100:.2f}%可接受")
            else:
                score -= 100
                reasons.append(f"精度下降过大({est_acc_drop*100:.2f}%)")
        else:
            score += (0.05 - est_acc_drop) * 100
            reasons.append(f"精度下降{est_acc_drop*100:.2f}%")
        
        score += {"low": 0, "medium": -5, "high": -10}.get(candidate["complexity"], 0)
        
        hardware = constraints.get("hardware", "cpu")
        if hardware == "cpu" and "int8" in candidate["name"].lower():
            score += 10
            reasons.append("CPU上INT8加速明显")
        elif hardware == "gpu" and "fp16" in candidate["name"].lower():
            score += 10
            reasons.append("GPU上FP16加速明显")
        
        return score, reasons

    def _calculate_confidence(self, candidate: Dict[str, Any]) -> float:
        """计算推荐置信度"""
        complexity_conf = {"low": 0.9, "medium": 0.75, "high": 0.6}
        base_conf = complexity_conf.get(candidate["complexity"], 0.7)
        if "+" in candidate["name"]:
            base_conf *= 0.85
        return round(base_conf, 2)

