"""压缩配置验证器"""
import os
import glob
from typing import Dict, Any, Optional

from compression.capabilities_v2 import get_registry_v2


class ConfigValidator:
    """压缩配置验证器"""

    def __init__(self):
        self.registry = get_registry_v2()

    def validate(self, framework: str, family: str, strategy_config: Dict[str, Any],
                 model_dir: Optional[str] = None) -> Dict[str, Any]:
        """验证压缩配置"""
        result = {"valid": True, "errors": [], "warnings": [], "suggestions": []}

        if not self.registry:
            result["valid"] = False
            result["errors"].append("Capability registry not available")
            return result

        cap = self.registry.get(framework, family)
        if not cap:
            result["valid"] = False
            result["errors"].append(f"未找到模型能力定义: {framework}.{family}")
            return result

        if strategy_config.get("quantize", {}).get("enable"):
            self._validate_quantize(cap, strategy_config["quantize"], model_dir, result)

        if strategy_config.get("prune", {}).get("enable"):
            self._validate_prune(cap, strategy_config["prune"], model_dir, result)

        if strategy_config.get("distill", {}).get("enable"):
            self._validate_distill(cap, strategy_config["distill"], model_dir, result)

        self._validate_combination(cap, strategy_config, result)

        return result

    def _validate_quantize(self, cap: Dict[str, Any], config: Dict[str, Any],
                          model_dir: Optional[str], result: Dict[str, Any]) -> None:
        """验证量化配置"""
        quant_cap = cap.get("compression", {}).get("quantize", {})
        if not quant_cap.get("enabled"):
            result["valid"] = False
            result["errors"].append("该模型不支持量化")
            return

        precision = config.get("precision")
        if not precision:
            result["valid"] = False
            result["errors"].append("量化配置缺少 precision 字段")
            return

        methods = quant_cap.get("methods", {})
        if precision not in methods:
            result["valid"] = False
            supported_methods = ", ".join(methods.keys())
            result["errors"].append(f"不支持的量化精度: {precision}，支持的精度: {supported_methods}")
            return

        method_config = methods[precision]
        required_files = method_config.get("required_extra_files", [])
        optional_files = method_config.get("optional_extra_files", [])
        
        if required_files:
            result["warnings"].append(f"量化方法 {precision} 需要额外文件: {', '.join(required_files)}")

        recommended = quant_cap.get("recommended")
        if precision != recommended:
            result["suggestions"].append(f"推荐使用 {recommended} 精度，通常在该模型上效果更好")

        if precision == "int8_static":
            if not config.get("calib_dir") and not config.get("calib_num"):
                result["warnings"].append("INT8 静态量化未提供校准数据，可能影响精度。建议提供 calib_dir 或 calib_num")

    def _validate_prune(self, cap: Dict[str, Any], config: Dict[str, Any],
                       model_dir: Optional[str], result: Dict[str, Any]) -> None:
        """验证剪枝配置"""
        prune_cap = cap.get("compression", {}).get("prune", {})
        if not prune_cap.get("enabled"):
            result["valid"] = False
            result["errors"].append("该模型不支持剪枝")
            return

        prune_type = config.get("type")
        methods = prune_cap.get("methods", {})
        
        # 验证剪枝类型
        if prune_type:
            # 检查是否有匹配的方法
            method_found = False
            for method_name in methods.keys():
                if (prune_type == "structured" and "structured" in method_name) or \
                   (prune_type == "unstructured" and "unstructured" in method_name):
                    method_found = True
                    break
            
            if not method_found:
                supported_types = []
                for method_name in methods.keys():
                    if "structured" in method_name:
                        supported_types.append("structured")
                    elif "unstructured" in method_name:
                        supported_types.append("unstructured")
                result["warnings"].append(f"剪枝类型 {prune_type} 可能不完全匹配，支持的类型: {', '.join(set(supported_types))}")

        target_sparsity = config.get("target_sparsity")
        if target_sparsity is not None:
            max_sparsity = 0.8  # 默认最大值
            if target_sparsity > max_sparsity:
                result["warnings"].append(f"目标稀疏度 {target_sparsity} 超过推荐最大值 {max_sparsity}，可能导致精度大幅下降")
            if target_sparsity < 0.1:
                result["warnings"].append(f"目标稀疏度 {target_sparsity} 过低，压缩效果可能不明显")

        # 检查文件要求
        if methods:
            method_config = list(methods.values())[0]
            required_files = method_config.get("required_extra_files", [])
            if required_files:
                result["warnings"].append(f"剪枝需要额外文件: {', '.join(required_files)}")

    def _validate_distill(self, cap: Dict[str, Any], config: Dict[str, Any],
                         model_dir: Optional[str], result: Dict[str, Any]) -> None:
        """验证蒸馏配置"""
        distill_cap = cap.get("compression", {}).get("distill", {})
        if not distill_cap.get("enabled"):
            result["valid"] = False
            result["errors"].append("该模型不支持知识蒸馏")
            return

        teacher_dir = config.get("teacher_dir")
        if not teacher_dir:
            result["valid"] = False
            result["errors"].append("知识蒸馏需要提供 teacher_dir（教师模型目录）")
            return

        if model_dir and not os.path.isdir(teacher_dir):
            result["warnings"].append(f"教师模型目录不存在: {teacher_dir}")

        temperature = config.get("temperature", 4.0)
        if temperature < 1.0 or temperature > 20.0:
            result["warnings"].append(f"温度参数 {temperature} 超出常用范围 [1.0, 20.0]，建议使用 2.0-8.0")

        alpha = config.get("alpha", 0.5)
        if alpha < 0 or alpha > 1:
            result["valid"] = False
            result["errors"].append(f"alpha 参数 {alpha} 必须在 [0, 1] 范围内")

        methods = distill_cap.get("methods", {})
        if methods:
            method_config = list(methods.values())[0]
            required_files = method_config.get("required_extra_files", [])
            if required_files:
                result["warnings"].append(f"知识蒸馏需要额外文件: {', '.join(required_files)}")

    def _validate_combination(self, cap: Dict[str, Any], strategy_config: Dict[str, Any],
                             result: Dict[str, Any]) -> None:
        """验证技术组合的合理性"""
        enabled_techs = []
        if strategy_config.get("quantize", {}).get("enable"):
            enabled_techs.append("quantize")
        if strategy_config.get("prune", {}).get("enable"):
            enabled_techs.append("prune")
        if strategy_config.get("distill", {}).get("enable"):
            enabled_techs.append("distill")

        if len(enabled_techs) == 0:
            result["warnings"].append("未启用任何压缩技术，将仅执行模型导出")

        if "distill" in enabled_techs and "prune" in enabled_techs:
            result["warnings"].append("同时使用蒸馏和剪枝可能导致训练不稳定，建议分步进行：先剪枝后蒸馏")

        if "quantize" in enabled_techs and "prune" in enabled_techs:
            quant_precision = strategy_config.get("quantize", {}).get("precision")
            if quant_precision == "int8_static":
                result["suggestions"].append("量化+剪枝组合时，建议先剪枝再量化，以获得更好的精度")

    def _check_file_requirements(self, requirements: Dict[str, Any], model_dir: Optional[str],
                                 tech_name: str, result: Dict[str, Any]) -> None:
        """检查文件要求是否满足"""
        if not model_dir or not os.path.isdir(model_dir):
            return

        required_files = requirements.get("required_files", [])
        for pattern in required_files:
            if pattern in ["teacher_dir", "train_data", "val_data", "calib_dir"]:
                continue
            matches = glob.glob(os.path.join(model_dir, pattern))
            if not matches:
                result["warnings"].append(f"{tech_name}需要的文件未找到: {pattern}")

        optional_files = requirements.get("optional_files", {})
        if optional_files:
            missing_optional = []
            for key, desc in optional_files.items():
                if key not in ["teacher_dir", "train_data", "val_data", "calib_dir"]:
                    continue
                missing_optional.append(f"{key} ({desc})")
            if missing_optional:
                result["suggestions"].append(f"{tech_name}可选文件: {', '.join(missing_optional)}")

