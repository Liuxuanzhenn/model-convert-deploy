"""模型能力配置

加载和查询 model_capabilities.json 配置
"""
import json
import os
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

try:
    from ..config.settings import Config
except ImportError:
    try:
        from config.settings import Config
    except ImportError:
        Config = None


class CapabilityRegistryV2:
    """模型能力注册表 V2"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            if Config:
                config_path = str(Config.MODEL_CAPABILITIES)
            else:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                config_path = os.path.join(base_dir, "configs", "model_capabilities.json")
        
        self.config_path = config_path
        self._capabilities = self._load_capabilities()
    
    def _load_capabilities(self) -> Dict[str, Any]:
        """加载能力配置文件并解析模板"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            templates = data.get("templates", {})
            models = data.get("models", {})
            
            # 解析模板引用
            resolved_models = {}
            for model_key, model_config in models.items():
                resolved_models[model_key] = self._resolve_templates(model_config, templates)
            
            return resolved_models
        except Exception as e:
            logger.error(f"Failed to load capabilities from {self.config_path}: {e}")
            return {}
    
    def _resolve_templates(self, config: Dict[str, Any], templates: Dict[str, Any]) -> Dict[str, Any]:
        """递归解析模板引用"""
        if isinstance(config, dict):
            # 检查是否有模板引用
            if "_template" in config:
                template_key = config["_template"]
                template = templates.get(template_key)
                if not template:
                    logger.warning(f"Template '{template_key}' not found, using config as-is")
                    return config
                
                # 合并模板和覆盖配置（深度合并）
                resolved = self._deep_merge(template.copy(), {k: v for k, v in config.items() if k != "_template"})
                return self._resolve_templates(resolved, templates)
            else:
                # 递归处理所有值
                return {k: self._resolve_templates(v, templates) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_templates(item, templates) for item in config]
        else:
            return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def get(self, framework: str, family: str) -> Optional[Dict[str, Any]]:
        """获取指定模型的能力信息"""
        key = f"{framework.lower()}.{family.lower()}"
        return self._capabilities.get(key)
    
    def get_file_types_mapping(self) -> Dict[str, str]:
        """获取文件类型映射"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("file_types", {})
        except Exception as e:
            logger.error(f"Failed to load file_types from {self.config_path}: {e}")
            return {}
    
    def get_supported_operations(self, framework: str, family: str) -> Dict[str, Any]:
        """获取模型支持的操作"""
        cap = self.get(framework, family)
        if not cap:
            return {}
        
        compression = cap.get("compression", {})
        operations = {}
        
        for op_type in ["quantize", "prune", "distill"]:
            op_config = compression.get(op_type, {})
            if isinstance(op_config, dict):
                operations[op_type] = {
                    "enabled": op_config.get("enabled", False),
                    "reason": op_config.get("reason"),
                    "methods": list(op_config.get("methods", {}).keys()) if op_config.get("methods") else [],
                    "recommended": op_config.get("recommended")
                }
        
        return operations
    
    def get_all_operation_requirements(self, framework: str, family: str) -> Dict[str, Any]:
        """获取所有操作的需求（需要的额外文件等）"""
        cap = self.get(framework, family)
        if not cap:
            return {}
        
        compression = cap.get("compression", {})
        requirements = {}
        
        for op_type in ["quantize", "prune", "distill"]:
            op_config = compression.get(op_type, {})
            if isinstance(op_config, dict) and op_config.get("enabled"):
                methods = op_config.get("methods", {})
                op_requirements = {}
                
                for method_name, method_config in methods.items():
                    op_requirements[method_name] = {
                        "required_extra_files": method_config.get("required_extra_files", []),
                        "optional_extra_files": method_config.get("optional_extra_files", []),
                        "configurable": method_config.get("configurable", {}),
                        "expected_effects": method_config.get("expected_effects", {})
                    }
                
                requirements[op_type] = op_requirements
        
        return requirements


# 全局单例
_registry_instance: Optional[CapabilityRegistryV2] = None


def get_registry_v2() -> CapabilityRegistryV2:
    """获取能力注册表单例"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = CapabilityRegistryV2()
    return _registry_instance

