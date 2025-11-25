"""方法映射器

将API的method参数转换为strategy格式
"""
import logging
from typing import Dict, Any, Optional, List

from services.files import ExtraFilesManager
from compression.capabilities_v2 import get_registry_v2

logger = logging.getLogger(__name__)


class MethodMapper:
    """方法映射器类"""
    
    def __init__(self):
        self.registry = get_registry_v2()
    
    def convert_to_strategy(
        self,
        method: Any,
        extra_manager: ExtraFilesManager,
        framework: str,
        family: str,
        export_formats: Optional[List[str]] = None,
        original_format: Optional[str] = None
    ) -> Dict[str, Any]:
        """将method参数转换为strategy格式
        
        Args:
            method: 压缩方法
            extra_manager: 额外文件管理器
            framework: 框架类型
            family: 模型家族
            export_formats: 用户指定的导出格式列表（可选）
            original_format: 检测到的原始格式（可选）
        """
        strategy = {}
        
        if isinstance(method, str):
            strategy = self._convert_single_method(method, extra_manager, framework, family)
        elif isinstance(method, dict):
            strategy = self._convert_composite_method(method, extra_manager, framework, family)
        else:
            raise ValueError(f"Invalid method type: {type(method)}")
        
        # 处理导出格式：自动使用原始格式
        if export_formats:
            # 用户指定了导出格式，直接使用（不再验证）
            strategy["export"] = {"formats": export_formats}
        elif "export" not in strategy:
            # 用户未指定导出格式，自动使用检测到的原始格式
            if original_format:
                strategy["export"] = {"formats": [original_format]}
            else:
                # 检测失败，根据framework推断默认格式
                default_format = self._get_format_by_framework(framework)
                strategy["export"] = {"formats": [default_format]}
                logger.warning(
                    f"Cannot detect original format, using default format '{default_format}' for framework '{framework}'"
                )
        
        return strategy
    
    def _convert_single_method(
        self,
        method: str,
        extra_manager: ExtraFilesManager,
        framework: str,
        family: str
    ) -> Dict[str, Any]:
        """转换单个方法字符串为strategy"""
        strategy = {}
        
        if method in ["fp16", "int8_dynamic", "int8_static", "int8", "qat"]:
            quantize_cfg = {"enable": True, "precision": method}
            
            if method == "int8":
                # INT8自动选择：优先静态（有校准数据），否则动态
                calib_dir = extra_manager.get_calib_dir()
                if calib_dir:
                    quantize_cfg["precision"] = "int8_static"
                    quantize_cfg["calib_dir"] = calib_dir
                    logger.info("INT8量化：检测到校准数据，使用INT8静态量化")
                else:
                    quantize_cfg["precision"] = "int8_dynamic"
                    logger.info("INT8量化：未检测到校准数据，使用INT8动态量化")
            
            elif method == "int8_static":
                calib_dir = extra_manager.get_calib_dir()
                if calib_dir:
                    quantize_cfg["calib_dir"] = calib_dir
                else:
                    logger.warning("int8_static requires calibration_data, falling back to int8_dynamic")
                    quantize_cfg["precision"] = "int8_dynamic"
            
            elif method == "qat":
                train_data_dir = extra_manager.get_train_data_dir()
                if train_data_dir:
                    quantize_cfg["train_data_dir"] = train_data_dir
                    quantize_cfg["epochs"] = 10
                else:
                    raise ValueError("qat requires train_data in extra_dir")
            
            strategy["quantize"] = quantize_cfg
        
        elif method == "auto":
            # 自动量化
            quantize_cfg = {"enable": True, "auto": True}
            strategy["quantize"] = quantize_cfg
        
        elif method == "auto_pruning":
            # 自动剪枝（必须在 endswith("_pruning") 之前检查）
            strategy["prune"] = {
                "enable": True,
                "type": "auto",
                "auto": True  # 标记为自动剪枝，便于适配器触发 select_sparsity
            }
        
        elif method.endswith("_pruning"):
            prune_type = method.replace("_pruning", "")
            if prune_type == "structured":
                strategy["prune"] = {
                    "enable": True,
                    "type": "structured",
                    "target_sparsity": 0.3
                }
            elif prune_type == "unstructured":
                strategy["prune"] = {
                    "enable": True,
                    "type": "unstructured",
                    "target_sparsity": 0.3
                }
            else:
                raise ValueError(f"Unknown pruning type: {prune_type}")
        
        elif method == "knowledge_distillation":
            teacher_dir = extra_manager.get_teacher_model_dir()
            train_data_dir = extra_manager.get_train_data_dir()
            
            if not teacher_dir:
                raise ValueError("knowledge_distillation requires teacher_model in extra_dir")
            if not train_data_dir:
                raise ValueError("knowledge_distillation requires train_data in extra_dir")
            
            strategy["distill"] = {
                "enable": True,
                "teacher_dir": teacher_dir,
                "train_data_dir": train_data_dir,
                "temperature": 4.0,
                "alpha": 0.7,
                "epochs": 20
            }
        
        elif method == "auto_distillation":
            # 自动蒸馏
            teacher_dir = extra_manager.get_teacher_model_dir()
            train_data_dir = extra_manager.get_train_data_dir()
            
            if not teacher_dir:
                raise ValueError("auto_distillation requires teacher_model in extra_dir")
            if not train_data_dir:
                raise ValueError("auto_distillation requires train_data in extra_dir")
            
            strategy["distill"] = {
                "enable": True,
                "teacher_dir": teacher_dir,
                "train_data_dir": train_data_dir,
                "temperature": 4.0,
                "alpha": 0.7,
                "epochs": 20
            }
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return strategy
    
    def _convert_composite_method(
        self,
        method: Dict[str, Any],
        extra_manager: ExtraFilesManager,
        framework: str,
        family: str
    ) -> Dict[str, Any]:
        """转换组合方法字典为strategy"""
        strategy = {}
        
        if "quantize" in method:
            quantize_method = method["quantize"]
            if isinstance(quantize_method, str):
                quantize_cfg = {"enable": True, "precision": quantize_method}
                
                if quantize_method == "int8":
                    # INT8自动选择：优先静态（有校准数据），否则动态
                    calib_dir = extra_manager.get_calib_dir()
                    if calib_dir:
                        quantize_cfg["precision"] = "int8_static"
                        quantize_cfg["calib_dir"] = calib_dir
                        logger.info("INT8量化：检测到校准数据，使用INT8静态量化")
                    else:
                        quantize_cfg["precision"] = "int8_dynamic"
                        logger.info("INT8量化：未检测到校准数据，使用INT8动态量化")
                    
                    if "quantize_config" in method:
                        config = method["quantize_config"]
                        if "calib_num" in config and calib_dir:
                            quantize_cfg["calib_num"] = config["calib_num"]
                
                elif quantize_method == "int8_static":
                    calib_dir = extra_manager.get_calib_dir()
                    if calib_dir:
                        quantize_cfg["calib_dir"] = calib_dir
                    else:
                        logger.warning("int8_static requires calibration_data, falling back to int8_dynamic")
                        quantize_cfg["precision"] = "int8_dynamic"
                    if "quantize_config" in method:
                        quantize_cfg.update(method["quantize_config"])
                
                elif quantize_method == "auto":
                    quantize_cfg = {"enable": True, "auto": True}
                    if "quantize_config" in method:
                        quantize_cfg.update(method["quantize_config"])
                
                elif quantize_method == "qat":
                    train_data_dir = extra_manager.get_train_data_dir()
                    if train_data_dir:
                        quantize_cfg["train_data_dir"] = train_data_dir
                    if "quantize_config" in method:
                        quantize_cfg.update(method["quantize_config"])
                    else:
                        quantize_cfg["epochs"] = 10
                
                strategy["quantize"] = quantize_cfg
            elif isinstance(quantize_method, dict):
                # 支持直接传入字典配置
                quantize_cfg = dict(quantize_method)
                if "enable" not in quantize_cfg:
                    quantize_cfg["enable"] = True
                
                # 处理 int8 自动选择
                if quantize_cfg.get("precision") == "int8":
                    calib_dir = extra_manager.get_calib_dir()
                    if calib_dir:
                        quantize_cfg["precision"] = "int8_static"
                        quantize_cfg["calib_dir"] = calib_dir
                        logger.info("INT8量化：检测到校准数据，使用INT8静态量化")
                    else:
                        quantize_cfg["precision"] = "int8_dynamic"
                        logger.info("INT8量化：未检测到校准数据，使用INT8动态量化")
                
                # 处理 int8_static 的 fallback
                elif quantize_cfg.get("precision") == "int8_static":
                    if "calib_dir" not in quantize_cfg:
                        calib_dir = extra_manager.get_calib_dir()
                        if calib_dir:
                            quantize_cfg["calib_dir"] = calib_dir
                        else:
                            logger.warning("int8_static requires calibration_data, falling back to int8_dynamic")
                            quantize_cfg["precision"] = "int8_dynamic"
                
                # 处理 auto
                if quantize_cfg.get("auto") or quantize_cfg.get("precision") == "auto":
                    quantize_cfg["auto"] = True
                    if "precision" in quantize_cfg and quantize_cfg["precision"] != "auto":
                        del quantize_cfg["precision"]
                
                strategy["quantize"] = quantize_cfg
        
        if "prune" in method:
            prune_method = method["prune"]
            if isinstance(prune_method, str):
                if prune_method == "auto":
                    prune_cfg = {
                        "enable": True,
                        "type": "auto",
                        "auto": True
                    }
                else:
                    prune_type = prune_method.replace("_pruning", "") if "_pruning" in prune_method else prune_method
                    prune_cfg = {
                        "enable": True,
                        "type": prune_type
                    }
                
                if "prune_config" in method:
                    prune_cfg.update(method["prune_config"])
                elif "target_sparsity" not in prune_cfg:
                    prune_cfg["target_sparsity"] = 0.3
                
                strategy["prune"] = prune_cfg
            elif isinstance(prune_method, dict):
                # 支持直接传入字典配置
                prune_cfg = dict(prune_method)
                if "enable" not in prune_cfg:
                    prune_cfg["enable"] = True
                if "type" not in prune_cfg:
                    # 如果有 method 字段，用 method；否则默认 structured
                    prune_cfg["type"] = prune_cfg.pop("method", "structured")
                if "target_sparsity" not in prune_cfg and prune_cfg.get("type") != "auto":
                    prune_cfg["target_sparsity"] = 0.3
                
                # 处理 auto
                if prune_cfg.get("type") == "auto" or prune_cfg.get("auto"):
                    prune_cfg["type"] = "auto"
                    prune_cfg["auto"] = True
                
                strategy["prune"] = prune_cfg
        
        if "distill" in method:
            distill_method = method.get("distill", "knowledge_distillation")
            
            if isinstance(distill_method, dict):
                # 支持直接传入字典配置
                distill_cfg = dict(distill_method)
                if "enable" not in distill_cfg:
                    distill_cfg["enable"] = True
                
                # 确保必需的目录参数
                if "teacher_dir" not in distill_cfg:
                    teacher_dir = extra_manager.get_teacher_model_dir()
                    if teacher_dir:
                        distill_cfg["teacher_dir"] = teacher_dir
                    else:
                        raise ValueError("distill requires teacher_model in extra_dir")
                
                if "train_data_dir" not in distill_cfg:
                    train_data_dir = extra_manager.get_train_data_dir()
                    if train_data_dir:
                        distill_cfg["train_data_dir"] = train_data_dir
                    else:
                        raise ValueError("distill requires train_data in extra_dir")
                
                # 设置默认参数
                if "temperature" not in distill_cfg:
                    distill_cfg["temperature"] = 4.0
                if "alpha" not in distill_cfg:
                    distill_cfg["alpha"] = 0.7
                if "epochs" not in distill_cfg:
                    distill_cfg["epochs"] = 20
                
                strategy["distill"] = distill_cfg
            elif isinstance(distill_method, str):
                if distill_method == "auto":
                    # 自动蒸馏
                    teacher_dir = extra_manager.get_teacher_model_dir()
                    train_data_dir = extra_manager.get_train_data_dir()
                    
                    if not teacher_dir:
                        raise ValueError("auto distill requires teacher_model in extra_dir")
                    if not train_data_dir:
                        raise ValueError("auto distill requires train_data in extra_dir")
                    
                    distill_cfg = {
                        "enable": True,
                        "teacher_dir": teacher_dir,
                        "train_data_dir": train_data_dir,
                        "temperature": 4.0,
                        "alpha": 0.7,
                        "epochs": 20
                    }
                    
                    if "distill_config" in method:
                        distill_cfg.update(method["distill_config"])
                    
                    strategy["distill"] = distill_cfg
                else:
                    # 字符串形式（兼容旧逻辑）
                    teacher_dir = extra_manager.get_teacher_model_dir()
                    train_data_dir = extra_manager.get_train_data_dir()
                    
                    if not teacher_dir:
                        raise ValueError("distill requires teacher_model in extra_dir")
                    if not train_data_dir:
                        raise ValueError("distill requires train_data in extra_dir")
                    
                    distill_cfg = {
                        "enable": True,
                        "teacher_dir": teacher_dir,
                        "train_data_dir": train_data_dir
                    }
                    
                    if "distill_config" in method:
                        distill_cfg.update(method["distill_config"])
                    else:
                        distill_cfg.update({
                            "temperature": 4.0,
                            "alpha": 0.7,
                            "epochs": 20
                        })
                    
                    strategy["distill"] = distill_cfg
            else:
                # 默认knowledge_distillation
                teacher_dir = extra_manager.get_teacher_model_dir()
                train_data_dir = extra_manager.get_train_data_dir()
                
                if not teacher_dir:
                    raise ValueError("distill requires teacher_model in extra_dir")
                if not train_data_dir:
                    raise ValueError("distill requires train_data in extra_dir")
                
                distill_cfg = {
                    "enable": True,
                    "teacher_dir": teacher_dir,
                    "train_data_dir": train_data_dir,
                    "temperature": 4.0,
                    "alpha": 0.7,
                    "epochs": 20
                }
                
                strategy["distill"] = distill_cfg
        
        return strategy
    
    def _get_format_by_framework(self, framework: str) -> str:
        """根据framework获取默认格式"""
        format_map = {
            "pytorch": "pt",
            "tensorflow": "pb",
            "paddlepaddle": "paddle_infer",
            "onnx": "onnx"
        }
        return format_map.get(framework.lower(), "pt")

