"""压缩API接口（新系统版本）

符合系统约定的API接口，使用model_dir/res_dir/extra_dir
"""
import logging
from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify

from services.model import ModelDetector, TeacherValidator
from services.files import ExtraFilesManager
from utils.path import PathManager
from utils.error import create_error_response, create_success_response, APIError, ErrorCode
from compression.capabilities_v2 import get_registry_v2
from api.method_mapper import MethodMapper
from core.engine import execute_optimize

logger = logging.getLogger(__name__)


def _get_method_key(method: str, operation_requirements: Dict[str, Any]) -> Optional[str]:
    """获取方法在operation_requirements中的key"""
    for op_type, methods in operation_requirements.items():
        if method in methods:
            return f"{op_type}.{method}"
    return None


def _get_method_requirement(method_key: str, operation_requirements: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """获取方法的requirements"""
    parts = method_key.split(".", 1)
    if len(parts) == 2:
        op_type, method_name = parts
        return operation_requirements.get(op_type, {}).get(method_name)
    return None


def _check_method_availability(operation_requirements: Dict[str, Any], extra_manager: ExtraFilesManager) -> Dict[str, Dict[str, Any]]:
    """检查方法的可用性"""
    availability = {}
    
    for op_type, methods in operation_requirements.items():
        for method_name, method_req in methods.items():
            key = f"{op_type}.{method_name}"
            required = method_req.get("required_extra_files", [])
            optional = method_req.get("optional_extra_files", [])
            
            required_check = extra_manager.check_requirements(required)
            optional_check = extra_manager.check_requirements(optional)
            
            all_required_available = all(required_check.values()) if required else True
            has_optional = any(optional_check.values()) if optional else False
            
            availability[key] = {
                "available": all_required_available,
                "required_files_status": required_check,
                "optional_files_status": optional_check,
                "has_optional": has_optional,
                "fallback": None
            }
            
            if not all_required_available:
                if op_type == "quantize" and method_name == "int8_static":
                    availability[key]["fallback"] = "int8_dynamic"
    
    return availability

compression_api_bp = Blueprint('compression_api', __name__)


@compression_api_bp.post("/detect-capabilities")
def detect_capabilities():
    """检测模型支持的压缩操作
    ---
    tags:
      - 模型压缩
    summary: 检测模型支持的压缩操作、导出格式和额外文件可用性
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - model_dir
          properties:
            model_dir:
              type: string
              description: 模型目录路径
            extra_dir:
              type: string
              description: 额外文件目录路径（可选）
    responses:
      200:
        description: 成功返回模型能力信息
        schema:
          type: object
          properties:
            code:
              type: integer
              example: 200
            message:
              type: string
              example: success
            data:
              type: object
              properties:
                framework:
                  type: string
                  example: pytorch
                family:
                  type: string
                  example: resnet
                original_format:
                  type: string
                  example: pt
                supported_operations:
                  type: object
                operation_requirements:
                  type: object
                available_files:
                  type: object
                method_availability:
                  type: object
      400:
        description: 请求参数错误
      500:
        description: 服务器内部错误
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        model_dir = data.get("model_dir")
        
        if not model_dir:
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                "model_dir is required"
            )), 400
        
        try:
            model_dir = PathManager.validate_model_dir(model_dir)
        except ValueError as e:
            return jsonify(create_error_response(
                ErrorCode.PATH_INVALID,
                str(e)
            )), 400
        
        detector = ModelDetector()
        detection = detector.detect_from_dir(model_dir)
        framework = detection["framework"]
        family = detection["family"]
        original_format = detection.get("original_format")
        
        registry = get_registry_v2()
        if not registry:
            return jsonify(create_error_response(
                ErrorCode.INTERNAL_ERROR,
                "Capability registry not available"
            )), 500
        
        supported_operations = registry.get_supported_operations(framework, family)
        operation_requirements = registry.get_all_operation_requirements(framework, family)
        
        extra_dir = data.get("extra_dir")
        available_files = {}
        method_availability = {}
        
        if extra_dir:
            try:
                extra_dir = PathManager.validate_extra_dir(extra_dir, create_if_not_exists=False)
                if extra_dir:
                    extra_manager = ExtraFilesManager(extra_dir)
                    available_files = extra_manager.list_available_files()
                    method_availability = _check_method_availability(
                        operation_requirements, extra_manager
                    )
            except ValueError:
                pass
        
        return jsonify(create_success_response({
            "framework": framework,
            "family": family,
            "original_format": original_format,
            "supported_operations": supported_operations,
            "operation_requirements": operation_requirements,
            "available_files": available_files,
            "method_availability": method_availability
        }))
        
    except APIError as e:
        return jsonify(e.to_dict()), e.code
    except Exception as e:
        logger.error(f"Error in detect_capabilities: {e}", exc_info=True)
        return jsonify(create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Internal error: {str(e)}"
        )), 500


@compression_api_bp.post("/execute")
def execute_compression():
    """执行模型压缩操作
    ---
    tags:
      - 模型压缩
    summary: 执行模型压缩操作（量化/剪枝/蒸馏）
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - model_dir
            - result_dir
            - method
          properties:
            model_dir:
              type: string
              description: 模型目录路径
            result_dir:
              type: string
              description: 结果输出目录路径
            extra_dir:
              type: string
              description: 额外文件目录路径（可选）
            method:
              type: string
              description: 压缩方法（如 "fp16", "int8_dynamic", "structured_pruning"）或字典格式的组合配置
              example: fp16
    responses:
      200:
        description: 压缩操作成功完成
        schema:
          type: object
          properties:
            code:
              type: integer
              example: 200
            message:
              type: string
              example: success
            data:
              type: object
              properties:
                job_id:
                  type: string
                  example: j_abc123
                result_dir:
                  type: string
                artifacts:
                  type: array
                  items:
                    type: string
                metrics:
                  type: object
                  properties:
                    size_before_mb:
                      type: number
                    size_after_mb:
                      type: number
                    compression_ratio:
                      type: number
      400:
        description: 请求参数错误
      500:
        description: 服务器内部错误
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        model_dir = data.get("model_dir")
        result_dir = data.get("result_dir")
        extra_dir = data.get("extra_dir")
        method = data.get("method")
        
        if not model_dir:
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                "model_dir is required"
            )), 400
        
        if not result_dir:
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                "result_dir is required"
            )), 400
        
        if not method:
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                "method is required"
            )), 400
        
        try:
            model_dir = PathManager.validate_model_dir(model_dir)
            result_dir = PathManager.validate_result_dir(result_dir, create_if_not_exists=True)
            extra_dir = PathManager.validate_extra_dir(extra_dir, create_if_not_exists=False)
        except ValueError as e:
            return jsonify(create_error_response(
                ErrorCode.PATH_INVALID,
                str(e)
            )), 400
        
        detector = ModelDetector()
        detection = detector.detect_from_dir(model_dir)
        framework = detection["framework"]
        family = detection["family"]
        original_format = detection.get("original_format")
        
        extra_manager = ExtraFilesManager(extra_dir)
        
        registry = get_registry_v2()
        operation_requirements = registry.get_all_operation_requirements(framework, family)
        warnings = []
        
        if isinstance(method, str):
            method_key = _get_method_key(method, operation_requirements)
            if method_key:
                method_req = _get_method_requirement(method_key, operation_requirements)
                if method_req:
                    required = method_req.get("required_extra_files", [])
                    required_check = extra_manager.check_requirements(required)
                    missing_required = [k for k, v in required_check.items() if not v]
                    if missing_required:
                        warnings.append(f"Method '{method}' requires missing files: {', '.join(missing_required)}")
        
        mapper = MethodMapper()
        export_formats = data.get("export_formats")
        strategy = mapper.convert_to_strategy(
            method, extra_manager, framework, family, 
            export_formats=export_formats,
            original_format=original_format
        )
        
        if warnings:
            logger.warning(f"Warnings for method '{method}': {', '.join(warnings)}")
        
        if strategy.get("distill", {}).get("enable", False):
            teacher_dir = extra_manager.get_teacher_model_dir()
            if teacher_dir:
                validator = TeacherValidator()
                validation_result = validator.validate(
                    student_model_dir=model_dir,
                    teacher_model_dir=teacher_dir,
                    student_framework=framework,
                    student_family=family
                )
                if not validation_result["valid"]:
                    return jsonify(create_error_response(
                        ErrorCode.TEACHER_MODEL_INVALID,
                        validation_result["reason"],
                        validation_result
                    )), 400
        
        optimize_data = {
            "framework": framework,
            "family": family,
            "model_dir": model_dir,
            "res_dir": result_dir,
            "strategy": strategy
        }
        
        result = execute_optimize(optimize_data)
        
        if "error" in result:
            return jsonify(create_error_response(
                ErrorCode.COMPRESSION_FAILED,
                result["error"]
            )), 500
        
        return jsonify(create_success_response({
            "job_id": result.get("job_id"),
            "result_dir": result_dir,
            "artifacts": result.get("artifacts", []),
            "metrics": result.get("metrics", {})
        }))
        
    except APIError as e:
        return jsonify(e.to_dict()), e.code
    except ValueError as e:
        return jsonify(create_error_response(
            ErrorCode.INVALID_METHOD,
            str(e)
        )), 400
    except Exception as e:
        logger.error(f"Error in execute_compression: {e}", exc_info=True)
        return jsonify(create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Internal error: {str(e)}"
        )), 500

