"""硬件编译API接口

提供模型硬件编译功能，支持TensorRT、Ascend、Cambricon等硬件平台
"""
import logging
import os
from typing import Dict, Any, Optional
from flask import Blueprint, request, jsonify

from utils.path import PathManager
from utils.error import create_error_response, create_success_response, APIError, ErrorCode
from compilers.registry import get_compiler, list_available_compilers, list_supported_hardware
from core.engine import execute_compile

logger = logging.getLogger(__name__)

compile_api_bp = Blueprint('compile_api', __name__)


@compile_api_bp.post("/compile")
def compile_model():
    """硬件编译接口
    ---
    tags:
      - 硬件编译
    summary: 将模型编译为硬件专用格式（TensorRT/Ascend/Cambricon/M9）
    parameters:
      - in: body
        name: body
        required: true
        schema:
          type: object
          required:
            - model_path
            - result_dir
            - target
          properties:
            model_path:
              type: string
              description: 模型文件路径（支持.pt/.pth/.onnx等）
            result_dir:
              type: string
              description: 结果输出目录路径
            target:
              type: string
              enum: [tensorrt, ascend, cambricon, m9]
              description: 目标硬件平台
            options:
              type: object
              description: 硬件配置选项（可选）
              properties:
                device:
                  type: string
                  description: Ascend专用：设备型号（如 "Ascend 310"）
                optimization:
                  type: object
                  properties:
                    fp16:
                      type: boolean
                      description: TensorRT/Ascend：是否使用FP16
                    int8:
                      type: boolean
                      description: TensorRT：是否使用INT8
                    workspace_size:
                      type: integer
                      description: TensorRT：工作空间大小（MB）
                input_shape:
                  type: string
                  description: 输入形状（如 "1,3,224,224"）
                input_format:
                  type: string
                  enum: [NCHW, NHWC]
                  description: Ascend：输入格式
    responses:
      200:
        description: 硬件编译成功
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
                result_dir:
                  type: string
                compiled_files:
                  type: array
                  items:
                    type: string
                target:
                  type: string
                input_format:
                  type: string
                output_format:
                  type: string
      400:
        description: 请求参数错误
      500:
        description: 服务器内部错误
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        model_path = data.get("model_path")
        result_dir = data.get("result_dir")
        target = data.get("target")
        options = data.get("options", {}) or {}
        
        # 参数验证
        if not model_path:
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                "model_path is required"
            )), 400
        
        if not result_dir:
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                "result_dir is required"
            )), 400
        
        if not target:
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                "target is required (tensorrt|ascend|cambricon|m9)"
            )), 400
        
        # 验证路径
        try:
            if not os.path.exists(model_path):
                return jsonify(create_error_response(
                    ErrorCode.MODEL_NOT_FOUND,
                    f"Model file not found: {model_path}"
                )), 400
            
            model_path = PathManager.normalize_path(model_path)
            result_dir = PathManager.validate_result_dir(result_dir, create_if_not_exists=True)
        except ValueError as e:
            return jsonify(create_error_response(
                ErrorCode.PATH_INVALID,
                str(e)
            )), 400
        
        # 验证target
        target_lower = str(target).lower()
        supported_targets = ["tensorrt", "ascend", "cambricon", "mlu", "m9"]
        if target_lower not in supported_targets:
            return jsonify(create_error_response(
                ErrorCode.BAD_REQUEST,
                f"Unsupported target: {target}. Supported: {supported_targets}"
            )), 400
        
        # 检查编译器是否可用
        available_compilers = list_available_compilers()
        compiler_key = target_lower
        if compiler_key == "mlu":
            compiler_key = "cambricon"
        
        if not available_compilers.get(compiler_key, False):
            return jsonify(create_error_response(
                ErrorCode.SERVICE_UNAVAILABLE,
                f"{target} compiler is not available. Please install the required dependencies."
            )), 503
        
        # 执行编译
        compile_data = {
            "artifact_path": model_path,
            "target": target_lower,
            "options": options
        }
        
        result = execute_compile(compile_data)
        
        if "error" in result:
            return jsonify(create_error_response(
                ErrorCode.COMPRESSION_FAILED,
                result["error"]
            )), 500
        
        # 确定输出格式
        output_format_map = {
            "tensorrt": "engine",
            "ascend": "om",
            "cambricon": "cambricon",
            "mlu": "cambricon",
            "m9": "m9"
        }
        output_format = output_format_map.get(target_lower, "unknown")
        
        # 生成job_id
        import time
        job_id = f"compile_{int(time.time())}"
        
        return jsonify(create_success_response({
            "job_id": job_id,
            "result_dir": result_dir,
            "compiled_files": result.get("compiled", []),
            "target": target_lower,
            "input_format": os.path.splitext(model_path)[1][1:],  # 从扩展名推断
            "output_format": output_format,
            "message": result.get("message", "Compilation completed successfully")
        }))
        
    except APIError as e:
        return jsonify(e.to_dict()), e.code
    except Exception as e:
        logger.error(f"Error in compile_model: {e}", exc_info=True)
        return jsonify(create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Internal error: {str(e)}"
        )), 500


@compile_api_bp.get("/list-hardware")
def list_hardware():
    """列出支持的硬件平台
    ---
    tags:
      - 硬件编译
    summary: 列出所有支持的硬件编译平台及其可用性
    responses:
      200:
        description: 成功返回硬件平台列表
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
              type: array
              items:
                type: object
                properties:
                  name:
                    type: string
                    example: tensorrt
                  display_name:
                    type: string
                    example: NVIDIA GPU / TensorRT
                  description:
                    type: string
                  devices:
                    type: array
                    items:
                      type: string
                  output_format:
                    type: string
                    example: engine
                  available:
                    type: boolean
                  input_formats:
                    type: array
                    items:
                      type: string
      500:
        description: 服务器内部错误
    """
    try:
        hardware_list = list_supported_hardware()
        
        # 添加输入格式信息
        for hw in hardware_list:
            hw["input_formats"] = ["onnx", "pt", "pth"]  # 所有硬件编译器都支持这些格式
        
        return jsonify(create_success_response(hardware_list))
        
    except Exception as e:
        logger.error(f"Error in list_hardware: {e}", exc_info=True)
        return jsonify(create_error_response(
            ErrorCode.INTERNAL_ERROR,
            f"Internal error: {str(e)}"
        )), 500

