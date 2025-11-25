"""API Schema定义

使用Pydantic定义请求和响应格式
"""
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator


class DetectCapabilitiesRequest(BaseModel):
    """接口1请求格式"""
    model_dir: str = Field(..., description="模型目录路径（系统提供的路径）")
    
    class Config:
        schema_extra = {
            "example": {
                "model_dir": "/nfs/dubhe-prod/train-manage/1/job-1099-hjs6q/model"
            }
        }


class OperationRequirement(BaseModel):
    """操作需求"""
    required_files: List[str] = Field(default_factory=list, description="必需的文件类型列表")
    optional_files: List[str] = Field(default_factory=list, description="可选的文件类型列表")
    configurable: Dict[str, Any] = Field(default_factory=dict, description="可配置参数")


class DetectCapabilitiesResponse(BaseModel):
    """接口1响应格式"""
    code: int = Field(..., description="状态码")
    message: str = Field(..., description="消息")
    data: Optional[Dict[str, Any]] = Field(None, description="数据")
    
    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "message": "success",
                "data": {
                    "framework": "pytorch",
                    "family": "yolo",
                    "supported_operations": ["fp16", "int8_dynamic", "structured_pruning"],
                    "operation_requirements": {
                        "fp16": {
                            "required_files": [],
                            "optional_files": []
                        },
                        "structured_pruning": {
                            "required_files": [],
                            "optional_files": ["val_data"],
                            "configurable": {
                                "target_sparsity": {
                                    "type": "float",
                                    "default": 0.3,
                                    "min": 0.1,
                                    "max": 0.7
                                }
                            }
                        }
                    }
                }
            }
        }


class ExecuteCompressionRequest(BaseModel):
    """接口2请求格式"""
    model_dir: str = Field(..., description="模型目录路径")
    result_dir: str = Field(..., description="结果目录路径")
    extra_dir: Optional[str] = Field(None, description="额外文件目录路径（可选）")
    method: Union[str, Dict[str, Any]] = Field(..., description="压缩方法，可以是字符串或对象")
    export_formats: Optional[List[str]] = Field(None, description="导出格式列表（可选）")
    
    @validator("method")
    def validate_method(cls, v):
        """验证method参数"""
        if isinstance(v, str):
            valid_methods = [
                "fp16", "int8_dynamic", "int8_static", "int8", "qat",
                "auto",
                "structured_pruning", "unstructured_pruning", "auto_pruning",
                "knowledge_distillation", "auto_distillation"
            ]
            if v not in valid_methods:
                raise ValueError(f"Invalid method: {v}. Must be one of {valid_methods}")
        elif isinstance(v, dict):
            valid_keys = ["quantize", "prune", "distill", "export"]
            for key in v.keys():
                if key not in valid_keys:
                    raise ValueError(f"Invalid key in method: {key}. Must be one of {valid_keys}")
        else:
            raise ValueError(f"method must be str or dict, got {type(v)}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "model_dir": "/nfs/.../model",
                "result_dir": "/nfs/.../result",
                "extra_dir": "/nfs/.../extra",
                "method": "fp16",
                "export_formats": ["pt", "onnx"]
            }
        }


class ExecuteCompressionResponse(BaseModel):
    """接口2响应格式"""
    code: int = Field(..., description="状态码")
    message: str = Field(..., description="消息")
    data: Optional[Dict[str, Any]] = Field(None, description="数据")
    
    class Config:
        schema_extra = {
            "example": {
                "code": 200,
                "message": "success",
                "data": {
                    "job_id": "j_xxx",
                    "result_dir": "/nfs/.../result",
                    "artifacts": ["model_fp16.pt", "model_fp16.onnx"],
                    "metrics": {
                        "size_before_mb": 6.2,
                        "size_after_mb": 3.1,
                        "compression_ratio": 0.5
                    }
                }
            }
        }


class ErrorResponse(BaseModel):
    """错误响应格式"""
    code: int = Field(..., description="错误码")
    message: str = Field(..., description="错误消息")
    data: Optional[Any] = Field(None, description="错误详情")
    
    class Config:
        schema_extra = {
            "example": {
                "code": 400,
                "message": "Invalid request",
                "data": {
                    "field": "model_dir",
                    "error": "model_dir is required"
                }
            }
        }

