"""错误处理

统一错误码和错误处理机制
"""
from enum import IntEnum
from typing import Dict, Any, Optional


class ErrorCode(IntEnum):
    """错误码枚举"""
    SUCCESS = 200
    
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    CONFLICT = 409
    VALIDATION_ERROR = 422
    
    INTERNAL_ERROR = 500
    SERVICE_UNAVAILABLE = 503
    
    MODEL_NOT_FOUND = 1001
    MODEL_LOAD_FAILED = 1002
    ADAPTER_NOT_FOUND = 1003
    CAPABILITY_NOT_FOUND = 1004
    INVALID_METHOD = 1005
    MISSING_REQUIRED_FILES = 1006
    TEACHER_MODEL_INVALID = 1007
    COMPRESSION_FAILED = 1008
    EXPORT_FAILED = 1009
    PATH_INVALID = 1010


class ErrorMessage:
    """错误消息映射"""
    MESSAGES: Dict[int, str] = {
        ErrorCode.SUCCESS: "success",
        ErrorCode.BAD_REQUEST: "Bad request",
        ErrorCode.UNAUTHORIZED: "Unauthorized",
        ErrorCode.FORBIDDEN: "Forbidden",
        ErrorCode.NOT_FOUND: "Not found",
        ErrorCode.METHOD_NOT_ALLOWED: "Method not allowed",
        ErrorCode.CONFLICT: "Conflict",
        ErrorCode.VALIDATION_ERROR: "Validation error",
        ErrorCode.INTERNAL_ERROR: "Internal server error",
        ErrorCode.SERVICE_UNAVAILABLE: "Service unavailable",
        ErrorCode.MODEL_NOT_FOUND: "Model not found",
        ErrorCode.MODEL_LOAD_FAILED: "Model load failed",
        ErrorCode.ADAPTER_NOT_FOUND: "Adapter not found",
        ErrorCode.CAPABILITY_NOT_FOUND: "Capability not found",
        ErrorCode.INVALID_METHOD: "Invalid compression method",
        ErrorCode.MISSING_REQUIRED_FILES: "Missing required files",
        ErrorCode.TEACHER_MODEL_INVALID: "Teacher model validation failed",
        ErrorCode.COMPRESSION_FAILED: "Compression failed",
        ErrorCode.EXPORT_FAILED: "Export failed",
        ErrorCode.PATH_INVALID: "Invalid path",
    }
    
    @classmethod
    def get_message(cls, code: int, custom_message: Optional[str] = None) -> str:
        """获取错误消息"""
        if custom_message:
            return custom_message
        return cls.MESSAGES.get(code, "Unknown error")


class APIError(Exception):
    """API错误异常类"""
    
    def __init__(self, code: int, message: Optional[str] = None, data: Optional[Any] = None):
        self.code = code
        self.message = ErrorMessage.get_message(code, message)
        self.data = data
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {"code": self.code, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result


def create_error_response(code: int, message: Optional[str] = None, data: Optional[Any] = None) -> Dict[str, Any]:
    """创建错误响应"""
    return {
        "code": code,
        "message": ErrorMessage.get_message(code, message),
        "data": data
    }


def create_success_response(data: Any = None, message: str = "success") -> Dict[str, Any]:
    """创建成功响应"""
    return {
        "code": ErrorCode.SUCCESS,
        "message": message,
        "data": data
    }

