"""自定义异常类

提供语义化的异常处理，便于错误追踪和用户反馈
"""


class AppException(Exception):
    """应用基础异常"""

    def __init__(self, message: str, code: str = "APP_ERROR", details: dict = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self):
        """转换为字典格式（用于 API 响应）"""
        return {
            "error": self.message,
            "code": self.code,
            "details": self.details
        }


class ModelException(AppException):
    """模型相关异常基类"""
    pass


class ModelNotFoundError(ModelException):
    """模型未找到"""

    def __init__(self, model_id: str, version_id: str = None):
        msg = f"模型未找到: {model_id}"
        if version_id:
            msg += f" (版本: {version_id})"
        super().__init__(msg, "MODEL_NOT_FOUND", {"model_id": model_id, "version_id": version_id})


class ModelLoadError(ModelException):
    """模型加载失败"""

    def __init__(self, model_path: str, reason: str = None):
        msg = f"模型加载失败: {model_path}"
        if reason:
            msg += f" - {reason}"
        super().__init__(msg, "MODEL_LOAD_ERROR", {"model_path": model_path, "reason": reason})


class AdapterNotFoundError(ModelException):
    """适配器未找到"""

    def __init__(self, framework: str, family: str):
        msg = f"未找到适配器: {framework}.{family}"
        super().__init__(msg, "ADAPTER_NOT_FOUND", {"framework": framework, "family": family})


class CompressionException(AppException):
    """压缩相关异常基类"""
    pass


class UnsupportedCompressionError(CompressionException):
    """不支持的压缩技术"""

    def __init__(self, framework: str, family: str, compression_type: str):
        msg = f"模型 {framework}.{family} 不支持 {compression_type} 压缩"
        super().__init__(msg, "UNSUPPORTED_COMPRESSION",
                        {"framework": framework, "family": family, "compression_type": compression_type})


class InvalidConfigError(CompressionException):
    """无效的压缩配置"""

    def __init__(self, errors: list):
        msg = f"压缩配置无效: {'; '.join(errors)}"
        super().__init__(msg, "INVALID_CONFIG", {"errors": errors})


class CompressionFailedError(CompressionException):
    """压缩执行失败"""

    def __init__(self, stage: str, reason: str = None):
        msg = f"压缩失败 ({stage})"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, "COMPRESSION_FAILED", {"stage": stage, "reason": reason})


class FileException(AppException):
    """文件相关异常基类"""
    pass


class FileNotFoundError(FileException):
    """文件未找到"""

    def __init__(self, file_path: str):
        msg = f"文件未找到: {file_path}"
        super().__init__(msg, "FILE_NOT_FOUND", {"file_path": file_path})


class InvalidFileError(FileException):
    """无效的文件"""

    def __init__(self, file_path: str, reason: str):
        msg = f"无效的文件: {file_path} - {reason}"
        super().__init__(msg, "INVALID_FILE", {"file_path": file_path, "reason": reason})


class FileTooLargeError(FileException):
    """文件过大"""

    def __init__(self, file_size: int, max_size: int):
        msg = f"文件过大: {file_size} bytes (最大: {max_size} bytes)"
        super().__init__(msg, "FILE_TOO_LARGE", {"file_size": file_size, "max_size": max_size})


class ConfigException(AppException):
    """配置相关异常基类"""
    pass


class MissingConfigError(ConfigException):
    """缺少配置"""

    def __init__(self, config_key: str):
        msg = f"缺少配置项: {config_key}"
        super().__init__(msg, "MISSING_CONFIG", {"config_key": config_key})


class InvalidConfigValueError(ConfigException):
    """无效的配置值"""

    def __init__(self, config_key: str, value: any, expected: str):
        msg = f"配置项 {config_key} 的值无效: {value} (期望: {expected})"
        super().__init__(msg, "INVALID_CONFIG_VALUE",
                        {"config_key": config_key, "value": str(value), "expected": expected})


class ValidationException(AppException):
    """验证相关异常基类"""
    pass


class ValidationError(ValidationException):
    """验证失败"""

    def __init__(self, field: str, message: str):
        msg = f"字段 {field} 验证失败: {message}"
        super().__init__(msg, "VALIDATION_ERROR", {"field": field, "message": message})


class SchemaValidationError(ValidationException):
    """Schema 验证失败"""

    def __init__(self, errors: list):
        msg = f"Schema 验证失败: {'; '.join(str(e) for e in errors)}"
        super().__init__(msg, "SCHEMA_VALIDATION_ERROR", {"errors": errors})

