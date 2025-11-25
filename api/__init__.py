"""API接口模块"""

from .compression import compression_api_bp
from .schemas import DetectCapabilitiesRequest, ExecuteCompressionRequest

__all__ = [
    "compression_api_bp",
    "DetectCapabilitiesRequest",
    "ExecuteCompressionRequest",
]

