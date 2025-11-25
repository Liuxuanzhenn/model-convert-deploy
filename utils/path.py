"""路径工具

合并 path_manager.py 和 path_utils.py
"""
import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def safe_join(base: str, *paths: str) -> Optional[str]:
    """安全的路径拼接，防止路径遍历攻击"""
    try:
        b = os.path.abspath(base)
        p = os.path.abspath(os.path.join(base, *paths))
        if p == b or p.startswith(b + os.sep):
            return p
    except (OSError, ValueError):
        pass
    return None


class PathManager:
    """路径管理器
    
    统一管理模型路径，采用系统路径（model_dir/res_dir/extra_dir）
    """
    
    @staticmethod
    def validate_path(path: str, path_type: str = "directory") -> bool:
        """验证路径是否存在"""
        if not path:
            return False
        
        if not os.path.exists(path):
            logger.warning(f"Path does not exist: {path}")
            return False
        
        if path_type == "directory" and not os.path.isdir(path):
            logger.warning(f"Path is not a directory: {path}")
            return False
        
        if path_type == "file" and not os.path.isfile(path):
            logger.warning(f"Path is not a file: {path}")
            return False
        
        return True
    
    @staticmethod
    def ensure_dir(path: str) -> str:
        """确保目录存在，如果不存在则创建"""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            logger.info(f"Created directory: {path}")
        return path
    
    @staticmethod
    def normalize_path(path: str) -> str:
        """标准化路径（转换为绝对路径）"""
        return os.path.abspath(os.path.normpath(path))
    
    @staticmethod
    def is_system_path(path: str) -> bool:
        """判断是否为系统路径"""
        normalized = PathManager.normalize_path(path)
        
        if normalized.startswith("/nfs/"):
            return True
        
        if "artifacts" in normalized and not normalized.startswith("/nfs/"):
            return False
        
        return True
    
    @staticmethod
    def validate_model_dir(model_dir: str) -> str:
        """验证并标准化model_dir"""
        if not model_dir:
            raise ValueError("model_dir cannot be empty")
        
        normalized = PathManager.normalize_path(model_dir)
        
        if not PathManager.validate_path(normalized, "directory"):
            raise ValueError(f"Invalid model_dir: {normalized}")
        
        return normalized
    
    @staticmethod
    def validate_result_dir(result_dir: str, create_if_not_exists: bool = True) -> str:
        """验证并标准化result_dir"""
        if not result_dir:
            raise ValueError("result_dir cannot be empty")
        
        normalized = PathManager.normalize_path(result_dir)
        
        if create_if_not_exists:
            PathManager.ensure_dir(normalized)
        elif not PathManager.validate_path(normalized, "directory"):
            raise ValueError(f"Invalid result_dir: {normalized}")
        
        return normalized
    
    @staticmethod
    def validate_extra_dir(extra_dir: Optional[str], create_if_not_exists: bool = False) -> Optional[str]:
        """验证并标准化extra_dir"""
        if not extra_dir:
            return None
        
        normalized = PathManager.normalize_path(extra_dir)
        
        if create_if_not_exists:
            PathManager.ensure_dir(normalized)
        elif not PathManager.validate_path(normalized, "directory"):
            raise ValueError(f"Invalid extra_dir: {normalized}")
        
        return normalized

