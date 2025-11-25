"""安全工具函数

用于验证和清理用户输入，防止路径遍历等安全问题
"""
import os
import re
from typing import Optional


def sanitize_input_shape(shape_str: str) -> str:
    """清理和验证输入形状字符串
    
    Args:
        shape_str: 输入形状字符串，如 "images:1,3,224,224"
    
    Returns:
        清理后的形状字符串
    
    Raises:
        ValueError: 如果格式无效
    """
    cleaned = re.sub(r'[^a-zA-Z0-9,:_]', '', str(shape_str))
    if not re.match(r'^[\w]+:[0-9,]+$', cleaned):
        raise ValueError(f"Invalid input_shape format: {shape_str}")
    return cleaned


def sanitize_path(path: str, base_dir: str) -> str:
    """验证路径安全性，防止路径遍历攻击
    
    Args:
        path: 要验证的路径
        base_dir: 基础目录
    
    Returns:
        规范化后的绝对路径
    
    Raises:
        ValueError: 如果检测到路径遍历
    """
    base_abs = os.path.abspath(base_dir)
    path_abs = os.path.abspath(os.path.join(base_dir, path))
    if not path_abs.startswith(base_abs):
        raise ValueError(f"Path traversal detected: {path}")
    return path_abs


def sanitize_filename(filename: str) -> str:
    """清理文件名，移除危险字符
    
    Args:
        filename: 原始文件名
    
    Returns:
        清理后的安全文件名
    """
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', str(filename))
    cleaned = cleaned.strip('. ')
    return cleaned if cleaned else "unnamed"


def validate_file_path(file_path: str, base_dir: Optional[str] = None) -> str:
    """验证文件路径安全性
    
    Args:
        file_path: 文件路径
        base_dir: 基础目录（可选）
    
    Returns:
        规范化后的安全路径
    
    Raises:
        ValueError: 如果路径不安全
    """
    if base_dir:
        return sanitize_path(file_path, base_dir)
    
    abs_path = os.path.abspath(file_path)
    if os.path.isabs(file_path) and not abs_path.startswith(os.path.abspath(".")):
        raise ValueError(f"Unsafe absolute path: {file_path}")
    return abs_path
