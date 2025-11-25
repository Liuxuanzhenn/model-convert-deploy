"""统一日志管理

提供标准化的日志接口，支持文件和控制台输出
"""
import logging
import sys
from pathlib import Path
from typing import Optional


class Logger:
    """日志管理器"""

    _loggers = {}

    @classmethod
    def get_logger(cls, name: str, log_file: Optional[Path] = None,
                   level: str = "INFO") -> logging.Logger:
        """获取日志记录器"""
        if name in cls._loggers:
            return cls._loggers[name]

        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))

        if logger.handlers:
            return logger

        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        cls._loggers[name] = logger
        return logger


def get_logger(name: str = "app") -> logging.Logger:
    """获取日志记录器的便捷函数"""
    try:
        from .settings import Config
        log_file = Config.get_log_path(name)
        level = Config.LOG_LEVEL
    except:
        log_file = None
        level = "INFO"

    return Logger.get_logger(name, log_file, level)

