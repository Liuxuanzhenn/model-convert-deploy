"""统一配置管理

集中管理所有路径、常量和配置项
"""
import os
from pathlib import Path
from typing import Optional


class Config:
    """应用配置类"""

    BASE_DIR = Path(__file__).parent.absolute()
    PROJECT_ROOT = BASE_DIR.parent

    STORAGE_DIR = PROJECT_ROOT / "storage"
    ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
    CONFIGS_DIR = PROJECT_ROOT / "configs"
    WEIGHTS_DIR = PROJECT_ROOT / "weights"
    LOGS_DIR = PROJECT_ROOT / "logs"

    MODELS_DB = STORAGE_DIR / "models_db.json"
    JOBS_DB = STORAGE_DIR / "jobs_db.json"

    # 配置文件
    MODEL_CAPABILITIES = CONFIGS_DIR / "model_capabilities.json"

    # 服务配置
    HOST = os.getenv("APP_HOST", "0.0.0.0")
    PORT = int(os.getenv("APP_PORT", "5000"))
    DEBUG = os.getenv("APP_DEBUG", "True").lower() == "true"

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = LOGS_DIR / "server.log"

    MAX_SPARSITY = 0.9
    DEFAULT_TEMPERATURE = 4.0
    DEFAULT_ALPHA = 0.5

    RECOMMEND_TOP_K = 3
    MIN_CONFIDENCE = 0.5

    MAX_UPLOAD_SIZE = 1024 * 1024 * 1024
    ALLOWED_EXTENSIONS = {".pt", ".pth", ".onnx", ".pb", ".h5"}

    @classmethod
    def ensure_dirs(cls):
        """确保所有必需目录存在"""
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if isinstance(attr, Path) and attr_name.endswith("_DIR"):
                attr.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_artifacts_path(cls, model_id: str, version_id: str, stage: str) -> Path:
        """获取产物路径"""
        return cls.ARTIFACTS_DIR / model_id / version_id / stage

    @classmethod
    def get_log_path(cls, name: Optional[str] = None) -> Path:
        """获取日志文件路径"""
        if name:
            return cls.LOGS_DIR / f"{name}.log"
        return cls.LOG_FILE


Config.ensure_dirs()

