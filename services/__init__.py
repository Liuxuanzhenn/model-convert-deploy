"""业务服务模块"""

from .model import ModelDetector, TeacherValidator
from .files import ExtraFilesManager
from .compression import CompressionEstimator, CompressionRecommender, ConfigValidator

__all__ = [
    "ModelDetector",
    "TeacherValidator",
    "ExtraFilesManager",
    "CompressionEstimator",
    "CompressionRecommender",
    "ConfigValidator",
]

# 便捷函数（向后兼容）
def estimate_effect(model_info, strategy):
    """便捷函数：预估压缩效果"""
    estimator = CompressionEstimator()
    return estimator.estimate(model_info, strategy)

def recommend_strategy(model_info, constraints=None):
    """便捷函数：推荐压缩策略"""
    recommender = CompressionRecommender()
    return recommender.recommend(model_info, constraints)

def validate_config(framework, family, strategy_config, model_dir=None):
    """便捷函数：验证压缩配置"""
    validator = ConfigValidator()
    return validator.validate(framework, family, strategy_config, model_dir)

