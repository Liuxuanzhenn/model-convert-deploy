"""压缩服务

统一导入接口，实际实现已拆分到独立文件
"""
from services.estimator import CompressionEstimator
from services.recommender import CompressionRecommender
from services.validator import ConfigValidator
