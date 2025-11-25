"""集中字段枚举定义（中文注释）。

作用：
- 作为参数与前端映射的稳定"字段标识"（唯一真源）。
- Pydantic Schema 直接引用这些枚举，保证接口收参稳定与可验证。
- 服务端在 /optimize 中将这些枚举解包为基础字符串再传入适配器。
"""

from enum import Enum
from typing import Dict, List, Type

class TargetType(str, Enum):
    deep_learning = "deep_learning"
    traditional_ml = "traditional_ml"
    llm_finetune = "llm_finetune"


class TaskDomain(str, Enum):
    """任务领域（一级分类）

    适用于：deep_learning 和 traditional_ml
    不适用于：llm_finetune（大模型微调不需要领域分类）
    """
    text = "text"
    image = "image"
    tabular = "tabular"
    other = "other"


class Framework(str, Enum):
    pytorch = "pytorch"
    tensorflow = "tensorflow"
    keras = "keras"
    paddlepaddle = "paddlepaddle"
    caffe = "caffe"
    oneflow = "oneflow"
    mindspore = "mindspore"
    sklearn = "sklearn"
    onnx = "onnx"

class ModelFormat(str, Enum):
    pt = "pt"
    pth = "pth"
    pkl = "pkl"
    safetensors = "safetensors"
    ckpt = "ckpt"
    onnx = "onnx"
    savedmodel = "savedmodel"
    h5 = "h5"
    caffemodel = "caffemodel"
    pb = "pb"
    pmml = "pmml"
    trt = "trt"
    om = "om"
    torchscript = "torchscript"
    air = "air"
    mindir = "mindir"
    zip = "zip"

class ModelCategory(str, Enum):
    image_classification = "image_classification"
    object_detection = "object_detection"
    instance_segmentation = "instance_segmentation"
    semantic_segmentation = "semantic_segmentation"
    text_classification = "text_classification"
    text_entity_extraction = "text_entity_extraction"
    text_relation_extraction = "text_relation_extraction"
    text_similarity = "text_similarity"
    nlp_ner = "nlp_ner"
    retrieval = "retrieval"
    kmeans = "kmeans"
    other = "other"

class Family(str, Enum):
    yolo = "yolo"
    resnet = "resnet"
    cnn = "cnn"
    transformer = "transformer"
    rnn = "rnn"
    lstm = "lstm"
    gcn = "gcn"
    vae = "vae"
    van = "van"
    dbscan = "dbscan"
    spectral_clustering = "spectral_clustering"
    sc = "sc"
    kmeans = "kmeans"
    other = "other"

class QuantPrecision(str, Enum):
    fp16 = "fp16"
    int8_dynamic = "int8_dynamic"
    int8_static = "int8_static"

class PruneType(str, Enum):
    structured = "structured"
    global_unstructured = "global_unstructured"

class PruneGranularity(str, Enum):
    channel = "channel"
    unstructured = "unstructured"

class ExportFormat(str, Enum):
    pt = "pt"
    torchscript = "torchscript"
    onnx = "onnx"
    tflite = "tflite"
    paddle_infer = "paddle_infer"

class Runtime(str, Enum):
    torch = "torch"
    onnxruntime = "onnxruntime"
    tensorrt = "tensorrt"
    tflite = "tflite"
    paddle = "paddle"

class CompileTarget(str, Enum):
    tensorrt = "tensorrt"
    ascend = "ascend"

_ALIAS: Dict[str, str] = {
    "pytorch": "pytorch", "torch": "pytorch", "pyTorch": "pytorch",
    "tensorflow": "tensorflow", "tf": "tensorflow",
    "paddle": "paddlepaddle", "paddlepaddle": "paddlepaddle",
    "mindspore": "mindspore", "ms": "mindspore",
    ".pt": "pt", ".pth": "pth", ".pkl": "pkl", ".safetensors": "safetensors",
    ".ckpt": "ckpt", ".onnx": "onnx",
    ".pb": "pb", ".h5": "h5", ".caffemodel": "caffemodel",
    ".pmml": "pmml", ".trt": "trt", ".om": "om", ".torchscript": "torchscript",
    ".air": "air", ".mindir": "mindir",
    ".zip": "zip",
    "图像分类": "image_classification",
    "图像 分类": "image_classification",
    "image分类": "image_classification",
    "物体检测": "object_detection",
    "目标检测": "object_detection",
    "对象检测": "object_detection",
    "实体分割": "instance_segmentation",
    "实例分割": "instance_segmentation",
    "语义分割": "semantic_segmentation",
    "文本分类": "text_classification",
    "文本实体抽取": "text_entity_extraction",
    "文本实体关系抽取": "text_relation_extraction",
    "短文本相似度": "text_similarity",
    "命名实体识别": "nlp_ner",
    "ner": "nlp_ner",
    "yolo系列": "yolo", "yolo": "yolo",
    "resnet系列": "resnet", "resnet": "resnet",
    "cnn": "cnn", "卷积神经网络": "cnn",
    "transformer": "transformer", "transformer系列": "transformer",
    "rnn": "rnn", "循环神经网络": "rnn",
    "lstm": "lstm", "LSTM": "lstm", "长短期记忆网络": "lstm",
    "gcn": "gcn", "图卷积网络": "gcn", "GCN": "gcn",
    "vae": "vae", "VAE": "vae", "变分自编码器": "vae",
    "van": "van", "VAN": "van", "视觉注意力网络": "van",
    "dbscan": "dbscan", "DBSCAN": "dbscan",
    "spectral_clustering": "spectral_clustering", "sc": "spectral_clustering",
    "谱聚类": "spectral_clustering", "谱聚类算法": "spectral_clustering",
    "kmeans": "kmeans", "k均值": "kmeans",
    "fp16": "fp16", "半精度": "fp16",
    "int8动态": "int8_dynamic", "int8_dynamic": "int8_dynamic",
    "int8静态": "int8_static", "int8_static": "int8_static",
    "结构化": "structured", "structured": "structured",
    "非结构化": "global_unstructured", "global_unstructured": "global_unstructured",
    "图像": "image", "image": "image",
    "文本": "text", "text": "text",
    "表格": "tabular", "tabular": "tabular",
    "深度学习": "deep_learning", "deep_learning": "deep_learning",
    "传统机器学习": "traditional_ml", "traditional_ml": "traditional_ml",
    "大模型微调": "llm_finetune", "llm_finetune": "llm_finetune",
}

def normalize(enum_cls: Type[Enum], value: str) -> str:
    if value is None:
        raise ValueError("enum value is None")
    key = str(value).strip()
    key = _ALIAS.get(key, key)
    key = key.lower()
    choices = {e.value for e in enum_cls}
    if key not in choices:
        raise ValueError(f"'{value}' not in {sorted(list(choices))}")
    return key


def get_category_hierarchy() -> Dict[str, List[str]]:
    """获取模型类别的层级结构

    适用于 deep_learning 和 traditional_ml
    llm_finetune 不使用此层级结构
    """
    return {
        "text": [
            "text_classification",
            "text_entity_extraction",
            "text_relation_extraction",
            "text_similarity",
            "nlp_ner",
        ],
        "image": [
            "image_classification",
            "object_detection",
            "instance_segmentation",
            "semantic_segmentation",
        ],
        "tabular": [
            "kmeans",
        ],
        "other": [
            "retrieval",
            "other",
        ]
    }


def export_enums() -> Dict[str, List[str]]:
    def names(cls: Type[Enum]) -> List[str]:
        return [e.value for e in cls]
    return {
        "target_type": names(TargetType),
        "task_domain": names(TaskDomain),
        "framework": names(Framework),
        "format": names(ModelFormat),
        "model_category": names(ModelCategory),
        "model_category_hierarchy": get_category_hierarchy(),
        "family": names(Family),
        "quant_precision": names(QuantPrecision),
        "prune_type": names(PruneType),
        "prune_granularity": names(PruneGranularity),
        "export_format": names(ExportFormat),
        "runtime": names(Runtime),
        "compile_target": names(CompileTarget),
    }

