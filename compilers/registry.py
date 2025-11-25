"""硬件编译器注册表"""
from typing import Dict, Type, Optional
from .base import HardwareCompiler
from .ascend import AscendCompiler
from .tensorrt import TensorRTCompiler
from .cambricon import CambriconCompiler
from .m9 import M9Compiler


# 全局注册表
_COMPILER_REGISTRY: Dict[str, Type[HardwareCompiler]] = {}


def register_compiler(hardware_name: str):
    """注册硬件编译器装饰器

    Args:
        hardware_name: 硬件名称（如 "ascend_npu", "tensorrt"）

    Example:
        @register_compiler("ascend_npu")
        class AscendCompiler(HardwareCompiler):
            ...
    """
    def decorator(cls: Type[HardwareCompiler]):
        _COMPILER_REGISTRY[hardware_name.lower()] = cls
        return cls
    return decorator


# 注册内置编译器
_COMPILER_REGISTRY["ascend_npu"] = AscendCompiler
_COMPILER_REGISTRY["ascend"] = AscendCompiler  # 别名
_COMPILER_REGISTRY["tensorrt"] = TensorRTCompiler
_COMPILER_REGISTRY["cuda_gpu"] = TensorRTCompiler  # 别名
_COMPILER_REGISTRY["nvidia_gpu"] = TensorRTCompiler  # 别名
_COMPILER_REGISTRY["cambricon"] = CambriconCompiler
_COMPILER_REGISTRY["mlu"] = CambriconCompiler  # 别名
_COMPILER_REGISTRY["m9"] = M9Compiler
_COMPILER_REGISTRY["iluvatar"] = M9Compiler  # 别名


def get_compiler(hardware_name: str, output_dir: str) -> Optional[HardwareCompiler]:
    """获取硬件编译器实例

    Args:
        hardware_name: 硬件名称（如 "ascend_npu", "tensorrt"）
        output_dir: 输出目录

    Returns:
        硬件编译器实例，如果不支持则返回None
    """
    compiler_cls = _COMPILER_REGISTRY.get(hardware_name.lower())
    if compiler_cls is None:
        return None
    return compiler_cls(output_dir)


def list_available_compilers() -> Dict[str, bool]:
    """列出所有注册的编译器及其可用性

    Returns:
        {
            "ascend_npu": True,
            "tensorrt": False,
            ...
        }
    """
    available = {}
    for name, compiler_cls in _COMPILER_REGISTRY.items():
        try:
            # 创建临时实例检查可用性
            temp = compiler_cls("")
            available[name] = temp.is_available()
        except Exception:
            available[name] = False
    return available


def list_supported_hardware() -> list:
    """列出支持的硬件列表（用于前端展示）

    Returns:
        [
            {
                "name": "ascend_npu",
                "display_name": "华为昇腾NPU",
                "devices": ["Ascend 310", "Ascend 310P", "Ascend 910"],
                "available": True
            },
            ...
        ]
    """
    availability = list_available_compilers()

    hardware_list = [
        {
            "name": "ascend_npu",
            "display_name": "华为昇腾NPU",
            "description": "支持Ascend 310/910系列",
            "devices": ["Ascend 310", "Ascend 310P", "Ascend 910"],
            "output_format": "om",
            "available": availability.get("ascend_npu", False)
        },
        {
            "name": "tensorrt",
            "display_name": "国产CUDA类GPU",
            "description": "支持NVIDIA及国产GPU（通过TensorRT）",
            "devices": ["支持国产/海外100多种"],
            "output_format": "engine",
            "available": availability.get("tensorrt", False)
        },
        {
            "name": "cambricon",
            "display_name": "寒武纪思元系列",
            "description": "支持MLU220/270/370系列",
            "devices": ["MLU220", "MLU270", "MLU370", "MLU370-X4", "MLU370-X8"],
            "output_format": "cambricon",
            "available": availability.get("cambricon", False)
        },
        {
            "name": "m9",
            "display_name": "通用高/M9系列",
            "description": "支持天数智芯M9系列GPU",
            "devices": ["M9", "M9-Pro", "M9-Ultra"],
            "output_format": "ixmodel",
            "available": availability.get("m9", False)
        },
    ]

    return hardware_list
