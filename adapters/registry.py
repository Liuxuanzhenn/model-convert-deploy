from typing import Dict, Tuple, Type

# Adapter 注册表： (framework, family) -> AdapterClass
# 运行时由 /optimize 解析后选择对应 Adapter

_REGISTRY: Dict[Tuple[str, str], Type] = {}


def register(framework: str, family: str):
    def _inner(cls: Type):
        _REGISTRY[(framework.lower(), family.lower())] = cls
        return cls
    return _inner


def get_adapter(framework: str, family: str):
    return _REGISTRY.get((framework.lower(), family.lower()))
