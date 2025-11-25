"""CPU 延时评估工具（中文注释）。

说明：
- 在 artifacts 目录中自动选择优先的模型文件（TorchScript .pt 或 ONNX .onnx）。
- 若本地安装了对应运行时（PyTorch/onnxruntime），则进行 10 次推理测平均延时；否则返回 None。
"""

from __future__ import annotations

import glob
import os
import time
from typing import Optional


def _pick_artifact(artifacts_dir: str) -> Optional[str]:
    # priority: torchscript *.pt -> onnx *.onnx
    for pat in [
        os.path.join(artifacts_dir, "*.torchscript.pt"),
        os.path.join(artifacts_dir, "*.pt"),
        os.path.join(artifacts_dir, "*.onnx"),
    ]:
        g = sorted(glob.glob(pat))
        if g:
            return g[0]
    return None


def _default_shape(family_hint: str) -> tuple[int, int, int, int]:
    f = (family_hint or "").lower()
    if "yolo" in f or "det" in f:
        return (1, 3, 640, 640)
    return (1, 3, 224, 224)


def _latency_torchscript(pt_path: str, shape: tuple[int, int, int, int]) -> Optional[float]:
    try:
        import torch  # type: ignore
    except Exception:
        return None
    try:
        m = torch.jit.load(pt_path, map_location="cpu")
        m.eval()
        x = torch.randn(*shape)
        # warmup
        for _ in range(2):
            _ = m(x)
        t0 = time.perf_counter()
        for _ in range(10):
            _ = m(x)
        t1 = time.perf_counter()
        return round((t1 - t0) * 1000.0 / 10.0, 3)
    except Exception:
        return None


def _latency_onnx(onnx_path: str, shape: tuple[int, int, int, int]) -> Optional[float]:
    try:
        import onnxruntime as ort  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None
    try:
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])  # type: ignore
        inputs = sess.get_inputs()
        if not inputs:
            return None
        name = inputs[0].name
        try:
            shp = inputs[0].shape
            if all(isinstance(v, int) and v > 0 for v in shp):
                shape = tuple(int(v) for v in shp)  # type: ignore
        except Exception:
            pass
        x = (np.random.randn(*shape)).astype("float32")
        # warmup
        for _ in range(2):
            _ = sess.run(None, {name: x})
        t0 = time.perf_counter()
        for _ in range(10):
            _ = sess.run(None, {name: x})
        t1 = time.perf_counter()
        return round((t1 - t0) * 1000.0 / 10.0, 3)
    except Exception:
        return None


def measure_latency_ms(artifacts_dir: str, family_hint: str = "") -> Optional[float]:
    """Measure CPU latency (ms) for an exported artifact in artifacts_dir.

    Returns None if no supported runtime is available or artifact not found.
    """
    art = _pick_artifact(artifacts_dir)
    if not art:
        return None
    shape = _default_shape(family_hint)
    low = art.lower()
    if low.endswith(".pt"):
        return _latency_torchscript(art, shape)
    if low.endswith(".onnx"):
        return _latency_onnx(art, shape)
    return None

