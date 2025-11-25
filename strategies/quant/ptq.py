"""后训练量化（PTQ）核心实现 - FP16、INT8动态、INT8静态量化"""

from typing import Any, Dict, Tuple, Optional, Sequence


def apply_fp16(model: Any) -> Tuple[Any, Dict[str, str]]:
    """FP16量化"""
    info: Dict[str, str] = {"precision": "fp16"}
    try:
        import torch
        if hasattr(model, "half"):
            model = model.half()
        return model, info
    except Exception:
        info["fallback"] = "exception"
        return model, info


def apply_int8_dynamic(model: Any, module_types: Optional[tuple] = None) -> Tuple[Any, Dict[str, str]]:
    """INT8动态量化"""
    info: Dict[str, str] = {"precision": "int8_dynamic"}
    try:
        import torch
        from torch import nn
        types = module_types or (nn.Linear,)
        model = torch.quantization.quantize_dynamic(model, set(types), dtype=torch.qint8)
        return model, info
    except Exception:
        info["fallback"] = "exception"
        return model, info


def apply_int8_static(
    model: Any,
    calib_dir: Optional[str] = None,
    calib_num: Optional[int] = None,
    input_shape: Sequence[int] = (1, 3, 224, 224),
) -> Tuple[Any, Dict[str, str]]:
    """INT8静态量化（PTQ）"""
    info: Dict[str, str] = {"precision": "int8_static"}
    try:
        import torch
        import os
        from torch.ao.quantization import get_default_qconfig_mapping, prepare_fx, convert_fx
    except ImportError:
        try:
            from torch.quantization import get_default_qconfig, prepare, convert
            model.eval()
            qconfig = get_default_qconfig("fbgemm")
            model.qconfig = qconfig
            prepared = prepare(model, inplace=False)
            steps = int(calib_num or 8)
            shp = tuple(int(v) for v in (input_shape or (1, 3, 224, 224)))
            for _ in range(max(1, min(128, steps))):
                prepared(torch.randn(*shp))
            return convert(prepared, inplace=False), info
        except Exception:
            info["fallback"] = "no_torch"
            return model, info
    except Exception:
        info["fallback"] = "no_torch"
        return model, info

    try:
        model.eval()
        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        prepared = prepare_fx(model, qconfig_mapping)
        steps = int(calib_num or 8)
        shp = tuple(int(v) for v in (input_shape or (1, 3, 224, 224)))

        if calib_dir and os.path.exists(calib_dir):
            try:
                from torchvision import datasets, transforms
                from torch.utils.data import DataLoader
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(shp[2]),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                dataset = datasets.ImageFolder(calib_dir, transform=transform)
                dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
                calibrated = 0
                for images, _ in dataloader:
                    if calibrated >= steps:
                        break
                    prepared(images)
                    calibrated += 1
                info["calibration"] = "real_data"
                info["calibration_samples"] = str(calibrated)
            except Exception:
                for _ in range(max(1, min(128, steps))):
                    prepared(torch.randn(*shp))
                info["calibration"] = "random_data"
        else:
            for _ in range(max(1, min(128, steps))):
                prepared(torch.randn(*shp))
            info["calibration"] = "random_data"

        return convert_fx(prepared), info
    except Exception:
        info["fallback"] = "exception"
        return model, info

