"""数据预处理工具函数"""

from typing import Any, Dict

try:
    from ..core.enums import normalize as _normalize, Framework, Family, ModelCategory, ModelFormat
except ImportError:
    try:
        from core.enums import normalize as _normalize, Framework, Family, ModelCategory, ModelFormat
    except ImportError:
        try:
            from app.core.enums import normalize as _normalize, Framework, Family, ModelCategory, ModelFormat
        except ImportError:
            _normalize = None
            Framework = None
            Family = None
            ModelCategory = None
            ModelFormat = None


def compat_preprocess(data: Dict[str, Any]) -> Dict[str, Any]:
    """兼容性预处理：归一化字段（统一版本）"""
    try:
        cp = dict(data or {})
    except (TypeError, ValueError):
        return data or {}

    if _normalize:
        for field, enum_cls in [
            ("framework", Framework),
            ("family", Family),
            ("model_category", ModelCategory),
            ("format", ModelFormat),
        ]:
            if cp.get(field) and enum_cls:
                try:
                    cp[field] = _normalize(enum_cls, cp[field])
                except (ValueError, TypeError):
                    pass
                except Exception:
                    pass

    if cp.get("model_type") and not cp.get("model_category"):
        try:
            if _normalize and ModelCategory:
                cat = _normalize(ModelCategory, cp["model_type"])
                cp["model_category"] = cat
                cp.pop("model_type", None)
                cat2family = {
                    "image_classification": "resnet",
                    "object_detection": "yolo",
                    "instance_segmentation": "other",
                    "semantic_segmentation": "other",
                }
                if not cp.get("family"):
                    cp["family"] = cat2family.get(cat, "generic")
        except (ValueError, TypeError):
            pass
        except Exception:
            pass

    return cp

