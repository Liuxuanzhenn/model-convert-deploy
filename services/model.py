"""模型服务

合并 model_detector.py 和 teacher_validator.py
"""
import os
import logging
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

from adapters.registry import get_adapter
from utils.path import PathManager


def detect_framework_from_files(model_dir: str) -> str:
    """根据文件扩展名识别framework"""
    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        return "pytorch"
    
    files = os.listdir(model_dir)
    extensions = set()
    
    # 先检查SavedModel目录结构
    for f in files:
        item_path = os.path.join(model_dir, f)
        if os.path.isdir(item_path):
            saved_model_pb = os.path.join(item_path, "saved_model.pb")
            if os.path.exists(saved_model_pb):
                return "tensorflow"
    
    # 检查文件扩展名
    for f in files:
        if os.path.isfile(os.path.join(model_dir, f)):
            ext = os.path.splitext(f)[1].lower()
            extensions.add(ext)
    
    # 只识别需要的格式：pytorch, tensorflow, paddlepaddle, onnx, safetensors
    if any(ext in ['.pt', '.pth', '.safetensors'] for ext in extensions):
        return "pytorch"
    if any(ext in ['.pb', '.h5', '.ckpt'] for ext in extensions):
        return "tensorflow"
    if any(ext in ['.pdmodel', '.pdparams'] for ext in extensions):
        return "paddlepaddle"
    if any(ext in ['.onnx'] for ext in extensions):
        return "onnx"
    
    return "pytorch"


def detect_original_format(model_dir: str, framework: str) -> Optional[str]:
    """检测模型的原始格式
    
    Args:
        model_dir: 模型目录路径
        framework: 检测到的框架类型
        
    Returns:
        原始格式字符串，如 "pt", "onnx", "safetensors" 等，如果检测失败返回 None
    """
    if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
        return None
    
    files = os.listdir(model_dir)
    if not files:
        return None
    
    # 先检查SavedModel目录（目录结构）
    for f in files:
        item_path = os.path.join(model_dir, f)
        if os.path.isdir(item_path):
            saved_model_pb = os.path.join(item_path, "saved_model.pb")
            if os.path.exists(saved_model_pb):
                return "savedmodel"
    
    # 根据文件扩展名判断原始格式（只识别需要的格式）
    for f in files:
        if not os.path.isfile(os.path.join(model_dir, f)):
            continue
        
        ext = os.path.splitext(f)[1].lower()
        
        # PyTorch 格式
        if ext in ['.pt', '.pth']:
            return "pt"
        elif ext == '.safetensors':
            return "safetensors"
        # ONNX 格式
        elif ext == '.onnx':
            return "onnx"
        # TensorFlow 格式
        elif ext in ['.pb', '.h5', '.ckpt']:
            if ext == '.pb':
                return "pb"
            elif ext == '.h5':
                return "h5"
            elif ext == '.ckpt':
                return "ckpt"
        # PaddlePaddle 格式
        elif ext in ['.pdmodel', '.pdparams']:
            return "paddle_infer"
    
    # 如果没找到明确的格式，根据 framework 推断默认格式
    if framework == "pytorch":
        return "pt"
    elif framework == "onnx":
        return "onnx"
    elif framework == "tensorflow":
        return "pb"
    elif framework == "paddlepaddle":
        return "paddle_infer"
    
    return None


def detect_family_from_model(model_dir: str, framework: str) -> str:
    """通过加载模型识别family（优先文件名检测，避免路径误导）"""
    # 1. 快速文件名检测（不加载模型，避免自定义类加载失败）
    try:
        import tempfile
        temp_artifacts_dir = tempfile.mkdtemp()
        AdapterCls = get_adapter(framework, "generic")
        if AdapterCls:
            adapter = AdapterCls(model_dir=model_dir, artifacts_dir=temp_artifacts_dir, family="generic")
            # 仅使用文件名检测（不使用路径，避免项目路径误导）
            quick_detected = adapter._detect_from_filename()
            if quick_detected:
                try:
                    import shutil
                    shutil.rmtree(temp_artifacts_dir, ignore_errors=True)
                except Exception:
                    pass
                return quick_detected
        
        try:
            import shutil
            shutil.rmtree(temp_artifacts_dir, ignore_errors=True)
        except Exception:
            pass
    except Exception:
        pass

    # 2. 加载模型后检测（字符串表示、state_dict键）
    AdapterCls = get_adapter(framework, "generic")
    if not AdapterCls:
        logger.warning(f"No adapter found for framework={framework}, returning generic")
        return "generic"
    
    try:
        import tempfile
        temp_artifacts_dir = tempfile.mkdtemp()
        adapter = AdapterCls(model_dir=model_dir, artifacts_dir=temp_artifacts_dir, family="generic")
        adapter.load()
        
        if adapter.model is None:
            # 加载失败，再次尝试文件名检测
            fallback_detected = adapter._detect_from_filename()
            if fallback_detected:
                logger.warning(f"Failed to load model, but detected family from filename: {fallback_detected}")
            else:
                logger.warning("Failed to load model, returning generic")
                fallback_detected = "generic"
            
            try:
                import shutil
                shutil.rmtree(temp_artifacts_dir, ignore_errors=True)
            except Exception:
                pass
            return fallback_detected
        
        # 优先使用adapter自身识别的family
        if adapter.family and adapter.family != "generic":
            detected = adapter.family
        else:
            detected = adapter._detect_family_from_model()
        
        try:
            import shutil
            shutil.rmtree(temp_artifacts_dir, ignore_errors=True)
        except Exception:
            pass
        
        return detected
        
    except Exception as e:
        logger.error(f"Error detecting family: {e}")
        return "generic"


class ModelDetector:
    """模型识别服务类"""
    
    def detect_from_dir(self, model_dir: str) -> Dict[str, Any]:
        """从model_dir识别framework、family和原始格式"""
        if not model_dir or not os.path.exists(model_dir):
            raise ValueError(f"Model directory not found: {model_dir}")
        
        if not os.path.isdir(model_dir):
            raise ValueError(f"Model directory is not a directory: {model_dir}")
        
        framework = detect_framework_from_files(model_dir)
        family = detect_family_from_model(model_dir, framework)
        original_format = detect_original_format(model_dir, framework)
        
        return {
            "framework": framework,
            "family": family,
            "original_format": original_format,
            "model_dir": model_dir,
            "detection_method": "auto"
        }


class TeacherValidator:
    """教师模型验证器"""
    
    def __init__(self):
        self.detector = ModelDetector()
    
    def validate(
        self,
        student_model_dir: str,
        teacher_model_dir: str,
        student_framework: Optional[str] = None,
        student_family: Optional[str] = None
    ) -> Dict[str, Any]:
        """验证教师模型是否是学生模型的"大模型"""
        if not self.detector:
            return {
                "valid": False,
                "reason": "ModelDetector not available",
                "student_info": {},
                "teacher_info": {}
            }
        
        try:
            student_model_dir = PathManager.validate_model_dir(student_model_dir)
            teacher_model_dir = PathManager.validate_model_dir(teacher_model_dir)
        except Exception as e:
            return {
                "valid": False,
                "reason": f"Invalid path: {str(e)}",
                "student_info": {},
                "teacher_info": {}
            }
        
        if not student_framework or not student_family:
            student_detection = self.detector.detect_from_dir(student_model_dir)
            student_framework = student_detection["framework"]
            student_family = student_detection["family"]
        
        student_info = {
            "framework": student_framework,
            "family": student_family,
            "size_mb": self._get_model_size(student_model_dir)
        }
        
        teacher_detection = self.detector.detect_from_dir(teacher_model_dir)
        teacher_info = {
            "framework": teacher_detection["framework"],
            "family": teacher_detection["family"],
            "size_mb": self._get_model_size(teacher_model_dir)
        }
        
        validation_result = self._validate_teacher_student(student_info, teacher_info)
        
        return {
            "valid": validation_result["valid"],
            "reason": validation_result["reason"],
            "student_info": student_info,
            "teacher_info": teacher_info
        }
    
    def _validate_teacher_student(
        self,
        student_info: Dict[str, Any],
        teacher_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """验证教师模型和学生模型的关系"""
        if student_info["framework"] != teacher_info["framework"]:
            return {
                "valid": False,
                "reason": f"Framework mismatch: student={student_info['framework']}, teacher={teacher_info['framework']}"
            }
        
        if student_info["family"] != teacher_info["family"]:
            logger.warning(
                f"Family mismatch: student={student_info['family']}, teacher={teacher_info['family']}. "
                "This may still work but is not recommended."
            )
        
        student_size = student_info.get("size_mb", 0)
        teacher_size = teacher_info.get("size_mb", 0)
        
        if student_size == 0 or teacher_size == 0:
            logger.warning("Cannot determine model size, skipping size validation")
            return {
                "valid": True,
                "reason": "Size validation skipped (size unknown)"
            }
        
        size_ratio = teacher_size / student_size if student_size > 0 else 0
        
        if size_ratio < 1.2:
            return {
                "valid": False,
                "reason": f"Teacher model is not significantly larger than student model. "
                          f"Size ratio: {size_ratio:.2f} (expected >= 1.2). "
                          f"Student: {student_size:.2f}MB, Teacher: {teacher_size:.2f}MB"
            }
        
        return {
            "valid": True,
            "reason": f"Teacher model validated. Size ratio: {size_ratio:.2f}"
        }
    
    def _get_model_size(self, model_dir: str) -> float:
        """获取模型大小（MB）"""
        if not os.path.exists(model_dir) or not os.path.isdir(model_dir):
            return 0.0
        
        total_size = 0.0
        
        try:
            for root, dirs, files in os.walk(model_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        total_size += os.path.getsize(file_path)
            
            return total_size / (1024 * 1024)
        except Exception as e:
            logger.warning(f"Error calculating model size: {e}")
            return 0.0

