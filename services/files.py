"""文件管理服务"""
import os
import logging
from zipfile import ZipFile, is_zipfile
from typing import Optional, Dict, List
from pathlib import Path
from utils.path import PathManager

logger = logging.getLogger(__name__)


class ExtraFilesManager:
    """Extra目录文件管理器
    
    按照约定的子目录结构管理文件：
    extra_dir/
    ├── calibration_data/     # 校准数据（INT8静态量化）
    ├── train_data/           # 训练数据（QAT、蒸馏）
    ├── val_data/             # 验证数据（剪枝评估）
    ├── teacher_model/       # 教师模型（知识蒸馏）
    └── metadata/             # 元数据文件
    """
    
    def __init__(self, extra_dir: Optional[str] = None):
        self.extra_dir = extra_dir
        self._subdirs = {
            "calibration_data": "calibration_data",
            "train_data": "train_data",
            "val_data": "val_data",
            "teacher_model": "teacher_model",
            "metadata": "metadata"
        }
    
    def get_calib_dir(self) -> Optional[str]:
        """获取校准数据目录（用于INT8静态量化）"""
        return self._get_subdir("calibration_data")
    
    def get_train_data_dir(self) -> Optional[str]:
        """获取训练数据目录（用于QAT、蒸馏）"""
        return self._get_subdir("train_data")
    
    def get_val_data_dir(self) -> Optional[str]:
        """获取验证数据目录（用于剪枝评估）"""
        return self._get_subdir("val_data")
    
    def get_teacher_model_dir(self) -> Optional[str]:
        """获取教师模型目录（用于知识蒸馏）"""
        return self._get_subdir("teacher_model")
    
    def get_metadata_dir(self) -> Optional[str]:
        """获取元数据目录"""
        return self._get_subdir("metadata")
    
    def _get_subdir(self, subdir_name: str) -> Optional[str]:
        """获取子目录路径"""
        if not self.extra_dir:
            return None
        
        if not os.path.exists(self.extra_dir):
            logger.warning(f"Extra directory not found: {self.extra_dir}")
            return None
        
        subdir_path = os.path.join(self.extra_dir, self._subdirs.get(subdir_name, subdir_name))
        
        if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
            return subdir_path
        
        return None
    
    def list_available_files(self) -> Dict[str, List[str]]:
        """列出所有可用的文件"""
        result = {}
        
        for key, subdir_name in self._subdirs.items():
            subdir_path = self._get_subdir(key)
            if subdir_path:
                files = []
                try:
                    for f in os.listdir(subdir_path):
                        file_path = os.path.join(subdir_path, f)
                        if os.path.isfile(file_path):
                            files.append(f)
                    if files:
                        result[key] = files
                except Exception as e:
                    logger.warning(f"Error listing files in {subdir_path}: {e}")
        
        return result
    
    def check_requirements(self, required_types: List[str]) -> Dict[str, bool]:
        """检查必需的文件是否存在"""
        result = {}
        for req_type in required_types:
            if req_type == "calibration_data":
                result[req_type] = self.get_calib_dir() is not None
            elif req_type == "train_data":
                result[req_type] = self.get_train_data_dir() is not None
            elif req_type == "val_data":
                result[req_type] = self.get_val_data_dir() is not None
            elif req_type == "teacher_model":
                result[req_type] = self.get_teacher_model_dir() is not None
            else:
                result[req_type] = False
        
        return result
    
    def extract_and_distribute(self, zip_path: str) -> Dict[str, List[str]]:
        """解压zip文件并自动识别分发到对应子目录"""
        if not self.extra_dir:
            raise ValueError("extra_dir is not set")
        
        if not os.path.exists(zip_path) or not is_zipfile(zip_path):
            raise ValueError(f"Invalid zip file: {zip_path}")
        
        PathManager.ensure_dir(self.extra_dir)
        result = {}
        
        with ZipFile(zip_path, 'r') as zip_file:
            top_level_dirs = self._get_top_level_dirs(zip_file)
            
            for dir_name in top_level_dirs:
                file_type = self._identify_file_type(dir_name)
                if file_type:
                    target_dir = os.path.join(self.extra_dir, self._subdirs[file_type])
                    PathManager.ensure_dir(target_dir)
                    
                    extracted_files = self._extract_directory(zip_file, dir_name, target_dir)
                    if extracted_files:
                        result[file_type] = extracted_files
        
        return result
    
    def _get_top_level_dirs(self, zip_file: ZipFile) -> List[str]:
        """获取zip中的顶层目录列表"""
        dirs = set()
        for name in zip_file.namelist():
            if ".." in name:
                continue
            parts = Path(name).parts
            if parts:
                dirs.add(parts[0])
        return sorted(dirs)
    
    def _identify_file_type(self, dir_name: str) -> Optional[str]:
        """根据目录名识别文件类型"""
        dir_lower = dir_name.lower()
        recognition_map = {
            "calibration_data": ["calibration_data", "calib", "calibration"],
            "train_data": ["train_data", "train", "training"],
            "val_data": ["val_data", "val", "validation", "valid"],
            "teacher_model": ["teacher_model", "teacher"]
        }
        
        for file_type, patterns in recognition_map.items():
            if dir_lower in patterns:
                return file_type
        return None
    
    def _extract_directory(self, zip_file: ZipFile, dir_name: str, target_dir: str) -> List[str]:
        """解压指定目录到目标目录"""
        extracted_files = []
        dir_prefix = dir_name + "/"
        
        for name in zip_file.namelist():
            if name.startswith(dir_prefix) and ".." not in name:
                relative_path = name[len(dir_prefix):]
                if not relative_path:
                    continue
                
                target_path = os.path.join(target_dir, relative_path)
                target_parent = os.path.dirname(target_path)
                if target_parent:
                    PathManager.ensure_dir(target_parent)
                
                if not name.endswith("/"):
                    zip_file.extract(name, target_dir)
                    extracted_files.append(relative_path)
        
        return extracted_files

