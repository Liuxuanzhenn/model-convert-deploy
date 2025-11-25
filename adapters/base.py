"""适配器基类"""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional

_STRATEGIES = {}

_STRATEGY_MAP = {
    'decide_and_apply_quant': ('strategies.quant.auto', 'decide_and_apply_quant'),
    'apply_structured': ('strategies.prune.structured', 'apply_structured'),
    'apply_unstructured': ('strategies.prune.unstructured', 'apply_unstructured'),
    'select_sparsity': ('strategies.prune.structured', 'select_sparsity'),
    'kd_minimal': ('strategies.distill.kd_cls', 'kd_minimal'),
}

_FAMILY_KEYWORDS = {
    'yolo': ['yolo'],
    'resnet': ['resnet'],
    'vgg': ['vgg'],
    'vit': ['vit', 'visiontransformer'],
    'inceptionv4': ['inception'],
    'transformer': ['transformer', 'bert'],
    'lstm': ['lstm'],
    'rnn': ['rnn'],
    'vae': ['vae', 'variational'],
    'van': ['van'],
    'gcn': ['gcn', 'graphconv'],
    'cnn': ['conv','cnn'],
}


def _get_strategy(name: str):
    """延迟加载策略模块"""
    if name not in _STRATEGIES:
        if name in _STRATEGY_MAP:
            path, func_name = _STRATEGY_MAP[name]
            for import_path in [path, f'..{path}']:
                try:
                    mod = __import__(import_path, fromlist=[func_name], level=0)
                    _STRATEGIES[name] = getattr(mod, func_name, None)
                    if _STRATEGIES[name]:
                        break
                except (ImportError, AttributeError):
                    continue
        if name not in _STRATEGIES:
            _STRATEGIES[name] = None
    return _STRATEGIES[name]


def _try_import_strategy(module_path: str, func_name: str):
    """尝试导入策略函数"""
    try:
        from importlib import import_module
        mod = import_module(module_path)
        return getattr(mod, func_name, None)
    except (ImportError, AttributeError):
        try:
            mod = __import__(module_path, fromlist=[func_name], level=1)
            return getattr(mod, func_name, None)
        except (ImportError, AttributeError):
            return None


class ModelAdapter(ABC):
    """模型适配器基类"""

    def __init__(self, model_dir: str, artifacts_dir: str, family: Optional[str] = None, model_file: Optional[str] = None) -> None:
        self.model_dir = model_dir
        self.artifacts_dir = artifacts_dir
        os.makedirs(self.artifacts_dir, exist_ok=True)
        self.model: Any = None
        self._operations: List[str] = []
        self.family = family
        self.model_file = model_file

    @abstractmethod
    def load(self) -> None:
        """加载模型到内存"""

    def apply_quant(self, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """量化模型"""
        if self.model is None:
            return {"status": "skipped", "reason": "model not loaded"}

        if not self.family or self.family == "generic":
            self.family = self._detect_family_from_model()

        qa_func = _get_strategy('decide_and_apply_quant')
        if not qa_func:
            return {"status": "fallback"}

        try:
            qc = {k: self._get_cfg(cfg, k) for k in ("precision", "bits", "auto", "calib_dir", "calib_num")}
            new_model, info = qa_func(self.model, qc, self.family)
            self.model = new_model
            if qc.get("auto", False):
                save_info = self._save_model("quantized_auto")
            else:
                precision = qc.get("precision") or info.get("precision") or f"{qc.get('bits', 'unknown')}bit"
                save_info = self._save_model(f"quantized_{precision}")
            if save_info:
                info.update(save_info)
            return info
        except Exception:
            return {"status": "fallback"}

    def _detect_family_from_model(self) -> str:
        """从已加载的模型自动识别family（优先级：文件名 > 字符串 > state_dict键，不使用路径避免误导）"""
        if self.model is None:
            return self._detect_from_filename() or "generic"

        # 1. 文件名检测（优先级最高）
        detected = self._detect_from_filename()
        if detected:
            return detected

        # 2. 模型字符串/类名检测
        detected = self._detect_from_string()
        if detected:
            return detected

        # 3. state_dict键名检测
        if isinstance(self.model, dict):
            detected = self._detect_from_keys(" ".join(self.model.keys()))
            if detected:
                return detected

        # 不再使用路径检测，避免项目路径误导
        return "generic"

    def _detect_from_filename(self) -> Optional[str]:
        """从文件名识别family"""
        weight = self._find_weight()
        if weight:
            filename = os.path.basename(weight).lower()
            for family, keywords in _FAMILY_KEYWORDS.items():
                if any(kw in filename for kw in keywords):
                    return family
        return None

    def _detect_from_string(self) -> Optional[str]:
        """从模型字符串表示识别family"""
        try:
            model_str = str(self.model).lower()
            class_name = self.model.__class__.__name__.lower() if hasattr(self.model, "__class__") else ""
            combined = f"{class_name} {model_str}"

            for family, keywords in _FAMILY_KEYWORDS.items():
                if any(kw in combined for kw in keywords):
                    if family == 'lstm' and 'rnn' in class_name:
                        continue
                    if family == 'rnn' and 'lstm' in class_name:
                        continue
                    return family
        except Exception:
            pass
        return None

    def _detect_from_keys(self, keys_str: str) -> Optional[str]:
        """从state_dict键名识别family"""
        keys_lower = keys_str.lower()
        for family, keywords in _FAMILY_KEYWORDS.items():
            if any(kw in keys_lower for kw in keywords):
                return family
        return None

    def _detect_from_path(self) -> Optional[str]:
        """从目录路径识别family"""
        path_lower = self.model_dir.lower()
        for family, keywords in _FAMILY_KEYWORDS.items():
            if any(kw in path_lower for kw in keywords):
                return family
        return None

    def apply_prune(self, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """剪枝模型"""
        if self.model is None:
            return {"status": "skipped", "reason": "model not loaded"}

        target = self._get_cfg(cfg, "target_sparsity")
        is_auto_mode = self._get_cfg(cfg, "auto", False)
        if is_auto_mode and target is None:
            sel_func = _get_strategy('select_sparsity')
            if sel_func:
                try:
                    target = sel_func(
                        self._get_cfg(cfg, "constraints"),
                        self._get_cfg(cfg, "search"),
                        default=0.3
                    )
                except Exception:
                    pass

        amount = max(0.0, min(0.9, float(target or 0.0)))
        if amount <= 0:
            return {"status": "skipped", "reason": "target_sparsity is 0"}

        if not self.family or self.family == "generic":
            self.family = self._detect_family_from_model()

        prune_func = _try_import_strategy('strategies.prune.auto', 'decide_and_apply_prune')
        if prune_func:
            try:
                prune_cfg = dict(cfg)
                prune_cfg["target_sparsity"] = amount
                res = prune_func(self.model, prune_cfg, self.family)
                if res:
                    operation_name = "pruned_auto" if is_auto_mode else f"pruned_{int(amount*100)}pct"
                    save_info = self._save_model(operation_name)
                    if save_info:
                        res.update(save_info)
                    return res
            except Exception:
                pass

        ptype = str(self._get_cfg(cfg, "type", "structured")).lower()
        fallback_func = _get_strategy('apply_unstructured' if ptype in ["unstructured", "global_unstructured"] else 'apply_structured')
        if fallback_func:
            try:
                res = fallback_func(self.model, target_sparsity=amount)
                operation_name = "pruned_auto" if is_auto_mode else f"pruned_{int(amount*100)}pct"
                save_info = self._save_model(operation_name)
                if save_info:
                    res = res or {}
                    res.update(save_info)
                return res or {"target_sparsity": amount}
            except Exception:
                pass

        return {"target_sparsity": amount, "status": "fallback"}

    def apply_distill(self, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """蒸馏模型"""
        if self.model is None:
            return {"status": "skipped", "reason": "model not loaded"}

        teacher_dir = self._get_cfg(cfg, "teacher_dir")
        if not teacher_dir:
            return {"status": "skipped", "reason": "teacher_dir not provided"}

        teacher_model = self._load_teacher(teacher_dir)
        if not teacher_model:
            return {"status": "skipped", "reason": "teacher load failed"}

        if not self.family or self.family == "generic":
            self.family = self._detect_family_from_model()

        distill_func = _try_import_strategy('strategies.distill.strategy', 'decide_and_apply_distill')
        if distill_func:
            try:
                distill_cfg = {
                    "temperature": self._get_cfg(cfg, "temperature", 4.0),
                    "alpha": self._get_cfg(cfg, "alpha", 0.5),
                    "epochs": self._get_cfg(cfg, "epochs", 10),
                    "batch_size": self._get_cfg(cfg, "batch_size", 32),
                    "lr": self._get_cfg(cfg, "lr", 1e-3),
                    "train_data_dir": self._get_cfg(cfg, "train_data_dir"),
                    "val_data_dir": self._get_cfg(cfg, "val_data_dir"),
                    "input_shape": self._get_cfg(cfg, "input_shape", (2, 3, 224, 224)),
                    "artifacts_dir": self.artifacts_dir
                }
                result = distill_func(student=self.model, teacher=teacher_model, cfg=distill_cfg, family=self.family)
                save_info = self._save_model("distilled")
                if save_info:
                    result = result or {}
                    result.update(save_info)
                return result
            except Exception:
                pass

        kd_func = _get_strategy('kd_minimal')
        if kd_func:
            try:
                result = kd_func(
                    student=self.model,
                    teacher=teacher_model,
                    temperature=self._get_cfg(cfg, "temperature", 4.0),
                    alpha=self._get_cfg(cfg, "alpha", 0.5),
                    steps=self._get_cfg(cfg, "epochs", 1) * 100,
                    input_shape=self._get_cfg(cfg, "input_shape", (2, 3, 224, 224)),
                    artifacts_dir=self.artifacts_dir
                )
                save_info = self._save_model("distilled")
                if save_info:
                    result = result or {}
                    result.update(save_info)
                return result
            except Exception:
                pass

        return {"status": "skipped", "reason": "distillation not available"}

    @abstractmethod
    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出为指定格式，返回产物路径列表"""

    def evaluate(self, artifacts: Optional[List[str]] = None) -> Dict[str, Any]:
        """评估模型性能 - 只统计本次操作生成的最新文件"""
        size_before = 0.0
        weight_file = self._find_weight()
        if weight_file and os.path.exists(weight_file):
            size_before = os.path.getsize(weight_file) / (1024 * 1024)

        size_after = 0.0
        if artifacts:
            pytorch_files = [p for p in artifacts if p.endswith((".pt", ".pth")) and os.path.exists(p)]
            if pytorch_files:
                optimized_files = [p for p in pytorch_files if any(kw in os.path.basename(p).lower() for kw in ["quantized", "pruned", "distilled"])]
                target_files = optimized_files if optimized_files else pytorch_files
                
                # 取最新生成的文件
                if target_files:
                    try:
                        latest_file = max(target_files, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0)
                        size_after = os.path.getsize(latest_file) / (1024 * 1024)
                    except Exception:
                        # Fallback：如果获取修改时间失败，仍然取最大的
                        max_size = max((os.path.getsize(p) for p in target_files if os.path.exists(p)), default=0.0)
                        size_after = max_size / (1024 * 1024)
            else:
                exclude_exts = (".json", ".txt", ".log", ".yaml", ".yml")
                size_after = sum(os.path.getsize(p) for p in artifacts 
                                if os.path.exists(p) and not p.endswith(exclude_exts)) / (1024 * 1024)
        
        # 如果artifacts为空或size_after仍为0，尝试从artifacts_dir查找最新的模型文件
        if size_after == 0.0 and os.path.exists(self.artifacts_dir):
            try:
                pt_files = [os.path.join(self.artifacts_dir, f) 
                           for f in os.listdir(self.artifacts_dir) 
                           if f.endswith((".pt", ".pth")) and os.path.isfile(os.path.join(self.artifacts_dir, f))]
                if pt_files:
                    # 取最新修改的文件
                    latest_file = max(pt_files, key=os.path.getmtime)
                    size_after = os.path.getsize(latest_file) / (1024 * 1024)
            except Exception:
                pass

        return {
            "size_before_mb": round(size_before, 4),
            "size_after_mb": round(size_after, 4),
            "latency_ms_cpu": None,
        }

    def write_metrics(self, metrics: Dict[str, Any], filename: str = "metrics.json") -> str:
        """写入指标文件"""
        path = os.path.join(self.artifacts_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        return path

    def _get_cfg(self, cfg: Any, key: str, default: Any = None) -> Any:
        """从配置中获取值"""
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default) if hasattr(cfg, key) else default

    def _find_weight(self, extensions: tuple = (".pt", ".pth", ".safetensors", ".onnx", ".pb", ".h5", ".ckpt", ".pdmodel", ".pdparams"), directory: Optional[str] = None) -> Optional[str]:
        """查找模型权重文件"""
        if self.model_file:
            file_path = os.path.join(self.model_dir, self.model_file)
            if os.path.exists(file_path):
                return file_path
        search_dir = directory or self.model_dir
        if not os.path.isdir(search_dir):
            return None
        for name in os.listdir(search_dir):
            if name.lower().endswith(extensions):
                return os.path.join(search_dir, name)
        return None

    def _find_weight_in_dir(self, directory: str) -> Optional[str]:
        """在指定目录查找权重文件"""
        return self._find_weight(directory=directory)

    def _check_int8_quantization(self) -> bool:
        """检查模型是否有INT8量化"""
        if not hasattr(self.model, 'named_modules'):
            return False
        try:
            for module in self.model.named_modules():
                if hasattr(module[1], 'qconfig') and module[1].qconfig is not None:
                    return True
                module_type = str(type(module[1])).lower()
                if 'quant' in module_type or 'quantized' in module_type:
                    return True
        except Exception:
            pass
        return False

    def _save_model(self, operation: str) -> Optional[Dict[str, Any]]:
        """保存PyTorch模型 - 压缩后保存完整模型"""
        if self.model is None:
            return None

        try:
            import torch
            self._operations.append(operation)
            filename = f"model_{operation}.pt" if len(self._operations) == 1 else f"model_{'_'.join(self._operations)}.pt"
            model_path = os.path.join(self.artifacts_dir, filename)
            torch.save(self.model, model_path, _use_new_zipfile_serialization=False)
            return {
                "pytorch_path": model_path,
                "outputs": [model_path],
                "pytorch_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 2),
                "operations": list(self._operations)
            }
        except Exception:
            return None

    def cleanup(self) -> None:
        """释放适配器持有的资源，确保长期运行时不会泄漏内存"""
        try:
            self._operations.clear()
        except Exception:
            self._operations = []
        self.model = None

    def _load_teacher(self, teacher_dir: str) -> Any:
        """加载教师模型"""
        try:
            import torch
            weight = self._find_weight_in_dir(teacher_dir)
            if not weight:
                return None
            try:
                model = torch.load(weight, map_location='cpu', weights_only=False)
            except TypeError:
                model = torch.load(weight, map_location='cpu')
            return model.get('model', model) if isinstance(model, dict) else model
        except Exception:
            return None

    def _load_weight_file(self, weight_path: str) -> Any:
        """加载权重文件，支持多种格式"""
        ext = weight_path.lower().split('.')[-1]
        try:
            if ext in ['pt', 'pth']:
                import torch
                try:
                    return torch.load(weight_path, map_location='cpu', weights_only=False)
                except TypeError:
                    return torch.load(weight_path, map_location='cpu')
            elif ext == 'pkl':
                import pickle
                with open(weight_path, 'rb') as f:
                    return pickle.load(f)
            elif ext == 'safetensors':
                try:
                    from safetensors.torch import load_file
                    return load_file(weight_path)
                except ImportError:
                    return None
            elif ext == 'ckpt':
                try:
                    from mindspore import load_checkpoint  # type: ignore
                    return load_checkpoint(weight_path)
                except ImportError:
                    return weight_path
            elif ext in ['air', 'mindir']:
                return weight_path
        except Exception:
            pass
        return None

    def _export_torchscript(self, example_input: Any, filename: str = "model.torchscript.pt") -> Optional[str]:
        """导出TorchScript"""
        try:
            import torch
            if hasattr(self.model, 'eval'):
                self.model.eval()
            ops = self._operations if self._operations else []
            if not ops:
                try:
                    ops = self._parse_ops_from_filename()
                except Exception:
                    ops = []
            if ops:
                base, ext = os.path.splitext(filename)
                filename = f"{base}_{'_'.join(ops)}{ext}"
            traced = torch.jit.trace(self.model, example_input)
            traced.save(os.path.join(self.artifacts_dir, filename))
            return os.path.join(self.artifacts_dir, filename)
        except Exception:
            return None

    def _export_onnx(self, example_input: Any, filename: str = "model.onnx", 
                     opset: int = 11, input_names: List[str] = None, 
                     output_names: List[str] = None) -> Optional[str]:
        """导出ONNX"""
        try:
            import torch
            if self.model is None:
                return None
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # 检测模型数据类型并匹配输入
            model_dtype = None
            is_int8_quantized = False
            if hasattr(self.model, 'parameters'):
                for param in self.model.parameters():
                    if param is not None:
                        model_dtype = param.dtype
                        # 只检查INT8量化数据类型
                        if model_dtype in (torch.qint8, torch.quint8):
                            is_int8_quantized = True
                        break
            
            # 从操作历史或文件名解析操作信息
            ops = self._operations if self._operations else []
            if not ops:
                try:
                    ops = self._parse_ops_from_filename()
                except Exception:
                    ops = []
            
            # 检查操作历史中是否包含int8量化
            if not is_int8_quantized and ops:
                is_int8_quantized = any('int8' in str(op).lower() for op in ops)
            
            if is_int8_quantized:
                # INT8量化模型无法直接导出ONNX，需要从原始模型导出
                raise ValueError(
                    f"INT8量化模型无法直接导出ONNX格式。"
                    f"原因：INT8量化改变了模型结构（使用QuantizedLinear等量化层），"
                    f"torch.onnx.export不支持这些量化算子。"
                    f"建议：1) 使用FP16量化（支持ONNX导出）；"
                    f"2) 从原始模型（raw目录）直接导出ONNX，然后使用ONNX量化工具进行INT8量化。"
                )
            
            if model_dtype == torch.float16:
                example_input = example_input.half()
            elif model_dtype is not None and model_dtype != torch.float32:
                example_input = example_input.to(dtype=model_dtype)
            
            if ops:
                base, ext = os.path.splitext(filename)
                filename = f"{base}_{'_'.join(ops)}{ext}"
            
            path = os.path.join(self.artifacts_dir, filename)
            input_names = input_names or ['input']
            output_names = output_names or ['output']
            torch.onnx.export(
                self.model, example_input, path,
                export_params=True, opset_version=opset,
                do_constant_folding=True,
                input_names=input_names, output_names=output_names,
                dynamic_axes={input_names[0]: {0: 'batch_size'}, output_names[0]: {0: 'batch_size'}}
            )
            return path
        except ValueError:
            raise
        except Exception:
            return None

    def _parse_ops_from_filename(self) -> List[str]:
        """从文件名解析操作信息"""
        weight = self._find_weight()
        if not weight:
            return []
        basename = os.path.basename(weight)
        if not basename.startswith("model_"):
            return []
        name_parts = basename.replace(".pt", "").replace(".pth", "").split("_")[1:]
        return name_parts if name_parts else []
