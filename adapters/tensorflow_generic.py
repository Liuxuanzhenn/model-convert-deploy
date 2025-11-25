"""TensorFlow 通用适配器"""
import os
import logging
from typing import Iterable, List, Dict, Any, Optional
from .base import ModelAdapter
from .registry import register

logger = logging.getLogger(__name__)


@register("tensorflow", "generic")
@register("tensorflow", "keras")
class TensorFlowGenericAdapter(ModelAdapter):
    """TensorFlow通用适配器 - 支持SavedModel和.h5格式"""

    def _find_model(self) -> str | None:
        """查找TensorFlow模型文件"""
        # 查找SavedModel目录
        for name in os.listdir(self.model_dir):
            model_path = os.path.join(self.model_dir, name)
            if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "saved_model.pb")):
                return model_path

        # 查找.h5文件
        for name in os.listdir(self.model_dir):
            if name.lower().endswith(".h5"):
                return os.path.join(self.model_dir, name)
        
        # 查找.ckpt文件（TensorFlow checkpoint）
        # TensorFlow checkpoint通常有多个文件：model.ckpt.data-00000-of-00001, model.ckpt.index, model.ckpt.meta
        # 或者单个文件：model.ckpt
        ckpt_files = []
        for name in os.listdir(self.model_dir):
            if name.lower().endswith(".ckpt"):
                base_name = name.rsplit(".ckpt", 1)[0]
                # 检查是否有完整的checkpoint文件组
                has_data = any(f.startswith(base_name) and ".data-" in f for f in os.listdir(self.model_dir))
                has_index = any(f.startswith(base_name) and ".index" in f for f in os.listdir(self.model_dir))
                has_meta = any(f.startswith(base_name) and ".meta" in f for f in os.listdir(self.model_dir))
                if has_data or has_index or has_meta:
                    ckpt_files.append(base_name)
        
        if ckpt_files:
            # 返回第一个找到的checkpoint基础名称
            return os.path.join(self.model_dir, ckpt_files[0])

        return None

    def load(self) -> None:
        """加载TensorFlow模型"""
        model_path = self._find_model()
        if not model_path:
            self.model = None
            return

        try:
            import tensorflow as tf
            
            # 判断是checkpoint还是SavedModel/H5
            if model_path.endswith(".ckpt") or (os.path.isfile(model_path) and not os.path.isdir(model_path) and not model_path.endswith(".h5")):
                # 尝试作为checkpoint加载
                # TensorFlow checkpoint需要先构建模型结构，然后加载权重
                # 这里尝试直接加载，如果失败则返回None
                try:
                    # 尝试使用tf.train.Checkpoint加载
                    checkpoint = tf.train.Checkpoint()
                    checkpoint.restore(model_path).expect_partial()
                    # 如果checkpoint只包含权重，需要模型结构才能加载
                    # 这里暂时返回None，需要用户提供模型结构
                    self.model = None
                    return
                except Exception:
                    # checkpoint加载失败，尝试其他方式
                    pass
            
            # SavedModel或H5格式
            self.model = tf.keras.models.load_model(model_path)
        except Exception as e:
            self.model = None

    def apply_quant(self, cfg) -> Dict[str, Any]:
        """应用量化（支持FP16和INT8）"""
        if self.model is None:
            return {"status": "skipped", "reason": "no model"}

        prec = self._get_cfg(cfg, "precision")

        if prec == "fp16":
            try:
                import tensorflow as tf
                from tensorflow.keras import mixed_precision
                mixed_precision.set_global_policy(mixed_precision.Policy('mixed_float16'))

                quantized_path = os.path.join(self.artifacts_dir, "model_fp16")
                self.model.save(quantized_path)
                return {
                    "precision": "fp16",
                    "status": "success",
                    "saved_model_path": quantized_path
                }
            except Exception as e:
                logger.error(f"FP16 quantization failed: {e}")
                return {"status": "error", "reason": str(e)}

        elif prec in ["int8", "int8_static"]:
            try:
                import tensorflow as tf
                converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_types = [tf.int8]
                tflite_model = converter.convert()

                tflite_path = os.path.join(self.artifacts_dir, "model_int8.tflite")
                with open(tflite_path, "wb") as f:
                    f.write(tflite_model)
                return {
                    "precision": "int8",
                    "status": "success",
                    "tflite_path": tflite_path
                }
            except Exception as e:
                logger.error(f"INT8 quantization failed: {e}")
                return {"status": "error", "reason": str(e)}

        return {"precision": prec, "status": "skipped"}

    def apply_prune(self, cfg: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """TensorFlow模型剪枝"""
        if self.model is None:
            return {"status": "skipped", "reason": "no model"}

        try:
            import tensorflow as tf
            import tensorflow_model_optimization as tfmot
        except ImportError:
            return {"status": "error", "reason": "tensorflow_model_optimization not installed"}

        target = self._get_cfg(cfg, "target_sparsity")
        if self._get_cfg(cfg, "auto", False) and target is None:
            target = 0.3

        amount = max(0.0, min(0.9, float(target or 0.3)))
        if amount <= 0:
            return {"status": "skipped", "reason": "target_sparsity is 0"}

        ptype = str(self._get_cfg(cfg, "type", "structured")).lower()
        
        try:
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=amount,
                    begin_step=0,
                    end_step=1000
                )
            }
            
            pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
                self.model, **pruning_params
            )
            
            train_data_dir = self._get_cfg(cfg, "train_data_dir")
            if train_data_dir and os.path.exists(train_data_dir):
                try:
                    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
                        train_data_dir,
                        batch_size=32,
                        image_size=(224, 224),
                        label_mode='int'
                    )
                    
                    pruned_model.compile(
                        optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    epochs = self._get_cfg(cfg, "epochs", 5)
                    pruned_model.fit(train_dataset, epochs=epochs, verbose=0)
                except Exception as e:
                    logger.warning(f"Pruning fine-tuning failed: {e}, using pruned model without fine-tuning")
            
            final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
            
            pruned_path = os.path.join(self.artifacts_dir, f"model_pruned_{int(amount*100)}pct")
            final_model.save(pruned_path)
            self.model = final_model
            self._operations.append(f"pruned_{int(amount*100)}pct")
            
            return {
                "status": "success",
                "target_sparsity": amount,
                "pruning_type": ptype,
                "pruned_path": pruned_path
            }
        except Exception as e:
            logger.error(f"TensorFlow pruning failed: {e}")
            return {"status": "error", "reason": str(e)}

    def export(self, formats: Iterable[str], targets: Iterable[str]) -> List[str]:
        """导出TensorFlow模型（只导出原始格式）"""
        if self.model is None:
            return []

        out = []
        fmts = [str(x).lower() for x in formats]

        try:
            import tensorflow as tf
        except ImportError:
            return out

        # SavedModel导出
        if "savedmodel" in fmts or "pb" in fmts:
            try:
                path = os.path.join(self.artifacts_dir, "saved_model")
                self.model.save(path)
                out.append(path)
            except Exception:
                pass

        # H5格式导出
        if "h5" in fmts:
            try:
                h5_path = os.path.join(self.artifacts_dir, "model.h5")
                self.model.save(h5_path)
                out.append(h5_path)
            except Exception:
                pass

        # Checkpoint导出（保存为checkpoint格式）
        if "ckpt" in fmts:
            try:
                ckpt_path = os.path.join(self.artifacts_dir, "model.ckpt")
                self.model.save_weights(ckpt_path)
                out.append(ckpt_path)
            except Exception:
                pass

        # ONNX格式导出
        if "onnx" in fmts:
            try:
                import tf2onnx
                model_path = self._find_model()
                if not model_path:
                    return out
                
                onnx_path = os.path.join(self.artifacts_dir, "model.onnx")
                
                if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "saved_model.pb")):
                    tf2onnx.convert.from_saved_model(
                        model_path,
                        output_path=onnx_path,
                        opset=13
                    )
                elif model_path.endswith(".pb"):
                    with tf.io.gfile.GFile(model_path, 'rb') as f:
                        graph_def = tf.compat.v1.GraphDef()
                        graph_def.ParseFromString(f.read())
                    
                    onnx_model = tf2onnx.convert.from_graph_def(
                        graph_def,
                        opset=13
                    )
                    
                    with open(onnx_path, 'wb') as f:
                        f.write(onnx_model.SerializeToString())
                
                if os.path.exists(onnx_path):
                    out.append(onnx_path)
            except ImportError:
                logger.warning("tf2onnx not installed, skipping TensorFlow→ONNX conversion")
            except Exception as e:
                logger.error(f"TensorFlow to ONNX conversion failed: {e}")

        return out
