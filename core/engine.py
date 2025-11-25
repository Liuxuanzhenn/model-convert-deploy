"""核心业务逻辑模块

包含 /optimize 和 /compile 的核心实现
"""
import os
import uuid
import time
from typing import Any, Dict

from utils.file import ensure_dir as _ensure_dir
from utils.data import compat_preprocess
from utils.path import PathManager
from config.logging import get_logger
from adapters.registry import get_adapter
from compression.capabilities_v2 import get_registry_v2
from evaluators import latency as latency_eval
from evaluators.accuracy_stub import compute_accuracy_stub as acc_eval

logger = get_logger("engine")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")


def execute_optimize(data: Dict[str, Any]) -> Dict[str, Any]:
    """执行模型优化（量化、剪枝、蒸馏、导出）"""
    data = compat_preprocess(data)

    framework = data.get("framework", "pytorch")
    family = data.get("family", "generic")
    model_dir = data.get("model_dir")
    model_id = data.get("model_id", f"m_{uuid.uuid4().hex[:6]}")
    version_id = data.get("version_id", f"v_{int(time.time())}")

    if not model_dir:
        model_dir = os.path.join(ARTIFACTS_DIR, model_id, version_id, "raw")

    artifacts_dir = data.get("res_dir") or os.path.join(ARTIFACTS_DIR, model_id, version_id, "optimized")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    strategy = data.get("strategy", {})
    export_config = strategy.get("export", {})
    user_specified_export = bool(export_config.get("formats"))
    auto_export_injected = False
    if not user_specified_export:
        # 如果没有指定格式，根据framework推断默认格式
        if "export" not in strategy:
            strategy["export"] = {}
        if framework == "pytorch":
            default_format = "pt"
        elif framework == "tensorflow":
            default_format = "pb"
        elif framework == "paddlepaddle":
            default_format = "paddle_infer"
        elif framework == "onnx":
            default_format = "onnx"
        else:
            default_format = "pt"
        strategy["export"]["formats"] = [default_format]
        auto_export_injected = True
        data["strategy"] = strategy

    try:
        AdapterCls = get_adapter(str(framework), str(family))
    except Exception as e:
        logger.error(f"Failed to get adapter: {e}")
        return {
            "job_id": f"j_{model_id}_{version_id}",
            "artifacts": [],
            "metrics": {},
            "error": f"Failed to get adapter: {str(e)}"
        }
    if not AdapterCls:
        logger.error(f"No adapter found for framework={framework}, family={family}")
        return {
            "job_id": f"j_{model_id}_{version_id}",
            "artifacts": [],
            "metrics": {},
            "error": f"no adapter for framework={framework}, family={family}"
        }

    logger.debug(f"Starting optimization for model_id={model_id}, version={version_id}")
    adapter = None

    try:
        adapter = AdapterCls(model_dir=model_dir, artifacts_dir=artifacts_dir, family=str(family))

        try:
            adapter.load()
            if adapter.model is None:
                logger.error("Model failed to load - adapter.model is None")
                return {
                    "job_id": f"j_{model_id}_{version_id}",
                    "artifacts": [],
                    "metrics": {},
                    "error": "Model failed to load"
                }
            if (not adapter.family or adapter.family == "generic") and adapter.model:
                detected = adapter._detect_family_from_model()
                if detected != "generic":
                    adapter.family = detected
                    logger.debug(f"Auto-detected family: {detected}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return {
                "job_id": f"j_{model_id}_{version_id}",
                "artifacts": [],
                "metrics": {},
                "error": f"Failed to load model: {str(e)}"
            }

        artifacts = []
        strategy = data.get("strategy", {})
        job_id = f"j_{model_id}_{version_id}"
        
        def _apply_operation(op_name: str, cfg: Dict[str, Any], apply_func) -> bool:
            """统一处理优化操作"""
            if not (cfg and cfg.get("enable")):
                return True
            try:
                logger.debug(f"Applying {op_name}")
                result = apply_func(cfg)
                if result and "outputs" in result:
                    artifacts.extend(result["outputs"])
                return True
            except Exception as e:
                logger.error(f"{op_name.capitalize()} failed: {e}", exc_info=True)
                return False
        
        # 执行优化操作（顺序：剪枝→量化→蒸馏），如果失败则返回错误（资源会在finally中清理）
        try:
            # 1. 剪枝（先执行，在FP32精度下进行）
            if not _apply_operation("pruning", strategy.get("prune", {}), adapter.apply_prune):
                return {"job_id": job_id, "artifacts": [], "metrics": {}, "error": "Pruning failed"}
            # 2. 量化（剪枝后再量化，避免量化后再剪枝导致精度类型转换）
            if not _apply_operation("quantization", strategy.get("quantize", {}), adapter.apply_quant):
                return {"job_id": job_id, "artifacts": [], "metrics": {}, "error": "Quantization failed"}
            # 3. 蒸馏（最后执行）
            if not _apply_operation("distillation", strategy.get("distill", {}), adapter.apply_distill):
                return {"job_id": job_id, "artifacts": [], "metrics": {}, "error": "Distillation failed"}
        except Exception as e:
            logger.error(f"Unexpected error during optimization: {e}", exc_info=True)
            return {"job_id": job_id, "artifacts": [], "metrics": {}, "error": f"Optimization failed: {str(e)}"}

        export_cfg = strategy.get("export", {})
        formats = export_cfg.get("formats") or []
        targets = export_cfg.get("targets", [])

        should_export = bool(formats)
        if auto_export_injected and artifacts:
            # 若只是自动注入的默认导出，并且已经有操作产物，则不再额外导出原始模型
            should_export = False

        if should_export:
            try:
                logger.debug(f"Exporting to formats: {formats}")
                export_artifacts = adapter.export(formats=formats, targets=targets)
                artifacts.extend(export_artifacts)
            except Exception as e:
                logger.error(f"Export failed: {e}")
                return {
                    "job_id": f"j_{model_id}_{version_id}",
                    "artifacts": [],
                    "metrics": {},
                    "error": f"Export failed: {str(e)}"
                }

        try:
            logger.debug("Evaluating model metrics")
            metrics = adapter.evaluate(artifacts=artifacts)
            metrics_path = adapter.write_metrics(metrics)
            if metrics_path not in artifacts:
                artifacts.append(metrics_path)
        except Exception as e:
            logger.warning(f"Evaluation failed: {e}")
            metrics = {}

        if latency_eval and not metrics.get("latency_ms_cpu"):
            try:
                lat = latency_eval.measure_latency_ms(artifacts_dir, family_hint=str(family))
                if lat:
                    metrics["latency_ms_cpu"] = lat
                    adapter.write_metrics(metrics)
            except Exception as e:
                logger.warning(f"Latency measurement failed: {e}")

        if acc_eval:
            try:
                acc = acc_eval(artifacts_dir, family_hint=str(family))
                if isinstance(acc, dict):
                    metrics.update(acc)
                    adapter.write_metrics(metrics)
            except Exception as e:
                logger.warning(f"Accuracy evaluation failed: {e}")

        logger.debug(f"Optimization completed for model_id={model_id}")
        return {
            "job_id": f"j_{model_id}_{version_id}",
            "artifacts": artifacts,
            "metrics": metrics,
        }
    finally:
        if adapter is not None:
            try:
                adapter.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Adapter cleanup failed: {cleanup_error}")


def execute_compile(data: Dict[str, Any]) -> Dict[str, Any]:
    """执行模型编译（TensorRT/Ascend/Cambricon/M9）"""
    artifact_path = data.get("artifact_path")
    target = data.get("target")
    options = data.get("options", {}) or {}

    if not artifact_path or not target:
        return {"error": "artifact_path and target are required"}

    try:
        if not os.path.exists(artifact_path):
            return {"error": f"Artifact path does not exist: {artifact_path}"}
        artifact_path = PathManager.normalize_path(artifact_path)
    except (OSError, ValueError) as e:
        return {"error": f"Invalid artifact path: {str(e)}"}

    version_dir = os.path.dirname(os.path.dirname(artifact_path))
    compiled_dir = os.path.join(version_dir, "compiled")
    target_l = str(target).lower()
    
    try:
        from compilers.registry import get_compiler
        
        compiler = get_compiler(target_l, os.path.join(compiled_dir, target_l))
        if not compiler:
            return {"error": f"Unsupported compilation target: {target}"}
        
        if not compiler.is_available():
            return {"error": f"{target} compiler is not available. Please install dependencies."}
        
        _ensure_dir(compiler.output_dir)
        result = compiler.compile(artifact_path, options)
        out_paths = [result.get("output_path", "")]
        
    except Exception as e:
        logger.error(f"Compilation failed for target {target}: {e}", exc_info=True)
        return {"error": f"Compilation failed: {str(e)}"}

    return {
        "compiled": out_paths,
        "message": "Compilation completed successfully"
    }

