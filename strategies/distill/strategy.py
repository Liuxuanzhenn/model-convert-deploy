"""蒸馏策略选择器 - 根据模型类型自动选择最优蒸馏方法"""

from typing import Any, Dict, Optional

try:
    from .kd_cls import kd_minimal
except ImportError:
    from strategies.distill.kd_cls import kd_minimal


def decide_and_apply_distill(
    student: Any,
    teacher: Any,
    cfg: Dict[str, Any],
    family: Optional[str] = None
) -> Dict[str, Any]:
    """蒸馏策略选择和应用（支持模型感知）
    
    Args:
        student: 学生模型
        teacher: 教师模型
        cfg: 蒸馏配置字典
        family: 模型家族（yolo/resnet/lstm/rnn/gcn/vae等）
    
    Returns:
        蒸馏结果字典
    """
    if not isinstance(cfg, dict):
        return {"status": "skipped", "reason": "invalid config"}
    
    family_lower = str(family or "generic").lower()
    
    # 序列模型和生成模型暂不支持标准蒸馏
    if family_lower in ["lstm", "rnn", "vae", "gcn"]:
        return {
            "status": "skipped",
            "reason": f"{family_lower}模型蒸馏需要特殊实现（MSE损失或重建损失）"
        }
    
    # 分类模型：使用KL散度蒸馏
    if family_lower in ["resnet", "vgg", "vit", "cnn", "van", "inceptionv4"]:
        return kd_minimal(
            student=student,
            teacher=teacher,
            temperature=cfg.get("temperature", 4.0),
            alpha=cfg.get("alpha", 0.5),
            steps=cfg.get("epochs", 1) * 100,
            input_shape=cfg.get("input_shape", (2, 3, 224, 224)),
            artifacts_dir=cfg.get("artifacts_dir"),
            train_data_dir=cfg.get("train_data_dir"),
            val_data_dir=cfg.get("val_data_dir"),
            epochs=cfg.get("epochs", 10),
            batch_size=cfg.get("batch_size", 32),
            lr=cfg.get("lr", 1e-3),
        )
    
    # 检测模型（YOLO）：使用标准蒸馏（未来可扩展检测特定损失）
    elif family_lower == "yolo":
        return kd_minimal(
            student=student,
            teacher=teacher,
            temperature=cfg.get("temperature", 4.0),
            alpha=cfg.get("alpha", 0.7),  # 检测任务通常alpha更高
            steps=cfg.get("epochs", 1) * 100,
            input_shape=cfg.get("input_shape", (2, 3, 640, 640)),  # YOLO默认输入
            artifacts_dir=cfg.get("artifacts_dir"),
            train_data_dir=cfg.get("train_data_dir"),
            val_data_dir=cfg.get("val_data_dir"),
            epochs=cfg.get("epochs", 10),
            batch_size=cfg.get("batch_size", 16),  # 检测任务batch通常更小
            lr=cfg.get("lr", 1e-3),
        )
    
    # 通用模型：使用标准蒸馏
    else:
        return kd_minimal(
            student=student,
            teacher=teacher,
            temperature=cfg.get("temperature", 4.0),
            alpha=cfg.get("alpha", 0.5),
            steps=cfg.get("epochs", 1) * 100,
            input_shape=cfg.get("input_shape", (2, 3, 224, 224)),
            artifacts_dir=cfg.get("artifacts_dir"),
            train_data_dir=cfg.get("train_data_dir"),
            val_data_dir=cfg.get("val_data_dir"),
            epochs=cfg.get("epochs", 10),
            batch_size=cfg.get("batch_size", 32),
            lr=cfg.get("lr", 1e-3),
        )

