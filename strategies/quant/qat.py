"""量化感知训练（QAT）策略实现。

说明：
- 在训练过程中模拟量化效果，使模型适应量化后的精度损失。
- 支持真实数据集训练。
- 相比PTQ（后训练量化），QAT能获得更高的量化精度。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    from ..common import write_report, evaluate_accuracy
except ImportError:
    from strategies.common import write_report, evaluate_accuracy


def quantization_aware_training(
    model: Any,
    *,
    train_data_dir: str,
    val_data_dir: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    qconfig: str = "fbgemm",
    artifacts_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """执行量化感知训练（QAT）。

    Args:
        model: 待量化的模型
        train_data_dir: 训练数据目录（ImageFolder格式）
        val_data_dir: 验证数据目录
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        qconfig: 量化配置（fbgemm用于x86, qnnpack用于ARM）
        artifacts_dir: 产物目录

    Returns:
        包含训练状态和精度的字典
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
        import torch.quantization as quant
    except ImportError:
        rep = {"status": "skipped", "reason": "missing dependencies"}
        write_report(artifacts_dir, rep, "qat_report.json")
        return rep

    if not os.path.exists(train_data_dir):
        rep = {"status": "error", "reason": "train_data_dir not found"}
        write_report(artifacts_dir, rep, "qat_report.json")
        return rep

    try:
        # 准备数据
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        val_loader = None
        if val_data_dir and os.path.exists(val_data_dir):
            val_dataset = datasets.ImageFolder(val_data_dir, transform=transform)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # 配置QAT
        model.train()
        model.qconfig = quant.get_default_qat_qconfig(qconfig)

        # 准备QAT模型
        model_prepared = quant.prepare_qat(model, inplace=False)

        # 训练
        optimizer = torch.optim.Adam(model_prepared.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            model_prepared.train()
            epoch_loss = 0.0
            num_batches = 0

            for images, labels in train_loader:
                outputs = model_prepared(images)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            train_losses.append(avg_loss)

            # 验证
            if val_loader:
                val_acc = evaluate_accuracy(model_prepared, val_loader)
                val_accuracies.append(val_acc)

        # 转换为量化模型
        model_prepared.eval()
        model_quantized = quant.convert(model_prepared, inplace=False)

        # 保存量化模型
        if artifacts_dir:
            os.makedirs(artifacts_dir, exist_ok=True)
            model_path = os.path.join(artifacts_dir, "model_qat.pth")
            torch.save(model_quantized.state_dict(), model_path)

        rep = {
            "status": "ok",
            "method": "qat",
            "epochs": epochs,
            "qconfig": qconfig,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_accuracy": val_accuracies[-1] if val_accuracies else None,
            "train_losses": train_losses,
            "val_accuracies": val_accuracies,
        }
        write_report(artifacts_dir, rep, "qat_report.json")
        return rep

    except Exception as e:
        rep = {"status": "error", "reason": str(e)}
        write_report(artifacts_dir, rep, "qat_report.json")
        return rep


def apply_qat(model: Any, qc: Dict[str, Any]) -> tuple[Any, Dict[str, Any]]:
    """应用QAT量化的统一接口

    Args:
        model: 待量化的模型
        qc: 量化配置字典

    Returns:
        (量化后的模型, 信息字典)
    """
    train_data_dir = qc.get("train_data_dir")

    # 如果没有训练数据，回退到 int8_dynamic
    if not train_data_dir:
        try:
            import torch
            quantized_model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            result = {
                "status": "fallback",
                "method": "int8_dynamic",
                "reason": "QAT requires train_data_dir, fallback to int8_dynamic"
            }

            # 保存模型
            artifacts_dir = qc.get("artifacts_dir")
            if artifacts_dir:
                os.makedirs(artifacts_dir, exist_ok=True)
                model_path = os.path.join(artifacts_dir, "model_qat_fallback.pt")
                torch.save(quantized_model, model_path)
                result["outputs"] = [model_path]

            return quantized_model, result
        except Exception as e:
            return model, {"status": "error", "reason": f"QAT fallback failed: {str(e)}"}

    val_data_dir = qc.get("val_data_dir")
    epochs = qc.get("epochs", 10)
    batch_size = qc.get("batch_size", 32)
    lr = qc.get("lr", 1e-4)
    qconfig = qc.get("qconfig", "fbgemm")
    artifacts_dir = qc.get("artifacts_dir")

    result = quantization_aware_training(
        model,
        train_data_dir=train_data_dir,
        val_data_dir=val_data_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        qconfig=qconfig,
        artifacts_dir=artifacts_dir
    )

    # 添加 outputs 字段
    if result.get("status") == "ok" and artifacts_dir:
        model_path = os.path.join(artifacts_dir, "model_qat.pth")
        if os.path.exists(model_path):
            result["outputs"] = [model_path]

    return model, result


