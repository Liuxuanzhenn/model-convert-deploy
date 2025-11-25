"""剪枝后微调策略实现。

说明：
- 在剪枝后对模型进行微调，恢复因剪枝导致的精度损失。
- 支持真实数据集训练。
- 建议剪枝比例较大时（>0.5）使用微调。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

try:
    from ..common import write_report, evaluate_accuracy
except ImportError:
    from strategies.common import write_report, evaluate_accuracy


def finetune_after_pruning(
    model: Any,
    *,
    train_data_dir: str,
    val_data_dir: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    warmup_epochs: int = 2,
    artifacts_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """剪枝后微调训练。

    Args:
        model: 已剪枝的模型
        train_data_dir: 训练数据目录（ImageFolder格式）
        val_data_dir: 验证数据目录
        epochs: 微调轮数
        batch_size: 批次大小
        lr: 学习率
        warmup_epochs: 预热轮数（逐渐增加学习率）
        artifacts_dir: 产物目录

    Returns:
        包含训练状态和精度的字典
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except ImportError:
        rep = {"status": "skipped", "reason": "missing dependencies"}
        write_report(artifacts_dir, rep, "finetune_report.json")
        return rep

    if not os.path.exists(train_data_dir):
        rep = {"status": "error", "reason": "train_data_dir not found"}
        write_report(artifacts_dir, rep, "finetune_report.json")
        return rep

    try:
        # 准备数据
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(train_data_dir, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        val_loader = None
        if val_data_dir and os.path.exists(val_data_dir):
            val_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            val_dataset = datasets.ImageFolder(val_data_dir, transform=val_transform)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # 设置优化器和学习率调度器
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # 使用余弦退火学习率调度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        criterion = nn.CrossEntropyLoss()

        train_losses = []
        val_accuracies = []
        learning_rates = []

        # 训练
        for epoch in range(epochs):
            model.train()

            # Warmup学习率
            if epoch < warmup_epochs:
                current_lr = lr * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr

            epoch_loss = 0.0
            num_batches = 0

            for images, labels in train_loader:
                outputs = model(images)
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

            # 记录学习率
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            # 验证
            if val_loader:
                val_acc = evaluate_accuracy(model, val_loader)
                val_accuracies.append(val_acc)

            # 更新学习率（warmup后）
            if epoch >= warmup_epochs:
                scheduler.step()

        model.eval()

        # 保存微调后的模型
        if artifacts_dir:
            os.makedirs(artifacts_dir, exist_ok=True)
            model_path = os.path.join(artifacts_dir, "model_finetuned.pth")
            torch.save(model.state_dict(), model_path)

        rep = {
            "status": "ok",
            "method": "finetune_after_pruning",
            "epochs": epochs,
            "warmup_epochs": warmup_epochs,
            "initial_lr": lr,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_accuracy": val_accuracies[-1] if val_accuracies else None,
            "train_losses": train_losses,
            "val_accuracies": val_accuracies,
            "learning_rates": learning_rates,
        }
        write_report(artifacts_dir, rep, "finetune_report.json")
        return rep

    except Exception as e:
        rep = {"status": "error", "reason": str(e)}
        write_report(artifacts_dir, rep, "finetune_report.json")
        return rep
