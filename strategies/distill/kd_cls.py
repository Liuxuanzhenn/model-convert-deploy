"""分类蒸馏（KD）策略实现（支持真实数据集训练）。

说明：
- 面向分类任务的 teacher-student 蒸馏，使用 KL 散度与温度缩放。
- 支持真实数据集训练（ImageFolder格式）。
- 无训练数据时，使用随机张量进行最小校准式迭代（仅用于演示）。
- 若缺少 PyTorch，则写占位报告后返回。
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

try:
    from ..common import write_report, evaluate_accuracy
except ImportError:
    from strategies.common import write_report, evaluate_accuracy


def kd_minimal(
    student: Any,
    teacher: Any,
    *,
    temperature: float = 4.0,
    alpha: float = 0.5,
    steps: int = 10,
    input_shape: Sequence[int] = (2, 3, 224, 224),
    artifacts_dir: Optional[str] = None,
    train_data_dir: Optional[str] = None,
    val_data_dir: Optional[str] = None,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> Dict[str, Any]:
    """执行知识蒸馏训练并写入报告。

    Args:
        student: 学生模型
        teacher: 教师模型
        temperature: 温度参数
        alpha: 蒸馏损失权重
        steps: 随机数据迭代步数（无真实数据时使用）
        input_shape: 输入形状
        artifacts_dir: 产物目录
        train_data_dir: 训练数据目录（ImageFolder格式）
        val_data_dir: 验证数据目录（ImageFolder格式）
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率

    Returns:
        包含 status/loss/accuracy 等键的字典
    """
    try:
        import torch
        from torch import nn
    except Exception:
        rep = {"status": "skipped", "reason": "no torch"}
        write_report(artifacts_dir, rep, "distill_report.json")
        return rep

    if student is None or not hasattr(student, "parameters"):
        rep = {"status": "skipped", "reason": "student invalid"}
        write_report(artifacts_dir, rep, "distill_report.json")
        return rep
    if teacher is None or not hasattr(teacher, "eval"):
        rep = {"status": "skipped", "reason": "no teacher"}
        write_report(artifacts_dir, rep, "distill_report.json")
        return rep

    # 如果提供了真实数据集，使用真实训练
    if train_data_dir and os.path.exists(train_data_dir):
        return _kd_with_real_data(
            student, teacher, temperature, alpha, artifacts_dir,
            train_data_dir, val_data_dir, epochs, batch_size, lr
        )
    else:
        # 否则使用随机数据（演示模式）
        return _kd_with_random_data(
            student, teacher, temperature, alpha, steps, input_shape, artifacts_dir
        )


def _kd_with_real_data(
    student: Any,
    teacher: Any,
    temperature: float,
    alpha: float,
    artifacts_dir: Optional[str],
    train_data_dir: str,
    val_data_dir: Optional[str],
    epochs: int,
    batch_size: int,
    lr: float,
) -> Dict[str, Any]:
    """使用真实数据集进行知识蒸馏训练"""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms

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

        # 设置模型
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        student.train()
        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        ce_loss = nn.CrossEntropyLoss()

        # 训练
        train_losses = []
        val_accuracies = []

        for epoch in range(epochs):
            student.train()
            epoch_loss = 0.0
            num_batches = 0

            for images, labels in train_loader:
                with torch.no_grad():
                    teacher_logits = teacher(images)
                    if isinstance(teacher_logits, (tuple, list)):
                        teacher_logits = teacher_logits[0]

                student_logits = student(images)
                if isinstance(student_logits, (tuple, list)):
                    student_logits = student_logits[0]

                # 蒸馏损失
                soft_targets = torch.softmax(teacher_logits / temperature, dim=-1)
                soft_prob = torch.log_softmax(student_logits / temperature, dim=-1)
                distill_loss = kl_loss(soft_prob, soft_targets) * (temperature ** 2)

                # 硬标签损失
                hard_loss = ce_loss(student_logits, labels)

                # 总损失
                loss = alpha * distill_loss + (1 - alpha) * hard_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
            train_losses.append(avg_loss)

            # 验证
            if val_loader:
                val_acc = evaluate_accuracy(student, val_loader)
                val_accuracies.append(val_acc)

        student.eval()

        rep = {
            "status": "ok",
            "mode": "real_data",
            "epochs": epochs,
            "temperature": float(temperature),
            "alpha": float(alpha),
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_val_accuracy": val_accuracies[-1] if val_accuracies else None,
            "train_losses": train_losses,
            "val_accuracies": val_accuracies,
        }
        write_report(artifacts_dir, rep, "distill_report.json")
        return rep

    except Exception as e:
        rep = {"status": "error", "reason": str(e)}
        write_report(artifacts_dir, rep, "distill_report.json")
        return rep


def _kd_with_random_data(
    student: Any,
    teacher: Any,
    temperature: float,
    alpha: float,
    steps: int,
    input_shape: Sequence[int],
    artifacts_dir: Optional[str],
) -> Dict[str, Any]:
    """使用随机数据进行蒸馏（演示模式）"""
    try:
        import torch
        from torch import nn

        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

        student.train()
        opt = torch.optim.SGD(student.parameters(), lr=1e-4, momentum=0.9)
        kldiv = nn.KLDivLoss(reduction="batchmean")

        steps = max(1, min(100, int(steps)))
        shape = tuple(int(v) for v in input_shape) if input_shape else (2, 3, 224, 224)

        for _ in range(steps):
            x = torch.randn(*shape)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            if isinstance(t_logits, (tuple, list)):
                t_logits = t_logits[0]
            if isinstance(s_logits, (tuple, list)):
                s_logits = s_logits[0]
            t_prob = torch.softmax(t_logits / temperature, dim=-1)
            s_log_prob = torch.log_softmax(s_logits / temperature, dim=-1)
            loss_kd = kldiv(s_log_prob, t_prob) * (temperature * temperature)
            opt.zero_grad()
            loss_kd.backward()
            opt.step()

        student.eval()

        rep = {
            "status": "ok",
            "mode": "random_data",
            "steps": steps,
            "temperature": float(temperature),
            "alpha": float(alpha),
        }
        write_report(artifacts_dir, rep, "distill_report.json")
        return rep
    except Exception:
        rep = {"status": "skipped", "reason": "exception"}
        write_report(artifacts_dir, rep, "distill_report.json")
        return rep

