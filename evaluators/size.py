"""目录体积评估工具（中文注释）。

说明：
- 递归统计目录下所有文件字节数并换算为 MB，供 metrics.json 写入使用。
"""

import os


def dir_size_mb(path: str) -> float:
    total = 0
    if not os.path.isdir(path):
        return 0.0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root, fn))
            except Exception:
                pass
    return round(total / (1024 * 1024), 4)

