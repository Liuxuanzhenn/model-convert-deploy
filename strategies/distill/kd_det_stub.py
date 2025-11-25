"""检测蒸馏占位实现（中文注释）。

说明：
- 由于检测 KD 需要数据集、匹配策略与损失构造等较多依赖，此处提供占位实现：
  写入 distill_report.json 并返回 `{"status":"skipped","reason":"det_stub"}`。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


def run_stub(artifacts_dir: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    rep = {"status": "skipped", "reason": "det_stub"}
    if extra:
        rep.update({k: v for k, v in extra.items() if k not in rep})
    if artifacts_dir:
        try:
            os.makedirs(artifacts_dir, exist_ok=True)
            p = os.path.join(artifacts_dir, "distill_report.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(rep, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return rep

