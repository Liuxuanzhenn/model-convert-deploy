"""文件工具

合并 file_utils.py 和 json_utils.py
"""
import os
import json
from typing import Any, List


def ensure_dir(p: str) -> None:
    """确保目录存在"""
    os.makedirs(p, exist_ok=True)


def ensure_json_array(path: str) -> None:
    """确保 JSON 数组文件存在"""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("[]")


def read_json_list(path: str) -> List[Any]:
    """读取 JSON 数组文件"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def write_json(path: str, obj: Any) -> None:
    """写入 JSON 文件"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

