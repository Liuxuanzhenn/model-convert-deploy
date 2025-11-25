"""检查文件大小"""
import os
from pathlib import Path

def count_lines(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except:
        return 0

def main():
    base_dir = Path(__file__).parent.parent
    files = []
    
    for root, dirs, fs in os.walk(base_dir):
        # 跳过一些目录
        if '__pycache__' in root or '.git' in root or 'artifacts' in root:
            continue
            
        for f in fs:
            if f.endswith('.py'):
                file_path = Path(root) / f
                lines = count_lines(file_path)
                if lines > 0:
                    rel_path = file_path.relative_to(base_dir)
                    files.append((rel_path, lines))
    
    files.sort(key=lambda x: x[1], reverse=True)
    
    print("=" * 80)
    print("文件行数统计（Top 20）")
    print("=" * 80)
    print(f"{'行数':<8} {'文件路径'}")
    print("-" * 80)
    
    for path, lines in files[:20]:
        print(f"{lines:<8} {path}")
    
    print("\n" + "=" * 80)
    print("大文件分析（>300行）")
    print("=" * 80)
    large_files = [f for f in files if f[1] > 300]
    for path, lines in large_files:
        print(f"{lines:<8} {path}")

if __name__ == '__main__':
    main()

