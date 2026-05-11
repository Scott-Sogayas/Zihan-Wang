import os
import glob
from pathlib import Path

def get_all_empty_labels(labels_dir):
    """
    自动扫描labels目录下所有空标签文件（大小为0或无有效内容）
    Args:
        labels_dir: 标签文件夹路径
    Returns:
        所有空标签文件的完整路径列表
    """
    empty_files = []
    for txt_file in glob.glob(os.path.join(labels_dir, "*.txt")):
        # 条件1：文件大小为0
        if os.path.getsize(txt_file) == 0:
            empty_files.append(txt_file)
            continue
        # 条件2：文件有内容但全是空行
        with open(txt_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
            if not any(lines):
                empty_files.append(txt_file)
    return empty_files

# 配置路径
labels_dir = "./labels/train"
images_dir = "./images/train"

# 自动获取所有空标签文件（精准83个）
empty_label_files = get_all_empty_labels(labels_dir)
print(f"✅ 自动扫描到 {len(empty_label_files)} 个空标签文件")

# 遍历删除标签+对应图片
deleted_count = 0
for label_path in empty_label_files:
    # 1. 删除标签文件
    if os.path.exists(label_path):
        os.remove(label_path)
        print(f"已删除标签文件：{label_path}")
    
    # 2. 删除对应图片文件（匹配所有格式）
    img_name = Path(label_path).stem
    # 查找所有格式的图片
    img_patterns = [
        os.path.join(images_dir, f"{img_name}.jpg"),
        os.path.join(images_dir, f"{img_name}.png"),
        os.path.join(images_dir, f"{img_name}.jpeg"),
        os.path.join(images_dir, f"{img_name}.bmp"),
    ]
    for img_path in img_patterns:
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"已删除图片文件：{img_path}")
            deleted_count += 1

print(f"\n✅ 共删除 {deleted_count} 个文件（标签+图片）")

# 验证删除结果
remaining_empty = get_all_empty_labels(labels_dir)
print(f"剩余未删除的空标签文件数：{len(remaining_empty)}")