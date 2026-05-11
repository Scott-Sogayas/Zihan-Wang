import os
from pathlib import Path
import argparse
from PIL import Image
from tqdm import tqdm

# 核心工具函数：VisDrone像素框 → YOLO归一化xywh框（不变，核心转换逻辑）
def convert_box(size, box):
    dw = 1. / size[0]  # 宽度归一化系数（1/图片宽）
    dh = 1. / size[1]  # 高度归一化系数（1/图片高）
    # 转换公式：左上角(x1,y1) → 中心(xy)，再归一化宽高
    return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

# 主转换函数：适配你的结构（标注在labels/train/val，图片在images/train/val）
def visdrone2yolo(root_dir, label_save_root, subset, img_suffix='.jpg'):
    """
    :param root_dir: 数据集总根路径（包含images/labels目录）
    :param label_save_root: 转换后YOLO标签的生成根路径
    :param subset: 子集名（仅train/val）
    :param img_suffix: 图片后缀，默认.jpg
    """
    # 你的原始路径：标注在labels/subset，图片在images/subset
    src_label_dir = Path(root_dir) / 'labels' / subset  # 原VisDrone格式标注
    img_dir = Path(root_dir) / 'images' / subset        # 对应图片
    # 转换后标签保存路径：指定根路径/subset（如/root/autodl-tmp/yolo_labels/train）
    dst_label_dir = Path(label_save_root) / subset
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    print(f"👉 开始处理[{subset}]：原标注={src_label_dir} | 图片={img_dir} | 新标签={dst_label_dir}")

    # 基础路径检查
    if not src_label_dir.exists():
        print(f"【错误】原标注目录不存在：{src_label_dir}，跳过[{subset}]！")
        return
    if not img_dir.exists():
        print(f"【错误】图片目录不存在：{img_dir}，跳过[{subset}]！")
        return

    # 遍历所有原标注txt文件
    src_label_files = list(src_label_dir.glob('*.txt'))
    if not src_label_files:
        print(f"【警告】原标注目录[{src_label_dir}]无txt文件，跳过[{subset}]！")
        return
    pbar = tqdm(src_label_files, desc=f'转换 {subset} 子集')

    for f in pbar:
        # 拼接对应图片路径（标注名 + 图片后缀）
        img_name = f.name.replace('.txt', img_suffix)
        img_path = img_dir / img_name
        if not img_path.exists():
            tqdm.write(f"【跳过】标注{f.name}无对应图片：{img_name}")
            continue

        # 读取图片尺寸（宽, 高），做异常处理
        try:
            with Image.open(img_path) as img:
                img_w, img_h = img.size
        except Exception as e:
            tqdm.write(f"【跳过】读取图片{img_name}失败：{str(e)[:50]}")
            continue

        # 解析原VisDrone标注（逗号分隔），转换为YOLO格式
        yolo_lines = []
        with open(f, 'r', encoding='utf-8') as file:
            # 逐行解析，过滤空行
            for row in [x.strip().split(',') for x in file.readlines() if x.strip()]:
                # 过滤无效标注：字段不足8个/忽略状态=0（VisDrone规则）
                if len(row) < 8 or row[4] == '0':
                    continue
                # 提取x1,y1,w,h（像素）和类别id，做类型转换异常处理
                try:
                    x1, y1, w, h = map(int, row[:4])  # VisDrone前4位：x1,y1,w,h（像素）
                    cls_vis = int(row[5])             # VisDrone类别id（从1开始）
                except ValueError:
                    tqdm.write(f"【跳过】{f.name}解析失败：{','.join(row)}")
                    continue
                # 类别id修正：1→0（YOLO要求从0开始）
                cls_yolo = cls_vis - 1
                # 转换为YOLO归一化框
                yolo_box = convert_box((img_w, img_h), (x1, y1, w, h))
                # 拼接YOLO行：类别id 中心x 中心y 宽 高（保留6位小数）
                yolo_lines.append(f"{cls_yolo} {' '.join(f'{x:.6f}' for x in yolo_box)}\n")

        # 写入转换后的YOLO标签文件
        dst_label_file = dst_label_dir / f.name
        with open(dst_label_file, 'w', encoding='utf-8') as fl:
            fl.writelines(yolo_lines)
        pbar.set_postfix(当前文件=f.name)  # 进度条显示当前处理的文件

    print(f"✅ [{subset}]转换完成！共处理{len(src_label_files)}个标注，新标签保存至：{dst_label_dir}\n")

if __name__ == '__main__':
    # 命令行参数解析：极简配置，适配你的使用场景
    parser = argparse.ArgumentParser(description='适配labels/train/val结构：VisDrone转YOLO标签')
    parser.add_argument('--root_dir', type=str,
                        default='/root/autodl-tmp',  # 你的数据集总根路径（Autodl环境）
                        help='数据集总根路径（包含images和labels目录）')
    parser.add_argument('--label_save_root', type=str,
                        default='/root/autodl-tmp/yolo_labels',  # 转换后标签的生成路径
                        help='转换后YOLO标签的生成根路径（自动分train/val）')
    parser.add_argument('--img_suffix', type=str,
                        default='.jpg',
                        help='图片文件后缀，如.jpg/.png/.jpeg')
    args = parser.parse_args()

    # 检查数据集总根路径
    if not Path(args.root_dir).exists():
        print(f"【致命错误】数据集总根路径不存在：{args.root_dir}")
        exit(1)

    for subset in ['test']: 
        visdrone2yolo(root_dir=args.root_dir, 
                      label_save_root=args.label_save_root, 
                      subset=subset, 
                      img_suffix=args.img_suffix) 

    print(f"🎉 测试集（test）转换完成！最终YOLO标签根路径：{args.label_save_root}")