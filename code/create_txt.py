import os

def write_image_paths_to_txt(root_dir, output_txt, img_formats=None):
    """
    收集指定目录下所有图像文件的路径，写入txt文件
    :param root_dir: 要遍历的根目录（如./images）
    :param output_txt: 输出的txt文件路径（如./image_paths.txt）
    :param img_formats: 要收集的图片格式，默认包含常见格式
    """
    # 默认支持的图片格式（小写）
    if img_formats is None:
        img_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    
    # 校验目录是否存在
    if not os.path.exists(root_dir):
        print(f"❌ 错误：目录 {root_dir} 不存在！")
        return
    
    # 收集所有图片路径
    image_paths = []
    # 递归遍历目录（含子目录）
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            # 获取文件后缀（小写）
            file_suffix = os.path.splitext(file)[1].lower()
            if file_suffix in img_formats:
                # 拼接完整路径（可选：改为相对路径/绝对路径）
                full_path = os.path.abspath(os.path.join(root, file))  # 绝对路径
                # full_path = os.path.relpath(os.path.join(root, file))  # 相对路径（可选）
                image_paths.append(full_path)
    
    # 写入txt文件
    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write('\n'.join(image_paths))
    
    # 输出统计信息
    print(f"✅ 成功收集 {len(image_paths)} 张图片路径")
    print(f"📝 路径已写入：{os.path.abspath(output_txt)}")

if __name__ == "__main__":
    # ==================== 配置参数（修改这里！）====================
    ROOT_DIR = "/root/autodl-tmp/images/val"  # 你的图像根目录
    OUTPUT_TXT = "/root/autodl-tmp/val.txt"  # 输出的txt路径
    # ==============================================================
    
    # 执行收集
    write_image_paths_to_txt(ROOT_DIR, OUTPUT_TXT)
