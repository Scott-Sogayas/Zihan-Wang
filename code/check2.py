import os

# 1. 设置你的标签文件夹路径
labels_path = '/root/autodl-tmp/labels/val'  # 请替换为你的实际路径

def change_class_to_3(folder_path):
    # 获取文件夹下所有文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    
    count = 0
    for filename in files:
        file_full_path = os.path.join(folder_path, filename)
        
        # 读取原始数据
        with open(file_full_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 使用空格分割每一行
            parts = line.split()
            
            # 将第一列修改为 '3'
            parts[0] = '3'
            
            # 重新组合成字符串
            new_lines.append(" ".join(parts))
            
        # 写回文件（覆盖原文件）
        with open(file_full_path, 'w') as f:
            f.write("\n".join(new_lines) + "\n")
            
        count += 1
        if count % 100 == 0:
            print(f"已处理 {count} 个文件...")

    print(f"处理完成！共修改了 {count} 个标签文件。")

# 执行任务
if __name__ == "__main__":
    change_class_to_3(labels_path)
