import os
import cv2
import torch
import yaml
import numpy as np
from ultralytics import RTDETR
from pathlib import Path

# -------------------------- 1. 配置参数 -------------------------- 
MODEL_PATH = './best.pt'
DATA_YAML_PATH = 'data-vis.yaml'
OUTPUT_DIR = 'custom_test_results'
IMGSZ = 640
CONF_THRESHOLD = 0.5
BOX_LINE_WIDTH = 1  # 调小宽度
COLOR = (0, 255, 0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -------------------------- 2. 加载模型 -------------------------- 
# 加载 RT-DETR 对象
rt_model = RTDETR(MODEL_PATH) 
# 获取底层的 PyTorch 模型并设为评估模式
model = rt_model.model.to(DEVICE)
model.eval()

# 获取类别名
with open(DATA_YAML_PATH, 'r', encoding='utf-8') as f: 
    data_config = yaml.safe_load(f) 
class_names = data_config.get('names', []) 
test_img_dir = data_config.get('test', '') 

os.makedirs(OUTPUT_DIR, exist_ok=True) 

# -------------------------- 3. 手动预处理与推理函数 -------------------------- 

@torch.no_grad()
def manual_predict(img_cv2, imgsz, conf_thres):
    # 1. 预处理: BGR -> RGB, Resize, Normalize
    h, w = img_cv2.shape[:2]
    img = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (imgsz, imgsz))
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(DEVICE).float()
    img_tensor /= 255.0  # 归一化
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dim

    # 2. 推理 (直接调用底层网络，避开 embed 报错)
    preds = model(img_tensor)
    
    # 3. 解析结果 (RT-DETR 默认输出为 [batch, 300, 4+类别数])
    # 不同的版本输出结构可能略有不同，通常 preds 是一个 Tensor
    if isinstance(preds, (list, tuple)):
        preds = preds[0]
        
    results = []
    # 假设输出 shape [1, 300, 4+num_classes]
    # RT-DETR 输出通常包含了 scores 和 boxes
    output = preds[0].cpu().numpy() # [300, 84] (对于COCO)
    
    for row in output:
        # 后 80/n 列是类别分数
        scores = row[4:]
        cls_id = np.argmax(scores)
        conf = scores[cls_id]
        
        if conf > conf_thres:
            # 前 4 列是坐标 (cx, cy, w, h) 且是归一化的
            cx, cy, nw, nh = row[:4]
            # 转换为 xyxy 格式并还原到原图尺寸
            x1 = int((cx - nw/2) * w)
            y1 = int((cy - nh/2) * h)
            x2 = int((cx + nw/2) * w)
            y2 = int((cy + nh/2) * h)
            results.append({
                'box': [x1, y1, x2, y2],
                'conf': conf,
                'cls': cls_id
            })
    return results

# -------------------------- 4. 绘图函数 -------------------------- 

def custom_draw(image, detections, class_names):
    img_draw = image.copy()
    for det in detections:
        box = det['box']
        conf = det['conf']
        cls_id = det['cls']
        name = class_names[cls_id] if cls_id < len(class_names) else f"ID:{cls_id}"

        # 绘制框
        cv2.rectangle(img_draw, (box[0], box[1]), (box[2], box[3]), COLOR, BOX_LINE_WIDTH)

        # 绘制极小标签 (字体大小 0.3)
        label_text = f"{name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3 
        thickness = 1
        cv2.putText(img_draw, label_text, (box[0], box[1] - 5), font, font_scale, (255, 255, 255), thickness)
        
    return img_draw

# -------------------------- 5. 遍历处理 -------------------------- 

test_img_paths = list(Path(test_img_dir).glob('*'))
print(f"开始处理 {len(test_img_paths)} 张图片...")

for img_path in test_img_paths:
    if img_path.suffix.lower() not in ['.jpg', '.png', '.jpeg']: continue
    
    img = cv2.imread(str(img_path))
    if img is None: continue
    
    # 执行手动预测
    detections = manual_predict(img, IMGSZ, CONF_THRESHOLD)
    
    # 绘图
    res_img = custom_draw(img, detections, class_names)
    
    # 保存
    cv2.imwrite(os.path.join(OUTPUT_DIR, img_path.name), res_img)
    print(f"已保存: {img_path.name}")

print(f"全部完成，保存在 {OUTPUT_DIR}")