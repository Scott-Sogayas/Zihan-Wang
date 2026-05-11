from ultralytics import RTDETR
import numpy as np
from ultralytics.utils.metrics import ap_per_class

# 1. 加载模型并执行验证（保留原始预测/标注数据）
model = RTDETR('/root/autodl-tmp/runs-visdrone/detect/final2/weights/best.pt') 

# 关键：设置 save_json=True 保留预测结果，方便后续拆解计算
metrics = model.val(
    data='car.yaml',
    split='test',
    imgsz=640,
    batch=8,
    conf=0.6,
    iou=0.3,
    verbose=True,
    save_json=True  # 保留原始预测数据
)
