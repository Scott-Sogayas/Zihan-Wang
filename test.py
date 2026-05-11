from ultralytics import RTDETR

# 1. 加载你训练好的最优模型
model = RTDETR('./best.pt') 

# 2. 在测试集上进行验证
metrics = model.val(
    data='data-vis.yaml',    # 指向你的数据配置文件
    split='val',            # 关键：手动指定运行测试集（默认是 val）
    imgsz=640,               # 保持与训练时一致的分辨率
    batch=8,                # 根据显存调整
    conf=0.001,              # 算 mAP 建议设低一点，保证更多框参与计算
    iou=0.6                  # NMS 阈值
)

# 3. 打印结果
print(f"测试集 mAP@0.5: {metrics.seg.map50 if 'seg' in dir(metrics) else metrics.box.map50}")
print(f"测试集 mAP@0.5:0.95: {metrics.seg.map if 'seg' in dir(metrics) else metrics.box.map}")