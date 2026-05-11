import argparse
import torch
import os
from ultralytics import RTDETR
from ultralytics.utils.plotting import plot_results
import os
import torch.distributed as dist

def main(opt): 
    custom_yaml="/root/miniconda3/lib/python3.12/site-packages/ultralytics/cfg/models/rt-detr/rtdetr-efficient.yaml"
    if not os.path.exists(custom_yaml): 
        raise FileNotFoundError(f"模型配置文件不存在：{custom_yaml}") 
    
    print(f" 使用自定义模型配置：{custom_yaml}") 
    torch.cuda.empty_cache() 
    
    # 逻辑调整：直接加载预训练权重训练，或者从yaml开始
    # 如果你有 best.pt，建议直接从权重开始，或者使用 model = RTDETR('best.pt') 续训
    model = RTDETR(custom_yaml)

    print("="*50) 
    print("【分类标签检查】") 
    if hasattr(model, 'names'): 
        for i, name in model.names.items(): 
            print(f"标签 ID {i}: {name}") 
    print("="*50) 

    # 2. 按新参数配置开始训练
    results = model.train( 
        data='data-vis.yaml',      # 数据配置文件路径
        epochs=260,            # 【建议】曲线还在涨，增加到300轮
        imgsz=640, 
        batch=4,              # 【建议】双卡训练，batch调至16或更高
        workers=8,           
        optimizer='AdamW', 
        amp=False,              # 【建议】开启混合精度，提速且省显存
        lr0=0.0001, 
        lrf=0.1, 
        momentum=0.9, 
        weight_decay=0.0001, 
        warmup_epochs=3.0, 
        warmup_momentum=0.8, 
        warmup_bias_lr=0.1, 
        box=5.0, 
        cls=1.0, 
        dfl=1.5, 
        label_smoothing=0.0, 
        
        nbs=64, 
        # 数据增强 (基于RT-DETR特性)
        hsv_h=0.015, 
        hsv_s=0.7, 
        hsv_v=0.4, 
        degrees=0.0, 
        translate=0.1, 
        scale=0.5, 
        shear=0.0, 
        perspective=0.0, 
        flipud=0.0, 
        fliplr=0.5, 
        mosaic=0.0,  # RT-DETR 开启mosaic通常有效
        mixup=0.2,   # 【建议】数据量大时，开启微小mixup增强泛化
        copy_paste=0.0, 
        auto_augment='randaugment', # 使用 RandAugment 提升鲁棒性
        erasing=0.1, 
        crop_fraction=1.0, 
        # 监控参数
        pretrained=True, 
        patience=15, # 既然Epoch多了，早停的阈值也要调高
        save=True, 
        val=True, 
        plots=True, 
        deterministic=False, # 分布式训练通常设为False以提高效率
    ) 

    # 3. 保存与可视化
    try:
        train_dir = results.save_dir
        plot_results(os.path.join(train_dir, 'results.csv')) 
        print(f" 指标曲线已保存到：{os.path.join(train_dir, 'results.png')}") 
    except Exception as e:
        print(f"绘图出错（可能是多卡环境同步问题），手动前往 {results.save_dir} 查看。")

def parse_opt(known=False): 
    parser = argparse.ArgumentParser() 
    # 注意修改为你实际的 yaml 路径
    parser.add_argument('--cfg', type=str, default='/root/miniconda3/lib/python3.12/site-packages/ultralytics/cfg/models/rt-detr/rtdetr-efficient.yaml', help='模型配置') 
    opt = parser.parse_known_args()[0] if known else parser.parse_args() 
    return opt

if __name__ == "__main__": 
    opt = parse_opt() 
    main(opt)