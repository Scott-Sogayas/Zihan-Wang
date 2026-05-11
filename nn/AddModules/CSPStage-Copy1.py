import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ———————————————————————————————————— 
# 1. 替换原本的 CoordAtt 为 EMA (Efficient Multi-Scale Attention)
# ———————————————————————————————————— 
class EMA(nn.Module):
    def __init__(self, channels, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((None, 1)), 
            nn.AdaptiveAvgPool2d((1, None))
        ])
        self.conv1x1 = nn.Conv2d(channels // factor, channels // factor, kernel_size=1)
        self.conv3x3 = nn.Conv2d(channels // factor, channels // factor, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        group_c = c // self.groups
        x_group = x.view(b * self.groups, group_c, h, w)
        
        # 支路 1: 全局特征提取
        y = self.avg_pool(x_group)
        y = self.conv1x1(y)
        
        # 支路 2: 空间多尺度编码 (类似 CoordAtt 但在组内进行)
        x_h = self.pools[0](x_group)
        x_w = self.pools[1](x_group).permute(0, 1, 3, 2)
        y_hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(y_hw, [h, w], dim=2)
        
        # 交互加权
        out = x_group * self.sigmoid(y) * self.sigmoid(x_h) * self.sigmoid(x_w.permute(0, 1, 3, 2))
        return out.view(b, c, h, w)

# ———————————————————————————————————— 
# 2. 基础组件保持不变 (SPPF, ConvBNAct, RepConv, BasicBlock)
# ———————————————————————————————————— 
class SPPF(nn.Module): 
    def __init__(self, in_channels, out_channels, k=5, act='silu'): 
        super().__init__() 
        c_ = in_channels // 2
        self.conv1 = ConvBNAct(in_channels, c_, 1, 1, act=act) 
        self.conv2 = ConvBNAct(c_ * 4, out_channels, 1, 1, act=act) 
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2) 

    def forward(self, x): 
        x = self.conv1(x) 
        y1 = self.m(x) 
        y2 = self.m(y1) 
        y3 = self.m(y2) 
        return self.conv2(torch.cat((x, y1, y2, y3), 1)) 

def get_activation(name='silu', inplace=True): 
    if name == 'silu': return nn.SiLU(inplace=inplace) 
    if name == 'relu': return nn.ReLU(inplace=inplace) 
    return nn.Identity() 

class ConvBNAct(nn.Module): 
    def __init__(self, in_channels, out_channels, ksize, stride=1, groups=1, bias=False, act='silu'): 
        super().__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride, (ksize-1)//2, groups=groups, bias=bias) 
        self.bn = nn.BatchNorm2d(out_channels) 
        self.act = get_activation(act) 

    def forward(self, x): 
        return self.act(self.bn(self.conv(x))) 

class RepConv(nn.Module): 
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, deploy=False, act='relu'): 
        super(RepConv, self).__init__() 
        self.deploy = deploy
        self.nonlinearity = get_activation(act) 
        if deploy: 
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=True) 
        else: 
            self.rbr_dense = nn.Sequential( 
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False), 
                nn.BatchNorm2d(out_channels) 
            ) 
            self.rbr_1x1 = nn.Sequential( 
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, groups=groups, bias=False), 
                nn.BatchNorm2d(out_channels) 
            ) 
            self.rbr_identity = nn.BatchNorm2d(out_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, x): 
        if hasattr(self, 'rbr_reparam'): return self.nonlinearity(self.rbr_reparam(x)) 
        id_out = self.rbr_identity(x) if self.rbr_identity is not None else 0
        return self.nonlinearity(self.rbr_dense(x) + self.rbr_1x1(x) + id_out) 

class BasicBlock_3x3_Reverse(nn.Module): 
    def __init__(self, ch_in, ch_hidden_ratio, ch_out, act='relu'): 
        super().__init__() 
        ch_hidden = int(ch_in * ch_hidden_ratio) 
        self.conv2 = RepConv(ch_in, ch_hidden, 3, act=act) 
        self.conv1 = ConvBNAct(ch_hidden, ch_out, 3, act=act) 

    def forward(self, x): 
        return x + self.conv1(self.conv2(x)) 

# ———————————————————————————————————— 
# 3. 更新后的 CSPStage (集成 EMA) 
# ———————————————————————————————————— 
class CSPStage(nn.Module): 
    def __init__(self, ch_in, ch_out, n=1, ch_hidden_ratio=0.5, act='silu', spp=False): 
        super(CSPStage, self).__init__() 
        ch_mid = ch_out // 2
        
        # 支路 1
        self.conv1 = ConvBNAct(ch_in, ch_mid, 1, act=act) 
        # 支路 2
        self.conv2 = ConvBNAct(ch_in, ch_mid, 1, act=act) 
        
        self.m = nn.Sequential(*[ 
            BasicBlock_3x3_Reverse(ch_mid, ch_hidden_ratio, ch_mid, act=act) for _ in range(n) 
        ]) 
        
        self.spp = SPPF(ch_mid, ch_mid, k=5, act=act) if spp else nn.Identity() 
        
        # 融合层
        self.conv3 = ConvBNAct(ch_mid * 2, ch_out, 1, act=act) 
        
        # ————————————————————————————————————
        # 修改点：将 self.ca 换成 self.ema
        # 注意：EMA 的 factor 需要能被通道数整除，默认 32 适用于大多数 128/256/512 通道
        # ————————————————————————————————————
        self.ema = EMA(ch_out, factor=32 if ch_out >= 32 else 8) 

    def forward(self, x): 
        y1 = self.conv1(x) 
        y2 = self.spp(self.m(self.conv2(x))) 
        
        y = torch.cat((y1, y2), dim=1) 
        y = self.conv3(y) 
        
        # 使用 EMA
        return self.ema(y)