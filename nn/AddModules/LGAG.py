import torch.nn as nn

def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    # activation layer
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    return layer

class LGAG(nn.Module):
    def __init__(self, F_int, kernel_size=3, groups=1, activation='relu'):
        super(LGAG,self).__init__()

        if kernel_size == 1:
            groups = 1

        self.W_g = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_fusion = nn.Sequential(
            nn.Conv2d(F_int, F_int, kernel_size=kernel_size, stride=1, padding=kernel_size//2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.activation = act_layer(activation, inplace=True)

                
    def forward(self, data):

        if isinstance(data, (list, tuple)) and len(data) == 2:
            x, y = data
        else:
            raise ValueError("LGAG expects a list or tuple of two feature maps")
        
        x1 = self.W_g(x)
        y1 = self.W_x(y)
        
        fused = x1 + y1
        fused = self.W_fusion(fused)
        
        psi = self.activation(fused)
        psi = self.psi(psi)
        
        result = x * psi + y * (1 - psi)
        return result



