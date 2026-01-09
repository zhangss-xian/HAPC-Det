import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class HSAM(nn.Module):
    """
    Hierarchical Scale Attention Module (HSAM)
    Structure: Local branch (Conv) + Global branch (Coordinate Attention) -> Fusion
    """
    def __init__(self, c1, c2=None):
        super().__init__()
        # 如果没有指定 c2，则默认保持通道数不变
        c_out = c2 if c2 is not None else c1
        
        # === 1. Local Structure Enhancement Path ===
        # 保持通道数 c1 -> c1 -> c1
        self.local_branch = nn.Sequential(
            Conv(c1, c1, 1, 1), # 1x1 Conv
            Conv(c1, c1, 3, 1)  # 3x3 Conv
        )
        
        # === 2. Global Spatial Attention Path (Coordinate Attention) ===
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, c1 // 32) # reduction ratio usually 32
        
        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish() 
        
        self.conv_h = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)
        
        # === 3. Fusion ===
        # Concat (Input C + Input C = 2C) -> Conv 1x1 -> Output C
        self.final_conv = Conv(c1 * 2, c_out, 1, 1)

    def forward(self, x):
        # Local Branch
        x_local = self.local_branch(x)
        
        # Global Branch (Coordinate Attention)
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        x_global = identity * a_w * a_h
        
        # Fusion (Concat + Conv)
        return self.final_conv(torch.cat((x_local, x_global), dim=1))
