import torch
import torch.nn as nn
from ultralytics.nn.modules.conv import Conv

class ECA(nn.Module):
    """Efficient Channel Attention (Used inside PCFB)"""
    def __init__(self, c, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        # view/squeeze dimensions to fit Conv1d
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class PCFB(nn.Module):
    """
    Parallel Context Fusion Block (PCFB).
    Parallel MaxPool (1x1, 3x3, 5x5) -> Concat -> Conv -> ECA.
    """
    def __init__(self, c1, c2, k=5): 
        super().__init__()
        self.c_in = c1
        
        # Step 1: Channel Compression
        # 这里的 c_mid 策略可以根据需要调整，通常减半或者保持
        c_mid = c2 // 2 if c2 > 256 else c2 
        
        self.conv_compress = Conv(c1, c_mid, 1, 1) # F0
        
        # Step 2: Parallel Multi-resolution
        # MaxPool 1x1 is logically identity, so we just use the feature map itself
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        
        # Step 3: Aggregation
        # Concat F0, F1(Identity), F2(Pool3), F3(Pool5) -> 4 * c_mid
        self.conv_fuse = nn.Sequential(
            Conv(c_mid * 4, c_mid, 1, 1),
            Conv(c_mid, c2, 3, 1) # Output matches c2
        )
        
        # Step 4: Efficient Channel Attention (ECA)
        self.eca = ECA(c2, k_size=3) 

    def forward(self, x):
        f0 = self.conv_compress(x)
        
        # Parallel branches
        f1 = f0 # F1 is Identity (Maxpool 1x1)
        f2 = self.pool3(f0)
        f3 = self.pool5(f0)
        
        # Concat
        f_cat = torch.cat([f0, f1, f2, f3], dim=1)
        
        # Fuse
        f_fused = self.conv_fuse(f_cat)
        
        # ECA Refinement
        return self.eca(f_fused)
