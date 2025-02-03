from torch import nn
import torch
from typing import Union, List

from .utils import Residual3D, PreNorm3D, Norm
from .attention import WindowAttention3D, FeedForward3D

class SwinBlock3D(nn.Module):  # 不会改变输入空间分辨率
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size: Union[int, List[int]],
                 relative_pos_embedding: bool = True, dropout: float = 0.0):
        super().__init__()
        self.attention_block = Residual3D(PreNorm3D(dim, WindowAttention3D(dim=dim,
                                                                           heads=heads,
                                                                           head_dim=head_dim,
                                                                           shifted=shifted,
                                                                           window_size=window_size,
                                                                           relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual3D(PreNorm3D(dim, FeedForward3D(dim=dim, hidden_dim=mlp_dim, dropout=dropout)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()
        groups = min(in_ch, out_ch)
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=groups),
            Norm(dim=out_ch),
            nn.PReLU(),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=groups),
            Norm(dim=out_ch),
            nn.PReLU(),
        )

    def forward(self, x):
        x2 = x.clone()
        x = self.net(x) * x2
        return x


class ConditionalGate(nn.Module):
    """Dynamically fuses encoder features with static data."""
    def __init__(self, feat_dim, static_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv3d(feat_dim + static_dim, feat_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(feat_dim // 2, feat_dim, kernel_size=3, padding=1),
            nn.Sigmoid()  # Outputs gating mask ∈ [0,1]
        )

    def forward(self, x, static_proj):
        # x: [B, C, H, W] (encoder features)
        # static_proj: [B, C_static_proj, H, W]
        static_proj = static_proj.unsqueeze(-1).expand(-1, -1, -1, -1, x.size(-1))
        fused = torch.cat([x, static_proj], dim=1)
        gate = self.fusion(fused)
        return x * gate + static_proj * (1 - gate)  # Blended features
