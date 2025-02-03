from torch import nn
import torch
from typing import Union, List
from einops import rearrange

from .utils import Residual3D, PreNorm3D, Norm
from .attention import WindowAttention3D, FeedForward3D

class SwinBlock3D(nn.Module):  # 不会改变输入空间分辨率
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size: Union[int, List[int]],
                 relative_pos_embedding: bool = True, dropout: float = 0.0, wind_context_dim=False):
        super().__init__()
        self.attention_block = Residual3D(PreNorm3D(dim, WindowAttention3D(dim=dim,
                                                                           heads=heads,
                                                                           head_dim=head_dim,
                                                                           shifted=shifted,
                                                                           window_size=window_size,
                                                                           relative_pos_embedding=relative_pos_embedding)))

        # 2. Cross-Attention to Wind Context
        if wind_context_dim is not False:
            self.cross_norm = nn.LayerNorm(dim)  # Added normalization
            self.cross_attn = nn.MultiheadAttention(  # No Residual3D wrapper
                embed_dim=dim,
                kdim=wind_context_dim,
                vdim=wind_context_dim,
                num_heads=heads,
                dropout=dropout,
                batch_first=True
            )
        self.mlp_block = Residual3D(PreNorm3D(dim, FeedForward3D(dim=dim, hidden_dim=mlp_dim, dropout=dropout)))

    def forward(self, x, wind_context=False):
        # Standard Swin Transformer flow
        x = self.attention_block(x)

        # Add wind context cross-attention
        B, H, W, D, C = x.shape
        x_flat = rearrange(x, 'b h w d c -> b (h w d) c')

        if wind_context is not False:
            # Apply normalization and attention
            attn_out, _ = self.cross_attn(
                query=self.cross_norm(x_flat),
                key=wind_context,
                value=wind_context
            )
            x = x + rearrange(attn_out, 'b (h w d) c -> b h w d c', h=H, w=W, d=D)

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

class WindContextEncoder(nn.Module):
    def __init__(self, in_dim=2, hidden_dim=64, num_heads=4):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True
            ),
            num_layers=2
        )

    def forward(self, wind_input):
        # wind_input: [B, T, 3]
        x = wind_input.permute(0, 2, 1)  # Swap time and channel dims
        x = self.proj(x)  # [B, T, hidden_dim]
        return self.transformer(x)  # [B, T, hidden_dim]
