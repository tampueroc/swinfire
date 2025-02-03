from torch import nn
from typing import Union, List
from einops.layers.torch import Rearrange

from .patch_layers import PatchMerging3D
from .blocks import ConvBlock, SwinBlock3D, ConditionalGate
from .utils import StaticProjector


class Encoder(nn.Module):
    def __init__(self, in_dims, hidden_dimension, layers, downscaling_factor, num_heads, head_dim,
                 window_size: Union[int, List[int]], relative_pos_embedding: bool = True, dropout: float = 0.0, static_channels: int = 0, expected_H: int = 0, expected_W: int = 0, wind_context_dim=64):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging3D(in_dim=in_dims, out_dim=hidden_dimension,
                                              downscaling_factor=downscaling_factor)
        self.conv_block = ConvBlock(in_ch=hidden_dimension, out_ch=hidden_dimension)

        self.re1 = Rearrange('b c h w d -> b h w d c')
        self.swin_layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.swin_layers.append(nn.ModuleList([
                SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            wind_context_dim=wind_context_dim, dropout=dropout),
                SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                            shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                            wind_context_dim=wind_context_dim, dropout=dropout)
            ]))
        self.re2 = Rearrange('b  h w d c -> b c h w d')

        self.static_projector = StaticProjector(
            in_channels=static_channels,
            out_channels=hidden_dimension,
            spatial_size=(expected_H, expected_W)  # Based on downscaling
        )
        self.gate = ConditionalGate(
            feat_dim=hidden_dimension,
            static_dim=hidden_dimension
        )

    def forward(self, x, static_data=None, wind_context=None):
        x = self.patch_partition(x)
        x2 = self.conv_block(x)  # Short dependencies

        x = self.re1(x)
        for regular_block, shifted_block in self.swin_layers:  # Long dependencies
            x = regular_block(x, wind_context)
            x = shifted_block(x, wind_context)
        x = self.re2(x)

        if static_data is not None:
            static_proj = self.static_projector(static_data)
            x = self.gate(x, static_proj)  # Fuse features

        x = x + x2  # Long and short length dependencies
        return x
