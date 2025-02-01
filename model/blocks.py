from torch import nn
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

class LandscapeConvBlock(nn.Module):
    def __init__(self, in_channels=8, out_features=64):
        """
        Lightweight CNN branch for extracting spatial features from an 8-channel static map.

        Args:
            in_channels (int): Number of input channels (default: 8).
            out_features (int): Desired number of output feature channels.
                                If set to 64, the projection layer becomes an identity.
        """
        super(LandscapeConvBlock, self).__init__()

        # First convolutional block: 8 -> 16 channels.
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # Halve spatial dimensions.
        )

        # Second convolutional block: 16 -> 32 channels.
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # Halve spatial dimensions.
        )

        # Third convolutional block: 32 -> 64 channels.
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Optional projection layer to adjust channel dimension if needed.
        self.projection = nn.Conv2d(64, out_features, kernel_size=1) if out_features != 64 else nn.Identity()

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (Tensor): Input tensor of shape [B, in_channels, H, W].

        Returns:
            Tensor: Extracted spatial features of shape [B, out_features, H_out, W_out].
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.projection(x)
        return x

