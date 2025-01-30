from torch import nn
import numpy as np

from .utils import Norm

class PatchMerging3D(nn.Module):
    def __init__(self, in_dim, out_dim, downscaling_factor):
        super().__init__()
        downscaling_factor = (downscaling_factor, downscaling_factor, 1)
        self.net = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=downscaling_factor, stride=downscaling_factor),
            Norm(dim=out_dim),
        )

    def forward(self, x):
        # x: B, C, H, W, D
        x = self.net(x)
        return x  # B,  H //down_scaling, W//down_scaling, D//down_scaling, out_dim


class PatchExpanding3D(nn.Module):
    def __init__(self, in_dim, out_dim, up_scaling_factor):
        super(PatchExpanding3D, self).__init__()
        upscaling_factor = (up_scaling_factor, up_scaling_factor, 1)
        stride = upscaling_factor
        kernel_size = upscaling_factor
        padding = (np.array(kernel_size) - np.array(stride)) // 2
        self.net = nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            Norm(out_dim),
        )

    def forward(self, x):
        '''X: B,C,X,Y,Z'''
        x = self.net(x)
        return x




class FinalExpand3D(nn.Module):  # 体素最终分类时使用
    def __init__(self, in_dim, out_dim, up_scaling_factor):  # stl为second_to_last的缩写
        super(FinalExpand3D, self).__init__()

        upscaling_factor = (up_scaling_factor, up_scaling_factor, 1)
        stride = upscaling_factor
        kernel_size = upscaling_factor
        padding = (np.array(kernel_size) - np.array(stride)) // 2
        self.net = nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
            Norm(out_dim),
            nn.PReLU()
        )

    def forward(self, x):
        '''X: B,C,H,W,D'''
        x = self.net(x)
        return x
