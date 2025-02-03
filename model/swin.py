from torch import nn
from typing import List, Union
import torch
import numpy as np
import torch.optim as optim
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F

from .encoder import Encoder
from .decoder import Decoder, Converge
from .patch_layers import FinalExpand3D
from .losses import WeightedFocalLoss
from timm.layers import trunc_normal_

class SwinUnet3D(pl.LightningModule):
    def __init__(self, *, hidden_dim, layers, heads, in_channel=1, num_classes=2, head_dim=32,
                 window_size: Union[int, List[int]] = 4, downscaling_factors=(4, 2, 2, 2),
                 relative_pos_embedding=True, dropout: float = 0.0, skip_style='stack',
                 stl_channels: int = 32, learning_rate: float = 3e-4, loss_fn: str = "bce"):  # second_to_last_channels
        super().__init__()
        self.save_hyperparameters('loss_fn')

        example_shape = (4, in_channel, 512, 512, 4)  # Batch size 1, example spatial size, temporal depth
        self.example_input_array = torch.randn(example_shape, dtype=torch.float32)

        self.learning_rate = learning_rate
         # Metrics for training
        self.train_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.train_precision = torchmetrics.classification.BinaryPrecision()
        self.train_recall = torchmetrics.classification.BinaryRecall()
        self.train_f1 = torchmetrics.classification.BinaryF1Score()

        # Metrics for training
        self.val_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.val_precision = torchmetrics.classification.BinaryPrecision()
        self.val_recall = torchmetrics.classification.BinaryRecall()
        self.val_f1 = torchmetrics.classification.BinaryF1Score()

        # Loss
        if loss_fn == "bce":
            self.loss_fn = F.binary_cross_entropy_with_logits
        elif loss_fn == "focal_loss":
            self.loss_fn = WeightedFocalLoss(
                    alpha=0.95,
                    gamma=2
            )

        self.dsf = downscaling_factors
        self.window_size = window_size

        self.enc12 = Encoder(in_dims=in_channel, hidden_dimension=hidden_dim, layers=layers[0],
                             downscaling_factor=downscaling_factors[0], num_heads=heads[0],
                             head_dim=head_dim, window_size=window_size, dropout=dropout,
                             relative_pos_embedding=relative_pos_embedding)
        self.enc3 = Encoder(in_dims=hidden_dim, hidden_dimension=hidden_dim * 2,
                            layers=layers[1],
                            downscaling_factor=downscaling_factors[1], num_heads=heads[1],
                            head_dim=head_dim, window_size=window_size, dropout=dropout,
                            relative_pos_embedding=relative_pos_embedding)
        self.enc4 = Encoder(in_dims=hidden_dim * 2, hidden_dimension=hidden_dim * 4,
                            layers=layers[2],
                            downscaling_factor=downscaling_factors[2], num_heads=heads[2],
                            head_dim=head_dim, window_size=window_size, dropout=dropout,
                            relative_pos_embedding=relative_pos_embedding)
        self.enc5 = Encoder(in_dims=hidden_dim * 4, hidden_dimension=hidden_dim * 8,
                            layers=layers[3],
                            downscaling_factor=downscaling_factors[3], num_heads=heads[3],
                            head_dim=head_dim, window_size=window_size, dropout=dropout,
                            relative_pos_embedding=relative_pos_embedding)

        self.dec4 = Decoder(in_dims=hidden_dim * 8, out_dims=hidden_dim * 4,
                            layers=layers[2],
                            up_scaling_factor=downscaling_factors[3], num_heads=heads[2],
                            head_dim=head_dim, window_size=window_size, dropout=dropout,
                            relative_pos_embedding=relative_pos_embedding)

        self.dec3 = Decoder(in_dims=hidden_dim * 4, out_dims=hidden_dim * 2,
                            layers=layers[1],
                            up_scaling_factor=downscaling_factors[2], num_heads=heads[1],
                            head_dim=head_dim, window_size=window_size, dropout=dropout,
                            relative_pos_embedding=relative_pos_embedding)

        self.dec12 = Decoder(in_dims=hidden_dim * 2, out_dims=hidden_dim,
                             layers=layers[0],
                             up_scaling_factor=downscaling_factors[1], num_heads=heads[0],
                             head_dim=head_dim, window_size=window_size, dropout=dropout,
                             relative_pos_embedding=relative_pos_embedding)

        self.converge4 = Converge(hidden_dim * 4)
        self.converge3 = Converge(hidden_dim * 2)
        self.converge12 = Converge(hidden_dim)

        self.final = FinalExpand3D(in_dim=hidden_dim, out_dim=stl_channels,
                                   up_scaling_factor=downscaling_factors[0])
        self.out = nn.Sequential(
            # nn.Linear(stl_channels, num_classes),
            # Rearrange('b h w d c -> b c h w d'),
            nn.Conv3d(stl_channels, num_classes, kernel_size=1)
        )
        # 参数初始化
        self.init_weight()

    def forward(self, img):
        window_size = self.window_size
        assert type(window_size) is int or len(window_size) == 3, 'window_size must be 1 or 3 dimension'
        if type(window_size) is int:
            window_size = np.array([window_size, window_size, window_size])
        _, _, x_s, y_s, z_s = img.shape
        x_ws, y_ws, z_ws = window_size

        assert x_s % (x_ws * 32) == 0, f'x-axis size ({x_s}) must be divisible by x_window_size * 32 ({x_ws * 32}).'
        assert y_s % (y_ws * 32) == 0, f'y-axis size ({y_s}) must be divisible by y_window_size * 32 ({y_ws * 32}).'


        down12_1 = self.enc12(img)  # (B,C, X//4, Y//4, Z//4)
        down3 = self.enc3(down12_1)  # (B, 2C,X//8, Y//8, Z//8)
        down4 = self.enc4(down3)  # (B, 4C,X//16, Y//16, Z//16)
        features = self.enc5(down4)  # (B, 8C,X//32, Y//32, Z//32)

        up4 = self.dec4(features)  # (B, 8C, X//16, Y//16, Z//16 )
        # up1和 down3融合
        up4 = self.converge4(up4, down4)  # (B, 4C, X//16, Y//16, Z//16)

        up3 = self.dec3(up4)  # ((B, 2C,X//8, Y//8, Z//8)
        # up2和 down2融合
        up3 = self.converge3(up3, down3)  # (B,2C, X//8, Y//8)

        up12 = self.dec12(up3)  # (B,C, X//4, Y//4, Z// 4)
        # up3和 down1融合
        up12 = self.converge12(up12, down12_1)  # (B,C, X//4, Y//4, Z//4)

        out = self.final(up12)  # (B,num_classes, X, Y, Z)
        out = self.out(out)
        # Slice last temporal step
        out = out[..., -1]
        return out

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def training_step(self, batch, batch_idx):
        fire_seq, static_data, wind_inputs, *isochrone_mask = batch
        isochrone_mask = isochrone_mask[0]
        pred = self(fire_seq)
        pred = pred[..., 56:-56, 56:-56]
        loss = self.loss_fn(pred, isochrone_mask)
        self.log("train_loss", loss)

        # Update metrics
        self.train_accuracy(pred, isochrone_mask)
        self.train_precision(pred, isochrone_mask)
        self.train_recall(pred, isochrone_mask)
        self.train_f1(pred, isochrone_mask)

        self.log("train_accuracy", self.train_accuracy, on_step=True, on_epoch=False)
        self.log("train_precision", self.train_precision, on_step=True, on_epoch=False)
        self.log("train_recall", self.train_recall, on_step=True, on_epoch=False)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=False)
        return {"loss": loss, "predictions": pred, "targets": isochrone_mask}

    def validation_step(self, batch, batch_idx):
        fire_seq, static_data, wind_inputs, *isochrone_mask = batch
        isochrone_mask = isochrone_mask[0]
        pred = self(fire_seq)
        pred = pred[..., 56:-56, 56:-56]
        loss = self.loss_fn(pred, isochrone_mask)
        self.log("val_loss", loss)

        # Update metrics
        self.val_accuracy(pred, isochrone_mask)
        self.val_precision(pred, isochrone_mask)
        self.val_recall(pred, isochrone_mask)
        self.val_f1(pred, isochrone_mask)

        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        return {"loss": loss, "predictions": pred, "targets": isochrone_mask}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

