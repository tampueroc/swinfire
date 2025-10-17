import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Dict, Any
from einops import rearrange

from timm.models.vision_transformer import VisionTransformer
from .losses import WeightedFocalLoss, AsymUnifiedFocalLoss


class SpatialEncoder(nn.Module):
    """Vision Transformer for spatial encoding of each timestep."""

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 32,
        in_chans: int = 1,
        static_chans: int = 8,
        embed_dim: int = 512,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()

        # Process static data
        self.static_conv = nn.Conv2d(static_chans, embed_dim, kernel_size=1)

        # Vision Transformer for fire data + static features
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=0,  # No classification head
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=dropout,
            global_pool='',  # Keep patch tokens
        )

        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # For attention extraction
        self.last_attention_maps = []

    def forward(self, fire_frame: torch.Tensor, static_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fire_frame: [B, C, H, W] - single timestep
            static_feat: [B, embed_dim, H, W] - projected static features

        Returns:
            [B, num_patches, embed_dim] - spatial tokens
        """
        # Extract spatial features from fire frame
        tokens = self.vit.forward_features(fire_frame)  # [B, num_patches+1, embed_dim] (with cls token)

        # Remove cls token if present
        if tokens.shape[1] == self.num_patches + 1:
            tokens = tokens[:, 1:, :]  # Remove cls token

        # Now tokens shape: [B, num_patches, embed_dim]
        # Interpolate static features to patch resolution
        B, P, C = tokens.shape
        H = W = int(P ** 0.5)  # num_patches = H*W
        
        # Reshape tokens to spatial grid
        tokens_spatial = rearrange(tokens, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Downsample static features to patch resolution
        static_patches = F.interpolate(static_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # Fuse static features per-patch (addition)
        tokens_spatial = tokens_spatial + static_patches
        
        # Reshape back to patch tokens
        tokens = rearrange(tokens_spatial, 'b c h w -> b (h w) c')

        return tokens  # [B, num_patches, embed_dim]

    def get_attention_maps(self):
        """Extract attention maps from ViT blocks (requires hooks)."""
        attention_maps = []
        for block in self.vit.blocks:
            if hasattr(block.attn, 'last_attn'):
                attention_maps.append(block.attn.last_attn)
        return attention_maps


class TemporalTransformer(nn.Module):
    """Transformer for temporal modeling across timesteps."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.0
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-norm for better stability
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, embed_dim] - temporal sequence
            valid_tokens: [B, T] - mask for valid timesteps (1=valid, 0=invalid)

        Returns:
            [B, T, embed_dim] - temporally encoded features
        """
        # Create mask for transformer (True means "mask out")
        mask = None
        if valid_tokens is not None:
            mask = (valid_tokens == 0)  # Invert: 0 means invalid, True means mask

        output = self.transformer(x, src_key_padding_mask=mask)
        return output


class SpatialDecoder(nn.Module):
    """Decoder to upsample from patches to full resolution."""

    def __init__(
        self,
        embed_dim: int = 512,
        patch_size: int = 32,
        img_size: int = 512,
        num_classes: int = 2,
        hidden_dim: int = 256,
        use_skip: bool = True
    ):
        super().__init__()

        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches_per_side = img_size // patch_size
        self.use_skip = use_skip

        # Calculate number of upsampling layers needed
        # From patch_size x patch_size grid to img_size x img_size
        # Each ConvTranspose2d with stride=2 doubles the resolution
        num_upsample_layers = int(torch.log2(torch.tensor(patch_size)).item())

        # Skip connection fusion layer
        if use_skip:
            self.skip_fusion = nn.Sequential(
                nn.Conv2d(embed_dim * 2, embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            )

        layers = []
        in_ch = embed_dim

        # Initial conv
        layers.extend([
            nn.Conv2d(in_ch, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        ])

        # Upsampling layers
        out_ch = hidden_dim
        for i in range(num_upsample_layers):
            layers.extend([
                nn.ConvTranspose2d(out_ch, out_ch // 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch // 2),
                nn.ReLU(inplace=True)
            ])
            out_ch = out_ch // 2

        # Final classification layer
        layers.append(nn.Conv2d(out_ch, num_classes, kernel_size=1))

        self.decoder = nn.Sequential(*layers)

    def forward(self, patch_features: torch.Tensor, skip_features: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            patch_features: [B, num_patches, embed_dim] - features from temporal transformer
            skip_features: [B, num_patches, embed_dim] - skip connection from spatial encoder

        Returns:
            [B, num_classes, H, W] - full resolution prediction
        """
        B, P, C = patch_features.shape

        # Reshape patches to 2D grid
        H = W = self.num_patches_per_side
        features = rearrange(patch_features, 'b (h w) c -> b c h w', h=H, w=W)

        # Fuse with skip connections if provided
        if self.use_skip and skip_features is not None:
            skip = rearrange(skip_features, 'b (h w) c -> b c h w', h=H, w=W)
            features = torch.cat([features, skip], dim=1)  # [B, embed_dim*2, H, W]
            features = self.skip_fusion(features)  # [B, embed_dim, H, W]

        # Upsample to full resolution
        output = self.decoder(features)  # [B, num_classes, ?, ?]

        # Ensure output is exactly img_size x img_size using interpolation
        if output.shape[-2:] != (self.img_size, self.img_size):
            output = F.interpolate(output, size=(self.img_size, self.img_size),
                                  mode='bilinear', align_corners=False)

        return output


class FactorizedFireTransformer(pl.LightningModule):
    """
    Spatiotemporal Factorized Transformer for Fire Prediction.

    Separates spatial and temporal attention for better explainability:
    - Spatial transformer: learns fire interactions across landscape
    - Temporal transformer: learns progression patterns over time
    - Simple fusion: wind modulates temporal dynamics, static grounds spatial features
    """

    def __init__(
        self,
        # Model architecture
        img_size: int = 512,
        patch_size: int = 32,
        in_channels: int = 1,
        static_channels: int = 8,
        num_classes: int = 2,
        embed_dim: int = 512,
        spatial_depth: int = 4,
        temporal_depth: int = 4,
        num_heads: int = 8,
        dropout: float = 0.0,
        # Training settings
        optimizer_settings: dict = {},
        lr_scheduler: dict = {},
        loss_fn: str = "bce",
        loss_fn_settings: dict = {},
    ):
        super().__init__()

        self.save_hyperparameters()

        # Initialize components
        self.static_encoder = nn.Conv2d(static_channels, embed_dim, kernel_size=1)

        self.spatial_encoder = SpatialEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            static_chans=static_channels,
            embed_dim=embed_dim,
            depth=spatial_depth,
            num_heads=num_heads,
            dropout=dropout
        )

        self.wind_embed = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        self.temporal_transformer = TemporalTransformer(
            d_model=embed_dim,
            nhead=num_heads,
            num_layers=temporal_depth,
            dim_feedforward=embed_dim * 4,
            dropout=dropout
        )

        self.decoder = SpatialDecoder(
            embed_dim=embed_dim,
            patch_size=patch_size,
            img_size=img_size,
            num_classes=num_classes
        )

        # Loss function
        if loss_fn == "bce":
            self.loss_fn = F.binary_cross_entropy_with_logits
        elif loss_fn == "focal_loss":
            self.loss_fn = WeightedFocalLoss(
                alpha=loss_fn_settings.get('alpha', 0.95),
                gamma=loss_fn_settings.get('gamma', 2)
            )
        elif loss_fn == "unified_focal_loss":
            self.loss_fn = AsymUnifiedFocalLoss(
                weight=loss_fn_settings.get('weight', 0.5),
                delta=loss_fn_settings.get('delta', 0.6),
                gamma=loss_fn_settings.get('gamma', 0.5)
            )

        # Metrics
        self.train_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.train_precision = torchmetrics.classification.BinaryPrecision()
        self.train_recall = torchmetrics.classification.BinaryRecall()
        self.train_f1 = torchmetrics.classification.BinaryF1Score()
        self.train_jaccard_index = torchmetrics.classification.BinaryJaccardIndex()

        self.val_accuracy = torchmetrics.classification.BinaryAccuracy()
        self.val_precision = torchmetrics.classification.BinaryPrecision()
        self.val_recall = torchmetrics.classification.BinaryRecall()
        self.val_f1 = torchmetrics.classification.BinaryF1Score()
        self.val_jaccard_index = torchmetrics.classification.BinaryJaccardIndex()

        # Explainability state
        self.explain_enabled = False
        self.explain_state = {}

        # Example input for logging
        example_shape_fire = (4, in_channels, img_size, img_size, 4)
        example_shape_static = (4, static_channels, img_size, img_size)
        example_shape_wind = (4, 2, 4)
        valid_tokens = torch.ones(4, 4, dtype=torch.float32)

        self.example_input_array = (
            torch.randn(example_shape_fire, dtype=torch.float32),
            torch.rand(example_shape_static, dtype=torch.float32),
            torch.rand(example_shape_wind, dtype=torch.float32),
            valid_tokens
        )

    def forward(
        self,
        fire_seq: torch.Tensor,
        static_data: torch.Tensor,
        wind_inputs: torch.Tensor,
        valid_tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            fire_seq: [B, C, H, W, T] - fire progression over time
            static_data: [B, C_static, H, W] - topography/fuel data
            wind_inputs: [B, 2, T] - wind vectors (u, v)
            valid_tokens: [B, T] - mask for valid timesteps

        Returns:
            [B, num_classes, H, W] - fire prediction for last timestep
        """
        B, C, H, W, T = fire_seq.shape

        # 1. Encode static data once
        static_feat = self.static_encoder(static_data)  # [B, embed_dim, H, W]

        # 2. Spatially encode each timestep - KEEP PATCH TOKENS
        spatial_tokens_all = []
        for t in range(T):
            frame = fire_seq[..., t]  # [B, C, H, W]
            tokens = self.spatial_encoder(frame, static_feat)  # [B, num_patches, embed_dim]
            spatial_tokens_all.append(tokens)

        # Stack into: [B, T, num_patches, embed_dim]
        spatial_tokens_all = torch.stack(spatial_tokens_all, dim=1)
        
        # Store first timestep tokens for skip connection
        skip_features = spatial_tokens_all[:, 0, :, :]  # [B, num_patches, embed_dim]

        # 3. Reshape for per-patch temporal modeling
        # [B, T, num_patches, embed_dim] -> [B, num_patches, T, embed_dim]
        spatial_tokens_all = spatial_tokens_all.transpose(1, 2)
        num_patches = spatial_tokens_all.shape[1]
        
        # Reshape to [B*num_patches, T, embed_dim] for temporal transformer
        temporal_input = rearrange(spatial_tokens_all, 'b p t c -> (b p) t c')

        # 4. Add wind context (broadcast to all patches)
        wind_feat = self.wind_embed(wind_inputs.transpose(1, 2))  # [B, T, embed_dim]
        wind_feat = wind_feat.unsqueeze(1).expand(B, num_patches, T, -1)  # [B, num_patches, T, embed_dim]
        wind_feat = rearrange(wind_feat, 'b p t c -> (b p) t c')
        temporal_input = temporal_input + wind_feat

        # 5. Temporal transformer - process each patch's temporal sequence
        # Expand valid_tokens for all patches
        if valid_tokens is not None:
            valid_tokens_expanded = valid_tokens.unsqueeze(1).expand(B, num_patches, T)
            valid_tokens_expanded = rearrange(valid_tokens_expanded, 'b p t -> (b p) t')
        else:
            valid_tokens_expanded = None
            
        temporal_output = self.temporal_transformer(temporal_input, valid_tokens_expanded)  # [B*num_patches, T, embed_dim]

        # 6. Get last timestep features per patch
        if valid_tokens is not None:
            # Get index of last valid token per sample
            last_valid_idx = valid_tokens.sum(dim=1).long() - 1  # [B]
            last_valid_idx = last_valid_idx.clamp(min=0, max=T-1)
            
            # Expand to all patches
            last_valid_idx = last_valid_idx.unsqueeze(1).expand(B, num_patches)  # [B, num_patches]
            last_valid_idx = rearrange(last_valid_idx, 'b p -> (b p)')
            
            # Gather last valid features
            batch_indices = torch.arange(B * num_patches, device=temporal_output.device)
            final_feat = temporal_output[batch_indices, last_valid_idx, :]  # [B*num_patches, embed_dim]
        else:
            final_feat = temporal_output[:, -1, :]  # [B*num_patches, embed_dim]

        # Reshape back to [B, num_patches, embed_dim]
        patch_features = rearrange(final_feat, '(b p) c -> b p c', b=B, p=num_patches)

        # 7. Decode to spatial prediction with skip connections
        output = self.decoder(patch_features, skip_features)  # [B, num_classes, H, W]

        return output

    def training_step(self, batch, batch_idx):
        fire_seq, static_data, wind_inputs, isochrone_mask, valid_tokens = batch
        pred = self(fire_seq, static_data, wind_inputs, valid_tokens)

        # --- DON'T crop target if it's already 400x400 ---
        # Only crop prediction to match target size
        target_fire = isochrone_mask[:, 1:2, :, :] if isochrone_mask.shape[1] == 2 else isochrone_mask

        # Crop or interpolate pred to match target size
        if pred.shape[-2:] != target_fire.shape[-2:]:
            if pred.shape[-2] > target_fire.shape[-2]:
                # Crop prediction
                crop_h = (pred.shape[-2] - target_fire.shape[-2]) // 2
                crop_w = (pred.shape[-1] - target_fire.shape[-1]) // 2
                pred = pred[..., crop_h:-crop_h if crop_h > 0 else None,
                          crop_w:-crop_w if crop_w > 0 else None]
            else:
                # Interpolate prediction up
                pred = F.interpolate(pred, size=target_fire.shape[-2:], mode='bilinear', align_corners=False)

        pred_fire = pred[:, 1:2, :, :] if pred.shape[1] == 2 else pred
        loss = self.loss_fn(pred_fire, target_fire)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # --- probabilities & binary targets for metrics ---
        pred_probs = torch.sigmoid(pred_fire).squeeze(1)          # [B,H,W]
        target_binary = target_fire.squeeze(1).int()              # [B,H,W]

        # --- STEP METRICS: fire-class specific for progress bar ---
        pred_labels = (pred_probs >= 0.5).int()
        
        # Fire pixels only
        fire_mask = (target_binary == 1)
        if fire_mask.sum() > 0:
            fire_precision_step = (pred_labels[fire_mask] == 1).float().mean()
            fire_recall_step = (pred_labels[fire_mask] == 1).float().mean()
        else:
            fire_precision_step = torch.tensor(0.0, device=pred_labels.device)
            fire_recall_step = torch.tensor(0.0, device=pred_labels.device)
        
        # Log class distribution for context
        fire_ratio = fire_mask.float().mean()
        
        self.log("train_fire_precision_step", fire_precision_step, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_fire_recall_step", fire_recall_step, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log("train_fire_ratio", fire_ratio, on_step=True, on_epoch=False, prog_bar=False, logger=True)

        # --- EPOCH METRICS: update stateful torchmetrics ---
        self.train_accuracy.update(pred_probs, target_binary)
        self.train_precision.update(pred_probs, target_binary)
        self.train_recall.update(pred_probs, target_binary)
        self.train_f1.update(pred_probs, target_binary)
        self.train_jaccard_index.update(pred_probs, target_binary)

        # Log epoch metrics (removed accuracy from display, still logged for reference)
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("train_precision", self.train_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_recall", self.train_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_iou", self.train_jaccard_index, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}


    def validation_step(self, batch, batch_idx):
        fire_seq, static_data, wind_inputs, isochrone_mask, valid_tokens = batch
        pred = self(fire_seq, static_data, wind_inputs, valid_tokens)

        # Extract target first
        target_fire = isochrone_mask[:, 1:2, :, :] if isochrone_mask.shape[1] == 2 else isochrone_mask

        # Match prediction size to target
        if pred.shape[-2:] != target_fire.shape[-2:]:
            if pred.shape[-2] > target_fire.shape[-2]:
                crop_h = (pred.shape[-2] - target_fire.shape[-2]) // 2
                crop_w = (pred.shape[-1] - target_fire.shape[-1]) // 2
                pred = pred[..., crop_h:-crop_h if crop_h > 0 else None,
                          crop_w:-crop_w if crop_w > 0 else None]
            else:
                pred = F.interpolate(pred, size=target_fire.shape[-2:], mode='bilinear', align_corners=False)

        pred_fire = pred[:, 1:2, :, :] if pred.shape[1] == 2 else pred

        loss = self.loss_fn(pred_fire, target_fire)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        pred_probs = torch.sigmoid(pred_fire).squeeze(1)  # [B,H,W]
        target_binary = target_fire.squeeze(1).int()      # [B,H,W]

        # Log class distribution for validation too
        fire_mask = (target_binary == 1)
        fire_ratio = fire_mask.float().mean()
        self.log("val_fire_ratio", fire_ratio, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        self.val_accuracy.update(pred_probs, target_binary)
        self.val_precision.update(pred_probs, target_binary)
        self.val_recall.update(pred_probs, target_binary)
        self.val_f1.update(pred_probs, target_binary)
        self.val_jaccard_index.update(pred_probs, target_binary)

        # Show fire-specific metrics in progress bar, hide accuracy
        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_iou", self.val_jaccard_index, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss}


    def configure_optimizers(self):
        optimizer_algorithm = self.hparams.optimizer_settings.get('optimizer', 'adam')
        learning_rate = self.hparams.optimizer_settings.get('learning_rate', 1e-3)
        weight_decay = self.hparams.optimizer_settings.get('weight_decay', 0)

        if optimizer_algorithm == 'adam':
            optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_algorithm == 'adamw':
            optimizer = optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_algorithm}")

        optim_dict = {'optimizer': optimizer}

        scheduler = self.hparams.lr_scheduler.get('scheduler')
        if scheduler == 'reduce_lr_on_plateau':
            scheduler_obj = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.hparams.lr_scheduler.get('factor', 0.5),
                patience=self.hparams.lr_scheduler.get('patience', 7),
                threshold=self.hparams.lr_scheduler.get('threshold', 0.001),
                cooldown=self.hparams.lr_scheduler.get('cooldown', 0)
            )
            optim_dict['lr_scheduler'] = {
                'scheduler': scheduler_obj,
                'monitor': self.hparams.lr_scheduler.get('monitor', 'val_loss'),
                'frequency': self.hparams.lr_scheduler.get('frequency', 1)
            }

        return optim_dict

    def enable_explain(self, enabled: bool = True):
        """Toggle explainability mode and set up attention hooks."""
        self.explain_enabled = enabled
        if enabled:
            self.explain_state = {
                'spatial_attention': [],
                'temporal_attention': [],
                'wind_features': None,
                'static_features': None
            }
            self._attention_hooks = []
            self._setup_attention_hooks()
        else:
            self._remove_attention_hooks()

    def _setup_attention_hooks(self):
        """Set up hooks to capture attention weights automatically."""

        def spatial_attention_hook(module, input, output):
            """Hook to capture spatial attention from ViT."""
            if hasattr(module, 'attn_drop'):  # This is an Attention module
                # Store attention weights if computed
                if hasattr(module, 'get_attention_map'):
                    attn = module.get_attention_map()
                    if attn is not None:
                        self.explain_state['spatial_attention'].append(attn.detach().cpu())

        def temporal_attention_hook(module, input, output):
            """Hook to capture temporal attention from TransformerEncoder."""
            if hasattr(output, 'size') and len(output.size()) == 3:  # [B, T, D]
                # TransformerEncoderLayer stores attention in some implementations
                if hasattr(module, 'self_attn') and hasattr(module.self_attn, '_attention_weights'):
                    attn = module.self_attn._attention_weights
                    self.explain_state['temporal_attention'].append(attn.detach().cpu())

        # Hook into spatial encoder ViT blocks
        for block in self.spatial_encoder.vit.blocks:
            handle = block.attn.register_forward_hook(spatial_attention_hook)
            self._attention_hooks.append(handle)

        # Hook into temporal transformer layers
        for layer in self.temporal_transformer.transformer.layers:
            handle = layer.register_forward_hook(temporal_attention_hook)
            self._attention_hooks.append(handle)

    def _remove_attention_hooks(self):
        """Remove all attention hooks."""
        if hasattr(self, '_attention_hooks'):
            for hook in self._attention_hooks:
                hook.remove()
            self._attention_hooks = []

    def explain(
        self,
        fire_seq: torch.Tensor,
        static_data: torch.Tensor,
        wind_inputs: torch.Tensor,
        valid_tokens: Optional[torch.Tensor] = None,
        method: str = 'gradient'
    ) -> Dict[str, Any]:
        """
        Compute predictions with explainability artifacts.

        Args:
            fire_seq: [B, C, H, W, T] fire progression
            static_data: [B, C, H, W] static features
            wind_inputs: [B, 2, T] wind vectors
            valid_tokens: [B, T] valid timestep mask
            method: 'gradient' or 'integrated_gradients'

        Returns:
            dict with keys:
                - pred: [B, num_classes, H, W] predictions
                - spatial_attention: list of attention maps per timestep
                - temporal_attention: attention weights over time
                - grads: input gradients {fire, static, wind}
        """
        self.enable_explain(True)

        if method == 'gradient':
            result = self._explain_gradients(fire_seq, static_data, wind_inputs, valid_tokens)
        elif method == 'integrated_gradients':
            result = self._explain_integrated_gradients(fire_seq, static_data, wind_inputs, valid_tokens)
        else:
            raise ValueError(f"Unknown explanation method: {method}")

        self.enable_explain(False)
        return result

    def _explain_gradients(
        self,
        fire_seq: torch.Tensor,
        static_data: torch.Tensor,
        wind_inputs: torch.Tensor,
        valid_tokens: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """Standard gradient-based explanation."""
        # Store original training mode
        was_training = self.training

        # Clone inputs and enable gradient tracking
        fire_seq = fire_seq.clone().detach().requires_grad_(True)
        static_data = static_data.clone().detach().requires_grad_(True)
        wind_inputs = wind_inputs.clone().detach().requires_grad_(True)
        if valid_tokens is not None:
            valid_tokens = valid_tokens.clone().detach()

        # Set to train mode for gradient computation
        self.train()

        try:
            # Enable gradient computation explicitly
            with torch.enable_grad():
                # Forward pass with attention hooks
                pred = self(fire_seq, static_data, wind_inputs, valid_tokens)

                # Compute gradients
                pred_sum = pred.sum()
                pred_sum.backward()

            result = {
                'pred': pred.detach(),
                'spatial_attention': self.explain_state.get('spatial_attention', []),
                'temporal_attention': self.explain_state.get('temporal_attention', []),
                'grads': {
                    'fire': fire_seq.grad.detach() if fire_seq.grad is not None else None,
                    'static': static_data.grad.detach() if static_data.grad is not None else None,
                    'wind': wind_inputs.grad.detach() if wind_inputs.grad is not None else None,
                }
            }
        except Exception as e:
            print(f"Warning: Gradient computation failed: {e}")
            # Return without gradients
            result = {
                'pred': pred.detach() if 'pred' in locals() else None,
                'spatial_attention': self.explain_state.get('spatial_attention', []),
                'temporal_attention': self.explain_state.get('temporal_attention', []),
                'grads': {
                    'fire': None,
                    'static': None,
                    'wind': None,
                }
            }
        finally:
            # Restore original training mode
            self.train(was_training)

        return result

    def _explain_integrated_gradients(
        self,
        fire_seq: torch.Tensor,
        static_data: torch.Tensor,
        wind_inputs: torch.Tensor,
        valid_tokens: Optional[torch.Tensor] = None,
        steps: int = 50
    ) -> Dict[str, Any]:
        """
        Integrated Gradients for smoother attribution.

        Computes gradients along interpolation path from baseline to input.
        """
        # Store original training mode
        was_training = self.training
        self.train()

        # Create baselines (zeros)
        fire_baseline = torch.zeros_like(fire_seq)
        static_baseline = torch.zeros_like(static_data)
        wind_baseline = torch.zeros_like(wind_inputs)

        # Accumulate gradients
        fire_grads = []
        static_grads = []
        wind_grads = []

        for step in range(steps):
            # Interpolate between baseline and input
            alpha = (step + 1) / steps

            fire_interp = fire_baseline + alpha * (fire_seq - fire_baseline)
            static_interp = static_baseline + alpha * (static_data - static_baseline)
            wind_interp = wind_baseline + alpha * (wind_inputs - wind_baseline)

            fire_interp.requires_grad_(True)
            static_interp.requires_grad_(True)
            wind_interp.requires_grad_(True)

            # Forward pass
            pred = self(fire_interp, static_interp, wind_interp, valid_tokens)

            # Backward pass
            pred_sum = pred.sum()
            pred_sum.backward()

            # Collect gradients
            if fire_interp.grad is not None:
                fire_grads.append(fire_interp.grad.detach())
            if static_interp.grad is not None:
                static_grads.append(static_interp.grad.detach())
            if wind_interp.grad is not None:
                wind_grads.append(wind_interp.grad.detach())

            # Clear gradients
            self.zero_grad()

        # Average gradients and multiply by input difference
        integrated_fire = (fire_seq - fire_baseline) * torch.stack(fire_grads).mean(dim=0)
        integrated_static = (static_data - static_baseline) * torch.stack(static_grads).mean(dim=0)
        integrated_wind = (wind_inputs - wind_baseline) * torch.stack(wind_grads).mean(dim=0)

        # Final prediction with full input
        with torch.no_grad():
            final_pred = self(fire_seq, static_data, wind_inputs, valid_tokens)

        result = {
            'pred': final_pred,
            'spatial_attention': self.explain_state.get('spatial_attention', []),
            'temporal_attention': self.explain_state.get('temporal_attention', []),
            'grads': {
                'fire': integrated_fire,
                'static': integrated_static,
                'wind': integrated_wind,
            }
        }

        # Restore original training mode
        self.train(was_training)

        return result
