import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

class SaliencyMapCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.static_proj_activations = None
        self.gate_maps = None

    def _static_proj_hook(self, module, input, output):
        # Capture static projector activations
        self.static_proj_activations = output.detach()

    def _gate_hook(self, module, input, output):
        # Capture the gating mask from the fusion module.
        # In ConditionalGate, output = x * gate + static_proj * (1 - gate),
        # so you might want to capture the gate itself.
        # One option is to modify the module to return gate as an additional output.
        # Alternatively, if the gate is computed inside a conv net, you might
        # insert a hook there.
        self.gate_maps = output[1].detach()

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        # Assuming the encoder is accessible as pl_module.encoder (or via pl_module.model.enc5, etc.)
        # and that the static projector and gate are members of the encoder.
        encoder = pl_module.enc5  # adjust this to the proper module

        # Register forward hooks on the static projector and gate modules.
        # You might need to know the exact attribute names, e.g. encoder.static_projector and encoder.gate.
        self.gate_hook_handle = encoder.gate.register_forward_hook(self._gate_hook)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.gate_hook_handle.remove()
        logger = pl_module.logger.experiment

        if self.gate_maps is not None:
            # gate_maps: [B, 1152, 16, 16, 4]; collapse the last dimension.
            gate_map = self.gate_maps[..., -1]  # Now [B, 1152, 16, 16]
            B, C_gate, H_ds, W_ds = gate_map.shape  # C_gate = 1152, H_ds=W_ds=16

            # Retrieve the StaticProjector conv weights from enc5.static_projector.proj[0]
            # This conv maps the original 8 static channels to 1152 channels.
            static_conv_weight = pl_module.enc5.static_projector.proj[0].weight
            # Average over the kernel dimensions to get a mapping matrix of shape [1152, 8].
            mapping_matrix = static_conv_weight.mean(dim=[2, 3])  # [1152, 8]
            mapping_matrix = mapping_matrix.to(gate_map.dtype)

            # Reshape gate_map: from [B, 1152, 16, 16] to [B, 1152, 16*16] and transpose:
            gate_reshaped = gate_map.view(B, C_gate, -1).transpose(1, 2)  # [B, 256, 1152]
            # Multiply to project each 1152-dim vector into 8 channels: [B, 256, 8]
            static_contrib = torch.matmul(gate_reshaped, mapping_matrix)  # [B, 256, 8]
            # Reshape to [B, 8, 16, 16]
            static_contrib = static_contrib.transpose(1, 2).view(B, 8, H_ds, W_ds)

            # Upsample to original static resolution (512x512)
            upsampled_contrib = F.interpolate(static_contrib, size=(512, 512), mode='bilinear', align_corners=False)
            # Now upsampled_contrib is [B, 8, 512, 512]

            # Log heat maps per original static channel for the first sample in the batch
            for channel in range(8):
                heatmap = upsampled_contrib[0, channel, :, :]  # [512, 512]
                # Normalize heatmap for visualization purposes (optional)
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
                logger.add_image(
                    f"ConditionalGate/StaticChannel_{channel}_heatmap",
                    heatmap.unsqueeze(0),  # add channel dimension for TensorBoard (1, H, W)
                    global_step=trainer.global_step
                )

        self.gate_maps = None
