import pytorch_lightning as pl
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

    def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        # Assuming the encoder is accessible as pl_module.encoder (or via pl_module.model.enc5, etc.)
        # and that the static projector and gate are members of the encoder.
        encoder = pl_module.enc5  # adjust this to the proper module

        # Register forward hooks on the static projector and gate modules.
        # You might need to know the exact attribute names, e.g. encoder.static_projector and encoder.gate.
        self.static_hook_handle = encoder.static_projector.register_forward_hook(self._static_proj_hook)
        self.gate_hook_handle = encoder.gate.register_forward_hook(self._gate_hook)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Remove hooks (if you only want to capture for one batch) or keep them for multiple batches.
        self.static_hook_handle.remove()
        self.gate_hook_handle.remove()

        # Now, process the captured activations
        if self.static_proj_activations is not None:
            # For per-channel contribution, average over spatial dims.
            # Assume static_proj_activations has shape [B, C_proj, H, W]
            channel_contrib = self.static_proj_activations.mean(dim=[2, 3])
            # For spatial saliency, you might normalize the activations per channel.
            spatial_saliency = self.static_proj_activations / (self.static_proj_activations.max() + 1e-6)

            # Log images to TensorBoard (if using TensorBoard logger)
            logger = trainer.logger.experiment
            # Make a grid of the spatial saliency maps for the first sample in the batch.
            grid = vutils.make_grid(spatial_saliency[0].unsqueeze(1), nrow=4, normalize=True)
            logger.add_image("StaticProjector/SpatialSaliency", grid, global_step=trainer.global_step)
            # Optionally log channel contributions as a histogram.
            for i, contrib in enumerate(channel_contrib[0]):
                logger.add_scalar(f"StaticProjector/Channel_{i}_contrib", contrib.item(), global_step=trainer.global_step)

        if self.gate_maps is not None:
            # Process gating maps similarly.
            # Depending on the shape of the gating maps, you might average or visualize them directly.
            gate_grid = vutils.make_grid(self.gate_maps[0].unsqueeze(1), nrow=4, normalize=True)
            logger.add_image("ConditionalGate/GateMap", gate_grid, global_step=trainer.global_step)

        # Reset captured activations for the next batch.
        self.static_proj_activations = None
        self.gate_maps = None

