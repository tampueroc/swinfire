import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchvision.utils as vutils

###############################################
# GradCAM Helper: Decoupled CAM Logic
###############################################
class GradCAM:
    def __init__(self, target_layer):
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        # Register a forward hook to save activations.
        handle_fwd = self.target_layer.register_forward_hook(self._save_activation)
        # Register a full backward hook to capture gradients.
        handle_bwd = self.target_layer.register_full_backward_hook(self._save_gradient)
        self.hook_handles.extend([handle_fwd, handle_bwd])

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output is a tuple; we assume the first element is needed.
        self.gradients = grad_output[0].detach()

    def compute_cam(self):
        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks have not captured activations/gradients.")
        # Assume activations and gradients shape: [B, C, H, W]
        # Compute channel-wise weights by averaging gradients spatially.
        weights = torch.mean(self.gradients, dim=(2, 3))  # shape: [B, C]
        # Compute the weighted combination of the activations.
        cam = torch.zeros(self.activations.shape[0], self.activations.shape[2], self.activations.shape[3],
                          device=self.activations.device)
        for i in range(self.activations.shape[0]):
            # Multiply each channel by its corresponding weight and sum.
            cam[i] = torch.sum(weights[i].unsqueeze(-1).unsqueeze(-1) * self.activations[i], dim=0)
            # Apply ReLU to consider only positive contributions.
            cam[i] = F.relu(cam[i])
        # Normalize each CAM map to [0, 1]
        cam_min = cam.view(cam.shape[0], -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam_max = cam.view(cam.shape[0], -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-6)
        return cam

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()

###############################################
# Updated SaliencyMapCallback using GradCAM
###############################################
class SaliencyMapCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        # Remove old attributes related to static projection.
        # We'll rely on GradCAM to compute our heatmaps.
        self.gradcam = None

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        # Choose a target layer for GradCAM.
        # For example, we might use the first convolution layer in the final expansion block.
        # Adjust this as needed (e.g., pl_module.final.net[0] if final is a Sequential).
        target_layer = pl_module.final.net[0]
        self.gradcam = GradCAM(target_layer)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        # Remove hooks after the batch is finished.
        if self.gradcam is not None:
            logger = pl_module.logger.experiment

            # Compute the CAM heatmaps.
            # Make sure that backward() has been called in your test_step so that gradients are available.
            cam_maps = self.gradcam.compute_cam()  # shape: [B, H, W]

            # For demonstration, log the CAM for the first sample.
            heatmap = cam_maps[0].cpu().numpy()
            # Optionally, resize heatmap to match the original input dimensions if needed.
            # Here we simply log it to TensorBoard.
            logger.add_image(
                "GradCAM/Heatmap",
                torch.tensor(heatmap).unsqueeze(0),  # shape [1, H, W] for TensorBoard
                global_step=trainer.global_step,
                dataformats='CHW'
            )
            self.gradcam.remove_hooks()
            self.gradcam = None

