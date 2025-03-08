import torch
import torchvision.utils as vutils
from pytorch_lightning.callbacks import Callback

class ImageLoggerHandler(Callback):
    """
    Custom callback to log predicted binary masks vs. targets as images in TensorBoard during validation.
    """
    def __init__(self, threshold=0.5, log_interval=1, num_images=8):
        """
        Args:
            threshold (float): Threshold to binarize predictions.
            log_interval (int): Log images every 'log_interval' epochs.
            num_images (int): Number of images to log per validation epoch.
        """
        super().__init__()
        self.threshold = threshold
        self.log_interval = log_interval
        self.num_images = num_images

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """
        Collect predictions for the first N images during validation.
        """
        if batch_idx == 0 and (pl_module.current_epoch % self.log_interval == 0):
            pred, target = outputs["predictions"], outputs["targets"]
            pred = pred.squeeze(-1)  # Remove the T=1 dimension
            target = target.squeeze(-1)  # If targets also have T=1

            # Binarize predictions
            pred_binary = (torch.sigmoid(pred.detach().cpu()) > self.threshold).float()
            target_images = target.detach().cpu()

            # Stack binary masks and targets vertically
            combined_images = torch.cat([pred_binary[:self.num_images], target_images[:self.num_images]], dim=1)  # [B, C * 2, H, W]

            # Create a grid for visualization
            comparison_grid = vutils.make_grid(combined_images, nrow=1, normalize=True, value_range=(0, 1))

            # Log the comparison grid to TensorBoard
            trainer.logger.experiment.add_image(
                "Predicted Binary Mask | Target",
                comparison_grid,
                global_step=pl_module.current_epoch  # Use `global_step` to track progression
            )

