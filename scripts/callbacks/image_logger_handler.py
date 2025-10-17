import torch
import torchvision.utils as vutils
from pytorch_lightning.callbacks import Callback
import wandb
import numpy as np

class ImageLoggerHandler(Callback):
    """
    Custom callback to log predicted fire masks vs. targets to WandB during validation.
    """
    def __init__(self, threshold=0.5, log_interval=1, num_images=4):
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
        Log predictions for the first N images during validation to WandB.
        """
        if batch_idx == 0 and (pl_module.current_epoch % self.log_interval == 0):
            # Extract predictions and targets
            pred = outputs["predictions"]
            target = outputs["targets"]
            
            # Extract fire class (channel 1 if 2 classes)
            if pred.shape[1] == 2:
                pred_fire = pred[:, 1:2, :, :]  # Fire channel
            else:
                pred_fire = pred
            
            if target.shape[1] == 2:
                target_fire = target[:, 1:2, :, :]
            else:
                target_fire = target
            
            # Apply sigmoid and binarize
            pred_probs = torch.sigmoid(pred_fire.detach().cpu())
            pred_binary = (pred_probs > self.threshold).float()
            target_binary = target_fire.detach().cpu().float()
            
            # Log to WandB
            if hasattr(trainer.logger, 'experiment'):
                images_to_log = []
                
                for i in range(min(self.num_images, pred.shape[0])):
                    # Convert to numpy [H, W]
                    pred_np = pred_binary[i, 0].numpy()
                    target_np = target_binary[i, 0].numpy()
                    pred_prob_np = pred_probs[i, 0].numpy()
                    
                    # Create masks for WandB (class_labels for overlay)
                    class_labels = {
                        0: "no_fire",
                        1: "fire"
                    }
                    
                    # Log as WandB Image with masks
                    images_to_log.append(wandb.Image(
                        pred_prob_np,  # Show probability heatmap
                        caption=f"Sample {i+1}",
                        masks={
                            "prediction": {"mask_data": pred_np, "class_labels": class_labels},
                            "ground_truth": {"mask_data": target_np, "class_labels": class_labels}
                        }
                    ))
                
                # Log to WandB
                trainer.logger.experiment.log({
                    "val/predictions": images_to_log,
                    "epoch": pl_module.current_epoch
                })
            else:
                # Fallback to grid for TensorBoard
                pred_binary = pred_binary[:self.num_images]
                target_binary = target_binary[:self.num_images]
                combined_images = torch.cat([pred_binary, target_binary], dim=0)
                comparison_grid = vutils.make_grid(combined_images, nrow=self.num_images, normalize=True)
                trainer.logger.experiment.add_image(
                    "val/predictions_vs_targets",
                    comparison_grid,
                    global_step=pl_module.current_epoch
                )

