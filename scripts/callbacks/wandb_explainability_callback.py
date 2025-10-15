"""
W&B Explainability Callback for FactorizedFireTransformer.

Automatically logs explainability artifacts during training.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers

import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.loggers import WandbLogger
from typing import Optional


class WandbExplainabilityCallback(pl.Callback):
    """
    Callback to log explainability visualizations to W&B during training/validation.
    
    Logs:
    - Prediction overlays (input, prediction, target)
    - Gradient attributions (fire, static, wind)
    - Attention maps (spatial, temporal)
    - Gradient histograms
    
    Args:
        log_every_n_epochs: Log explainability every N epochs (default: 5)
        num_samples: Number of validation samples to explain (default: 3)
        method: 'gradient' or 'integrated_gradients'
    """
    
    def __init__(
        self,
        log_every_n_epochs: int = 5,
        num_samples: int = 3,
        method: str = 'gradient'
    ):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.method = method
    
    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ):
        """Log explainability artifacts at the end of validation."""
        
        # Skip during sanity check
        if trainer.sanity_checking:
            return
        
        # Only log every N epochs
        if trainer.current_epoch % self.log_every_n_epochs != 0:
            return
        
        # Only proceed if using WandbLogger
        if not isinstance(trainer.logger, WandbLogger):
            return
        
        # Get validation dataloader
        val_dataloader = trainer.val_dataloaders
        if val_dataloader is None:
            return
        
        # Get samples
        samples_logged = 0
        for batch_idx, batch in enumerate(val_dataloader):
            if samples_logged >= self.num_samples:
                break
            
            fire_seq, static_data, wind_inputs, isochrone_mask, valid_tokens = batch
            
            # Move to model device
            device = pl_module.device
            fire_seq = fire_seq.to(device)
            static_data = static_data.to(device)
            wind_inputs = wind_inputs.to(device)
            valid_tokens = valid_tokens.to(device)
            isochrone_mask = isochrone_mask.to(device)
            
            # Take only first sample from batch
            fire_seq = fire_seq[:1]
            static_data = static_data[:1]
            wind_inputs = wind_inputs[:1]
            valid_tokens = valid_tokens[:1]
            isochrone_mask = isochrone_mask[:1]
            
            # Generate explanation
            pl_module.eval()
            with torch.no_grad():
                # Get prediction
                pred = pl_module(fire_seq, static_data, wind_inputs, valid_tokens)
                
                # Get explanation (using gradient method for speed)
                explain_results = pl_module.explain(
                    fire_seq, static_data, wind_inputs, valid_tokens,
                    method=self.method
                )
            
            # Log to wandb
            self._log_sample(
                trainer.logger,
                explain_results,
                fire_seq,
                pred,
                isochrone_mask,
                trainer.current_epoch,
                samples_logged
            )
            
            samples_logged += 1
    
    def _log_sample(
        self,
        logger: WandbLogger,
        explain_results: dict,
        fire_input: torch.Tensor,
        prediction: torch.Tensor,
        target: torch.Tensor,
        epoch: int,
        sample_idx: int
    ):
        """Log a single sample's explainability artifacts."""
        import matplotlib.pyplot as plt
        import numpy as np
        
        prefix = f"explainability_epoch{epoch}_sample{sample_idx}"
        
        # 1. Prediction Overlay
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Input (last timestep)
        fire_frame = fire_input[0, 0, :, :, -1].cpu().numpy()
        axes[0].imshow(fire_frame, cmap='YlOrRd', interpolation='nearest')
        axes[0].set_title('Input Fire (t=-1)')
        axes[0].axis('off')
        
        # Prediction (crop to match target)
        pred = prediction[0, 0, 56:-56, 56:-56].detach().cpu().numpy()
        pred_prob = 1 / (1 + np.exp(-pred))
        axes[1].imshow(pred_prob, cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        # Target
        tgt = target[0, 0].cpu().numpy()
        axes[2].imshow(tgt, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
        axes[2].set_title('Ground Truth')
        axes[2].axis('off')
        
        plt.tight_layout()
        logger.experiment.log({
            f"{prefix}/prediction_overlay": wandb.Image(fig)
        })
        plt.close(fig)
        
        # 2. Gradient Attribution
        grads = explain_results['grads']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Fire temporal attribution
        if grads.get('fire') is not None:
            fire_grad = grads['fire'].abs().mean(dim=(1, 2, 3))[0].cpu().numpy()
            axes[0].bar(range(len(fire_grad)), fire_grad, color='orangered', alpha=0.7)
            axes[0].set_title('Fire Temporal Attribution')
            axes[0].set_xlabel('Timestep')
            axes[0].set_ylabel('Attribution')
            axes[0].grid(axis='y', alpha=0.3)
        
        # Static spatial attribution
        if grads.get('static') is not None:
            static_grad = grads['static'].abs().mean(dim=1)[0].cpu().numpy()
            im = axes[1].imshow(static_grad, cmap='YlOrRd', interpolation='bilinear')
            axes[1].set_title('Static Data Attribution')
            axes[1].axis('off')
            plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Wind attribution
        if grads.get('wind') is not None:
            wind_grad = grads['wind'][0].cpu().numpy()
            x = np.arange(wind_grad.shape[1])
            axes[2].bar(x - 0.2, np.abs(wind_grad[0]), 0.4, label='U (East-West)', alpha=0.7)
            axes[2].bar(x + 0.2, np.abs(wind_grad[1]), 0.4, label='V (North-South)', alpha=0.7)
            axes[2].set_title('Wind Attribution')
            axes[2].set_xlabel('Timestep')
            axes[2].set_ylabel('Attribution')
            axes[2].legend()
            axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        logger.experiment.log({
            f"{prefix}/gradient_attribution": wandb.Image(fig)
        })
        plt.close(fig)
        
        # 3. Log gradient statistics
        if grads.get('fire') is not None:
            logger.experiment.log({
                f"{prefix}/grad_fire_mean": grads['fire'].abs().mean().item(),
                f"{prefix}/grad_fire_max": grads['fire'].abs().max().item()
            })
        
        if grads.get('static') is not None:
            logger.experiment.log({
                f"{prefix}/grad_static_mean": grads['static'].abs().mean().item(),
                f"{prefix}/grad_static_max": grads['static'].abs().max().item()
            })
        
        if grads.get('wind') is not None:
            logger.experiment.log({
                f"{prefix}/grad_wind_mean": grads['wind'].abs().mean().item(),
                f"{prefix}/grad_wind_max": grads['wind'].abs().max().item()
            })
