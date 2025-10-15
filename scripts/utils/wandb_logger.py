"""
Weights & Biases Logger Utility for Fire Prediction.

Provides wandb integration for PyTorch Lightning.
"""

import wandb
from pytorch_lightning.loggers import WandbLogger
from typing import Optional, Dict, Any


def get_wandb_logger(
    project: str,
    name: str,
    entity: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    save_dir: str = "./wandb",
    log_model: bool = True,
    **kwargs
) -> WandbLogger:
    """
    Create and configure a Weights & Biases logger for PyTorch Lightning.
    
    Args:
        project: W&B project name (e.g., "fire-prediction")
        name: Run name (e.g., "factorized-transformer-v1")
        entity: W&B team/user name (optional)
        config: Additional config to log as hyperparameters
        save_dir: Directory to save wandb logs
        log_model: Whether to log model checkpoints to W&B
        **kwargs: Additional arguments for WandbLogger
    
    Returns:
        WandbLogger instance
    
    Example:
        >>> logger = get_wandb_logger(
        ...     project="fire-prediction",
        ...     name="experiment-1",
        ...     entity="my-team",
        ...     config={"batch_size": 4, "lr": 1e-4}
        ... )
    """
    
    # Create WandbLogger
    logger = WandbLogger(
        project=project,
        name=name,
        entity=entity,
        save_dir=save_dir,
        log_model=log_model if log_model else False,
        config=config,
        **kwargs
    )
    
    return logger


def log_explainability_images(
    logger: WandbLogger,
    images: Dict[str, Any],
    step: Optional[int] = None,
    prefix: str = "explainability"
):
    """
    Log explainability visualizations to W&B.
    
    Args:
        logger: WandbLogger instance
        images: Dictionary of {name: image_array or wandb.Image}
        step: Optional step/epoch number
        prefix: Prefix for logged images
    
    Example:
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2, 3])
        >>> log_explainability_images(
        ...     logger,
        ...     {"gradient_attribution": wandb.Image(fig)},
        ...     step=10
        ... )
    """
    if logger.experiment is None:
        return
    
    log_dict = {}
    for name, image in images.items():
        key = f"{prefix}/{name}" if prefix else name
        
        # Convert to wandb.Image if not already
        if not isinstance(image, wandb.Image):
            log_dict[key] = wandb.Image(image)
        else:
            log_dict[key] = image
    
    # Log to wandb
    if step is not None:
        logger.experiment.log(log_dict, step=step)
    else:
        logger.experiment.log(log_dict)


def log_attention_maps(
    logger: WandbLogger,
    attention_maps: Dict[str, Any],
    step: Optional[int] = None
):
    """
    Log attention maps to W&B.
    
    Args:
        logger: WandbLogger instance
        attention_maps: Dict with 'spatial' and/or 'temporal' attention
        step: Optional step/epoch number
    """
    log_explainability_images(
        logger,
        attention_maps,
        step=step,
        prefix="attention"
    )


def log_gradients_histogram(
    logger: WandbLogger,
    gradients: Dict[str, Any],
    step: Optional[int] = None
):
    """
    Log gradient histograms to W&B.
    
    Args:
        logger: WandbLogger instance
        gradients: Dict of gradient tensors
        step: Optional step/epoch number
    """
    if logger.experiment is None:
        return
    
    log_dict = {}
    for name, grad in gradients.items():
        if grad is not None:
            # Convert to numpy and create histogram
            grad_np = grad.detach().cpu().numpy().flatten()
            log_dict[f"gradients/{name}"] = wandb.Histogram(grad_np)
    
    if step is not None:
        logger.experiment.log(log_dict, step=step)
    else:
        logger.experiment.log(log_dict)


def log_model_graph(
    logger: WandbLogger,
    model: Any,
    input_array: Optional[tuple] = None
):
    """
    Log model architecture graph to W&B.
    
    Args:
        logger: WandbLogger instance
        model: PyTorch model
        input_array: Example input for the model
    """
    if logger.experiment is None:
        return
    
    try:
        if input_array is not None:
            logger.watch(model, log="all", log_freq=100)
    except Exception as e:
        print(f"Warning: Could not log model graph: {e}")


class WandbExplainabilityLogger:
    """
    Helper class to log explainability artifacts to W&B throughout training.
    """
    
    def __init__(self, logger: WandbLogger):
        self.logger = logger
        self.step_counter = 0
    
    def log_explanation_report(
        self,
        explain_results: Dict[str, Any],
        fire_input: Any,
        prediction: Any,
        target: Optional[Any] = None,
        step: Optional[int] = None
    ):
        """
        Log a complete explanation report to W&B.
        
        Args:
            explain_results: Output from model.explain()
            fire_input: Input fire sequence
            prediction: Model prediction
            target: Ground truth (optional)
            step: Training step
        """
        import matplotlib.pyplot as plt
        import torch
        
        if step is None:
            step = self.step_counter
            self.step_counter += 1
        
        # 1. Log prediction vs target
        if target is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Input (last timestep)
            fire_frame = fire_input[0, 0, :, :, -1].cpu().numpy()
            axes[0].imshow(fire_frame, cmap='YlOrRd')
            axes[0].set_title('Input Fire (last timestep)')
            axes[0].axis('off')
            
            # Prediction
            pred = prediction[0, 0].detach().cpu().numpy()
            pred_prob = 1 / (1 + torch.exp(-torch.tensor(pred)).numpy())
            axes[1].imshow(pred_prob, cmap='RdYlBu_r', vmin=0, vmax=1)
            axes[1].set_title('Prediction')
            axes[1].axis('off')
            
            # Target
            tgt = target[0, 0].cpu().numpy() if target.dim() == 4 else target[0].cpu().numpy()
            axes[2].imshow(tgt, cmap='gray', vmin=0, vmax=1)
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')
            
            plt.tight_layout()
            
            self.logger.experiment.log({
                "predictions/overlay": wandb.Image(fig),
                "step": step
            })
            plt.close(fig)
        
        # 2. Log gradient attribution
        if 'grads' in explain_results:
            grads = explain_results['grads']
            
            # Wind gradients (most interpretable)
            if grads.get('wind') is not None:
                fig, ax = plt.subplots(figsize=(10, 6))
                wind_grad = grads['wind'][0].cpu().numpy()
                
                x = range(wind_grad.shape[1])
                ax.bar([i - 0.2 for i in x], abs(wind_grad[0]), 0.4, label='U (East-West)', alpha=0.7)
                ax.bar([i + 0.2 for i in x], abs(wind_grad[1]), 0.4, label='V (North-South)', alpha=0.7)
                ax.set_xlabel('Timestep')
                ax.set_ylabel('Attribution Magnitude')
                ax.set_title('Wind Attribution')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                
                self.logger.experiment.log({
                    "explainability/wind_attribution": wandb.Image(fig),
                    "step": step
                })
                plt.close(fig)
            
            # Log gradient histograms
            log_gradients_histogram(self.logger, grads, step=step)
        
        # 3. Log attention maps (if available)
        if explain_results.get('spatial_attention'):
            self.logger.experiment.log({
                "attention/spatial_available": True,
                "attention/num_spatial_layers": len(explain_results['spatial_attention']),
                "step": step
            })
        
        if explain_results.get('temporal_attention'):
            self.logger.experiment.log({
                "attention/temporal_available": True,
                "attention/num_temporal_layers": len(explain_results['temporal_attention']),
                "step": step
            })
