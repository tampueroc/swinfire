import pytorch_lightning as pl
import torchmetrics
import torch
from torchmetrics import classification

class FinalMetricsCallback(pl.Callback):
    def __init__(self, on_training_data=True, on_validation_data=True):
        """
        Callback to compute final metrics after training and log them under hparams.

        Args:
            on_training_data (bool): Compute metrics on the training dataset.
            on_validation_data (bool): Compute metrics on the validation dataset.
        """
        self.on_training_data = on_training_data
        self.on_validation_data = on_validation_data

    def on_train_start(self, trainer, pl_module):
        """Initialize hyperparameters logging at the start of training based on initialization arguments."""
        metrics_init = {}

        if self.on_training_data:
            metrics_init.update({
                "hp/train_accuracy": 0, "hp/train_precision": 0, "hp/train_recall": 0, "hp/train_f1": 0, "hp/train_jaccard_index": 0
            })

        if self.on_validation_data:
            metrics_init.update({
                "hp/val_accuracy": 0, "hp/val_precision": 0, "hp/val_recall": 0, "hp/val_f1": 0, "hp/val_jaccard_index": 0
            })

        # WandbLogger has different signature than TensorBoardLogger
        from pytorch_lightning.loggers import WandbLogger
        if isinstance(trainer.logger, WandbLogger):
            # W&B doesn't need metrics_init upfront, just log hyperparams
            if hasattr(pl_module, 'hparams') and pl_module.hparams:
                trainer.logger.log_hyperparams(pl_module.hparams)
        else:
            # TensorBoard logger accepts metrics
            trainer.logger.log_hyperparams(pl_module.hparams, metrics_init)

    def compute_metrics(self, model, dataloader):
        """Computes accuracy, precision, recall, and F1-score for a given dataset using torchmetrics."""
        accuracy = classification.BinaryAccuracy().to(model.device)
        precision = classification.BinaryPrecision().to(model.device)
        recall = classification.BinaryRecall().to(model.device)
        f1 = classification.BinaryF1Score().to(model.device)
        jaccard_index = classification.BinaryJaccardIndex().to(model.device)

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                fire_seq, static_data, wind_inputs, *isochrone_mask = batch
                fire_seq = fire_seq.to(model.device)
                static_data = static_data.to(model.device)
                wind_inputs = wind_inputs.to(model.device)
                isochrone_mask = isochrone_mask[0].to(model.device)
                pred = model(fire_seq, static_data, wind_inputs)
                
                # Extract fire class and apply sigmoid
                target_fire = isochrone_mask[:, 1:2, :, :] if isochrone_mask.shape[1] == 2 else isochrone_mask
                pred_fire = pred[:, 1:2, :, :] if pred.shape[1] == 2 else pred
                
                # Match shapes (crop or interpolate)
                if pred_fire.shape[-2:] != target_fire.shape[-2:]:
                    if pred_fire.shape[-2] > target_fire.shape[-2]:
                        crop_h = (pred_fire.shape[-2] - target_fire.shape[-2]) // 2
                        crop_w = (pred_fire.shape[-1] - target_fire.shape[-1]) // 2
                        pred_fire = pred_fire[..., crop_h:-crop_h if crop_h > 0 else None,
                                             crop_w:-crop_w if crop_w > 0 else None]
                    else:
                        pred_fire = torch.nn.functional.interpolate(
                            pred_fire, size=target_fire.shape[-2:], mode='bilinear', align_corners=False
                        )
                
                # Convert to probabilities and binary targets
                pred_probs = torch.sigmoid(pred_fire).squeeze(1)  # [B, H, W]
                target_binary = target_fire.squeeze(1).int()     # [B, H, W]

                accuracy.update(pred_probs, target_binary)
                precision.update(pred_probs, target_binary)
                recall.update(pred_probs, target_binary)
                f1.update(pred_probs, target_binary)
                jaccard_index.update(pred_probs, target_binary)

        return {
            "accuracy": accuracy.compute().item(),
            "precision": precision.compute().item(),
            "recall": recall.compute().item(),
            "f1": f1.compute().item(),
            "jaccard_index": jaccard_index.compute().item()
        }

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of training to log final metrics under hparams."""
        from pytorch_lightning.loggers import WandbLogger
        
        is_wandb = isinstance(trainer.logger, WandbLogger)
        logger = pl_module.logger.experiment

        if self.on_training_data and trainer.train_dataloader is not None:
            train_metrics = self.compute_metrics(pl_module, trainer.train_dataloader)
            for k, v in train_metrics.items():
                if is_wandb:
                    logger.log({f"hp/train_{k}": v})
                else:
                    logger.add_scalar(f"hp/train_{k}", v)

        if self.on_validation_data:
            val_loader = (
                trainer.datamodule.val_dataloader() if trainer.datamodule else trainer.val_dataloaders
            )
            if val_loader is not None:
                val_metrics = self.compute_metrics(pl_module, val_loader)
                for k, v in val_metrics.items():
                    if is_wandb:
                        logger.log({f"hp/val_{k}": v})
                    else:
                        logger.add_scalar(f"hp/val_{k}", v)

        print("Logged Final Metrics.")

