import pytorch_lightning as pl
import torchmetrics
import torch

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
                "hp/train_accuracy": 0, "hp/train_precision": 0, "hp/train_recall": 0, "hp/train_f1": 0
            })

        if self.on_validation_data:
            metrics_init.update({
                "hp/val_accuracy": 0, "hp/val_precision": 0, "hp/val_recall": 0, "hp/val_f1": 0
            })

        trainer.logger.log_hyperparams(pl_module.hparams, metrics_init)

    def compute_metrics(self, model, dataloader):
        """Computes accuracy, precision, recall, and F1-score for a given dataset using torchmetrics."""
        accuracy = torchmetrics.classification.BinaryAccuracy().to(model.device)
        precision = torchmetrics.classification.BinaryPrecision().to(model.device)
        recall = torchmetrics.classification.BinaryRecall().to(model.device)
        f1 = torchmetrics.classification.BinaryF1Score().to(model.device)

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                fire_seq, static_data, wind_inputs, *isochrone_mask = batch
                fire_seq = fire_seq.to(model.device)
                static_data = static_data.to(model.device)
                wind_inputs = wind_inputs.to(model.device)
                isochrone_mask = isochrone_mask[0].to(model.device)
                pred = model(fire_seq, static_data, wind_inputs)
                pred = pred[..., 56:-56, 56:-56]  # Cropping like in validation_step

                accuracy.update(pred, isochrone_mask.int())
                precision.update(pred, isochrone_mask.int())
                recall.update(pred, isochrone_mask.int())
                f1.update(pred, isochrone_mask.int())

        return {
            "accuracy": accuracy.compute().item(),
            "precision": precision.compute().item(),
            "recall": recall.compute().item(),
            "f1": f1.compute().item(),
        }

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of training to log final metrics under hparams."""
        tensorboard_logger = pl_module.logger.experiment

        if self.on_training_data and trainer.train_dataloader is not None:
            train_metrics = self.compute_metrics(pl_module, trainer.train_dataloader)
            for k, v in train_metrics.items():
                tensorboard_logger.add_scalar(f"hp/train_{k}", v)

        if self.on_validation_data:
            val_loader = (
                trainer.datamodule.val_dataloader() if trainer.datamodule else trainer.val_dataloaders
            )
            if val_loader is not None:
                val_metrics = self.compute_metrics(pl_module, val_loader)
                for k, v in val_metrics.items():
                    tensorboard_logger.add_scalar(f"hp/val_{k}", v)

        print("Logged Final Metrics.")

