import pytorch_lightning as pl
from torch.utils.tensorboard import SummaryWriter

class LoggingCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.valid_losses = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """Collects training loss at the end of each batch."""
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss'].detach().item()
            self.train_losses.append(loss)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Collects validation loss at the end of each batch."""
        if isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss'].detach().item()
            self.valid_losses.append(loss)

    def on_epoch_end(self, trainer, pl_module):
        """Logs average training and validation loss per epoch in TensorBoard under 'Average Loss per Epoch'."""
        if self.train_losses and self.valid_losses:  # Ensure non-empty lists
            avg_train_loss = sum(self.train_losses) / len(self.train_losses)
            avg_val_loss = sum(self.valid_losses) / len(self.valid_losses)

            writer = pl_module.logger.experiment

            writer.add_scalars("Average Loss per Epoch",  # Custom plot name
                               {"Train": avg_train_loss, "Validation": avg_val_loss},
                               global_step=trainer.current_epoch)

        # Reset lists for next epoch
        self.train_losses.clear()
        self.valid_losses.clear()

