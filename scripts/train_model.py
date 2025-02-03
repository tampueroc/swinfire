import yaml
import argparse
import pytorch_lightning as pl

from utils import Logger
from data import FireDataModule
from callbacks import EarlyStoppingHandler, ImageLoggerHandler
from model.swin import SwinUnet3D

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def main(args):
    global_cfg = load_yaml_config(args.global_config)
    model_cfg = load_yaml_config(args.model_config)
    trainer_cfg = load_yaml_config(args.trainer_config)
    data_cfg = load_yaml_config(args.data_config)

    # Logger
    logger_cfg = trainer_cfg['logger']
    if logger_cfg['enabled'] is True:
        logger = Logger.get_tensorboard_logger(
            save_dir=logger_cfg['dir'],
            name=logger_cfg['name']
        )
    else:
        logger = None

    # Callbacks
    callbacks_cfg = trainer_cfg['callbacks']
    callbacks = []

    early_stopper_cfg = callbacks_cfg['early_stopper']
    if early_stopper_cfg.get('enabled', False) is True:
        early_stopping_callback = EarlyStoppingHandler.get_early_stopping_callback(
            monitor=early_stopper_cfg['monitor'],
            patience=early_stopper_cfg['patience'],
            mode=early_stopper_cfg['mode'],
            min_delta=early_stopper_cfg['min_delta']
        )
        callbacks.append(early_stopping_callback)
    image_logger_cfg= callbacks_cfg['image_logger']
    if image_logger_cfg.get('enabled', False) is True:
        image_prediction_logger_callback = ImageLoggerHandler()
        callbacks.append(image_prediction_logger_callback)

    # Datamodule
    datamodule = FireDataModule(
        data_dir=data_cfg['data_dir'],
        sequence_length=data_cfg['sequence_length'],
        batch_size=data_cfg['batch_size'],
        num_workers=data_cfg['num_workers'],
        drop_last=data_cfg['drop_last'],
        pin_memory=data_cfg['pin_memory'],
        use_collate_fn=data_cfg['use_collate_fn'],
        seed=global_cfg.get('seed', 42)
    )
    datamodule.setup()

    # Model
    model = SwinUnet3D(
        in_channel=model_cfg['in_channels'],
        hidden_dim=model_cfg['hidden_dim'],
        layers=model_cfg['layers'],
        downscaling_factors=model_cfg['downscaling_factors'],
        heads=model_cfg['heads'],
        head_dim=model_cfg['head_dim'],
        window_size=model_cfg['window_size'],
        dropout=model_cfg['dropout'],
        relative_pos_embedding=model_cfg['relative_pos_embedding'],
        num_classes=model_cfg['num_classes'],
        loss_fn=model_cfg['loss_fn']
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=trainer_cfg['max_epochs'],
        accelerator=trainer_cfg['accelerator'],
        devices=trainer_cfg['devices'],
        precision=trainer_cfg['precision'],
        logger=logger,
        strategy=trainer_cfg['strategy'],
        fast_dev_run=trainer_cfg['fast_dev_run'],
        accumulate_grad_batches=trainer_cfg['accumulate_grad_batches'],
        callbacks=callbacks
    )
    trainer.fit(
            model=model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader()
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_config", default="configs/global_config.yaml", help="Path to global config.")
    parser.add_argument("--trainer_config", default="configs/trainer_config.yaml", help="Path to trainer config.")
    parser.add_argument("--data_config", default="configs/data_config.yaml", help="Path to data config.")
    parser.add_argument("--model_config", default="configs/model_config.yaml", help="Path to model config.")
    args = parser.parse_args()
    main(args)
