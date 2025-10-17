import yaml
import argparse
import pytorch_lightning as pl
from lightning.pytorch.profilers import AdvancedProfiler
from pytorch_lightning.callbacks import LearningRateMonitor
import sys
import os
import wandb

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.utils import Logger
from scripts.utils.wandb_logger import get_wandb_logger
from data import FireDataModule
from scripts.callbacks import EarlyStoppingHandler, ImageLoggerHandler, LoggingCallback, FinalMetricsCallback
from scripts.callbacks.wandb_explainability_callback import WandbExplainabilityCallback
from model.factorized_transformer import FactorizedFireTransformer


def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    global_cfg = load_yaml_config(args.global_config)
    model_cfg = load_yaml_config(args.model_config)
    trainer_cfg = load_yaml_config(args.trainer_config)
    data_cfg = load_yaml_config(args.data_config)
    
    # Override config with WandB sweep parameters (if running in sweep)
    if wandb.run is not None and hasattr(wandb.config, 'keys'):
        print("=" * 60)
        print("WandB Sweep detected - overriding config parameters:")
        print("=" * 60)
        for key in wandb.config.keys():
            if '.' in key:
                # Handle nested keys like 'loss_fn_settings.weight'
                parts = key.split('.')
                if parts[0] in model_cfg:
                    if isinstance(model_cfg[parts[0]], dict):
                        model_cfg[parts[0]][parts[1]] = wandb.config[key]
                        print(f"  {key} = {wandb.config[key]}")
            else:
                # Handle top-level keys
                if key in model_cfg:
                    model_cfg[key] = wandb.config[key]
                    print(f"  {key} = {wandb.config[key]}")
        print("=" * 60 + "\n")

    # Logger - Use W&B instead of TensorBoard
    logger_cfg = trainer_cfg['logger']
    if logger_cfg['enabled'] is True:
        logger_type = logger_cfg.get('type', 'wandb')  # Default to wandb
        
        if logger_type == 'wandb':
            # Prepare config for wandb
            wandb_config = {
                **model_cfg,
                **data_cfg,
                **global_cfg
            }
            
            logger = get_wandb_logger(
                project=logger_cfg.get('project', 'fire-prediction'),
                name=logger_cfg.get('name', 'factorized-transformer'),
                entity=logger_cfg.get('entity', None),
                config=wandb_config,
                save_dir=logger_cfg.get('dir', './wandb'),
                log_model=logger_cfg.get('log_model', True)
            )
        else:
            # Fallback to TensorBoard
            logger = Logger.get_tensorboard_logger(
                save_dir=logger_cfg['dir'],
                name=logger_cfg['name'],
                default_hp_metric=logger_cfg.get('default_hp_metric', False)
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
        
    image_logger_cfg = callbacks_cfg['image_logger']
    if image_logger_cfg.get('enabled', False) is True:
        image_prediction_logger_callback = ImageLoggerHandler()
        callbacks.append(image_prediction_logger_callback)
        
    logging_callback_cfg = callbacks_cfg['logging_callback']
    if logging_callback_cfg.get('enabled', False) is True:
        logging_callback = LoggingCallback()
        callbacks.append(logging_callback)
        
    final_metrics_callback_cfg = callbacks_cfg['final_metrics_callback']
    if final_metrics_callback_cfg.get('enabled', False) is True:
        final_metrics_callback = FinalMetricsCallback(
            on_training_data=final_metrics_callback_cfg.get('on_training_data', False),
            on_validation_data=final_metrics_callback_cfg.get('on_validation_data', True)
        )
        callbacks.append(final_metrics_callback)
        
    learning_rate_monitor_callback_cfg = callbacks_cfg['learning_rate_monitor']
    if learning_rate_monitor_callback_cfg.get('enabled', False) is not False:
        learning_rate_monitor = LearningRateMonitor()
        callbacks.append(learning_rate_monitor)
    
    # W&B Explainability Callback
    wandb_explainability_cfg = callbacks_cfg.get('wandb_explainability', {})
    if wandb_explainability_cfg.get('enabled', False) is True and logger_cfg.get('type') == 'wandb':
        wandb_explainability_callback = WandbExplainabilityCallback(
            log_every_n_epochs=wandb_explainability_cfg.get('log_every_n_epochs', 5),
            num_samples=wandb_explainability_cfg.get('num_samples', 3),
            method=wandb_explainability_cfg.get('method', 'gradient')
        )
        callbacks.append(wandb_explainability_callback)

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

    # Model - FactorizedFireTransformer
    print("=" * 60)
    print("Initializing FactorizedFireTransformer...")
    print("=" * 60)
    
    model = FactorizedFireTransformer(
        img_size=model_cfg['img_size'],
        patch_size=model_cfg['patch_size'],
        in_channels=model_cfg['in_channels'],
        static_channels=model_cfg['static_channels'],
        num_classes=model_cfg['num_classes'],
        embed_dim=model_cfg['embed_dim'],
        spatial_depth=model_cfg['spatial_depth'],
        temporal_depth=model_cfg['temporal_depth'],
        num_heads=model_cfg['num_heads'],
        dropout=model_cfg['dropout'],
        optimizer_settings=model_cfg['optimizer_settings'],
        lr_scheduler=model_cfg['lr_scheduler'],
        loss_fn=model_cfg['loss_fn'],
        loss_fn_settings=model_cfg['loss_fn_settings']
    )
    
    # Don't call save_hyperparameters() - wandb config already logged during logger init
    # model.hparams.update(datamodule.hparams)
    # model.save_hyperparameters()
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"  - Spatial depth: {model_cfg['spatial_depth']}")
    print(f"  - Temporal depth: {model_cfg['temporal_depth']}")
    print(f"  - Embed dim: {model_cfg['embed_dim']}")
    print(f"  - Patch size: {model_cfg['patch_size']}")

    # Choose profiler
    profiler_config = trainer_cfg.get('profiler', 'simple')
    if profiler_config == "advanced":
        profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
    else:
        profiler = profiler_config

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
        callbacks=callbacks,
        profiler=profiler
    )
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    trainer.fit(
        model=model,
        train_dataloaders=datamodule.train_dataloader(),
        val_dataloaders=datamodule.val_dataloader()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FactorizedFireTransformer model")
    parser.add_argument("--global_config", default="configs/global_config.yaml", help="Path to global config.")
    parser.add_argument("--trainer_config", default="configs/trainer_config.yaml", help="Path to trainer config.")
    parser.add_argument("--data_config", default="configs/data_config.yaml", help="Path to data config.")
    parser.add_argument("--model_config", default="configs/factorized_model_config.yaml", 
                        help="Path to model config (default: factorized_model_config.yaml)")
    args = parser.parse_args()
    main(args)
