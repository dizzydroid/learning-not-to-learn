# src/main.py

import argparse
import os
import logging
import torch
import torch.optim as optim
from torch.optim import lr_scheduler

# Ensure a consistent way to import local modules,
# whether running as a script or as part of a package.
if __package__ is None or __package__ == '':
    # Allow direct execution of main.py for development/testing
    # This assumes main.py is in src/ and other modules are in src/
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parent.parent)) # Add project root to path
    from src.utils import load_config, set_seeds, setup_logging, get_device, ensure_dir
    from src.data_loader import get_data_loaders
    from src.models import get_models
    from src.trainer import Trainer
else:
    # Standard package imports
    from .utils import load_config, set_seeds, setup_logging, get_device, ensure_dir
    from .data_loader import get_data_loaders
    from .models import get_models
    from .trainer import Trainer


def get_optimizer(optimizer_config: dict, model_parameters) -> optim.Optimizer:
    """
    Creates an optimizer based on the configuration.
    """
    opt_type = optimizer_config.get('type', 'Adam').lower()
    lr = optimizer_config.get('lr', 1e-3)
    weight_decay = optimizer_config.get('weight_decay', 0)

    if opt_type == 'adam':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay, betas=betas)
    elif opt_type == 'sgd':
        momentum = optimizer_config.get('momentum', 0)
        nesterov = optimizer_config.get('nesterov', False)
        return optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    elif opt_type == 'adamw':
        betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
        return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay, betas=betas)
    elif opt_type == 'adagrad':
        return optim.Adagrad(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer type: {opt_type}")

def get_scheduler(scheduler_config: dict, optimizer: optim.Optimizer):
    """
    Creates a learning rate scheduler based on the configuration.
    Returns None if scheduler_config is None or type is null/empty.
    """
    if not scheduler_config or not scheduler_config.get('type'):
        return None

    scheduler_type = scheduler_config['type'].lower()
    params = scheduler_config.get('params', {})

    if scheduler_type == 'steplr':
        return lr_scheduler.StepLR(optimizer, **params)
    elif scheduler_type == 'multisteplr':
        return lr_scheduler.MultiStepLR(optimizer, **params)
    elif scheduler_type == 'cosineannealinglr':
        return lr_scheduler.CosineAnnealingLR(optimizer, **params)
    elif scheduler_type == 'reducelronplateau':
        return lr_scheduler.ReduceLROnPlateau(optimizer, **params)
    else:
        logging.warning(f"Unsupported scheduler type: {scheduler_type}. No scheduler will be used.")
        return None

def main():
    parser = argparse.ArgumentParser(description="Learning Not To Learn - Implementation")
    parser.add_argument('--config', type=str, required=True,
                        help="Path to the YAML configuration file.")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help="Mode of operation: 'train' or 'evaluate'.")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="Path to a model checkpoint for evaluation or resuming training.")
    parser.add_argument('--num_epochs', type=int, default=None,
                        help="Override number of training epochs from config.")
    parser.add_argument('--device', type=str, default=None,
                        help="Override device from config (e.g., 'cuda:0', 'cpu').")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Override the base output directory from config.")
    
    args = parser.parse_args()

    # 1. Initial logging setup (console only, before config is loaded)
    setup_logging(log_to_console=True, log_file=None)

    # 2. Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logging.error(f"Failed to load config: {args.config}. Error: {e}")
        return # Exit if config loading fails

    # 3. Override config with command-line arguments if provided
    if args.num_epochs is not None:
        config['training']['num_epochs'] = args.num_epochs
        logging.info(f"Overriding num_epochs with command-line argument: {args.num_epochs}")
    if args.device is not None:
        config['training']['device'] = args.device
        logging.info(f"Overriding device with command-line argument: {args.device}")
    if args.output_dir is not None:
        config['logging']['output_dir'] = args.output_dir
        logging.info(f"Overriding output_dir with command-line argument: {args.output_dir}")


    # 4. Determine output directory and set up file logging
    output_dir = config['logging']['output_dir']
    experiment_name = config['project']['experiment_name']
    if config['logging'].get('use_experiment_subfolder', True):
        experiment_output_dir = os.path.join(output_dir, experiment_name)
    else:
        experiment_output_dir = output_dir
    ensure_dir(experiment_output_dir)
    
    log_filename = config['logging'].get('log_filename', 'run.log')
    log_file_path = os.path.join(experiment_output_dir, log_filename)
    setup_logging(log_file=log_file_path, log_to_console=True) # Re-init with file logging

    logging.info(f"Full configuration loaded: {config}")
    logging.info(f"Output will be saved to: {experiment_output_dir}")


    # 5. Set random seeds and device
    set_seeds(config['project'].get('seed'))
    device = get_device(config['training'].get('device', 'auto'))

    # 6. Get DataLoaders
    try:
        # Determine if validation loader should be created based on config or mode
        create_val_loader = config['data'].get('create_val_loader', args.mode == 'train')
        val_split_ratio = config['data'].get('val_split_ratio', 0.1)
        
        train_loader, test_loader, val_loader = get_data_loaders(
            config, 
            create_val_loader=create_val_loader,
            val_split_ratio=val_split_ratio
        )
        dataloaders = {'train': train_loader, 'test': test_loader}
        if val_loader:
            dataloaders['val'] = val_loader
    except Exception as e:
        logging.error(f"Failed to initialize data loaders: {e}", exc_info=True)
        return

    # 7. Get Models
    try:
        models_dict = get_models(config, device)
    except Exception as e:
        logging.error(f"Failed to initialize models: {e}", exc_info=True)
        return

    # 8. Initialize Optimizers
    try:
        # Parameters for F and G are often optimized together
        params_fg = list(models_dict['feature_extractor_f'].parameters()) + \
                    list(models_dict['task_classifier_g'].parameters())
        optimizer_fg = get_optimizer(config['training']['optimizer_fg'], params_fg)
        
        params_h = models_dict['bias_predictor_h'].parameters()
        optimizer_h = get_optimizer(config['training']['optimizer_h'], params_h)
        
        optimizers = {'optimizer_fg': optimizer_fg, 'optimizer_h': optimizer_h}
    except Exception as e:
        logging.error(f"Failed to initialize optimizers: {e}", exc_info=True)
        return

    # 9. Initialize LR Schedulers (optional)
    scheduler_fg = get_scheduler(config['training'].get('scheduler_fg'), optimizer_fg)
    scheduler_h = get_scheduler(config['training'].get('scheduler_h'), optimizer_h)
    lr_schedulers = {'scheduler_fg': scheduler_fg, 'scheduler_h': scheduler_h}

    # 10. Initialize Trainer
    try:
        trainer = Trainer(
            config=config,
            models=models_dict,
            optimizers=optimizers,
            lr_schedulers=lr_schedulers,
            dataloaders=dataloaders,
            device=device
        )
    except Exception as e:
        logging.error(f"Failed to initialize Trainer: {e}", exc_info=True)
        return

    # 11. Run based on mode
    if args.mode == 'train':
        if args.checkpoint_path:
            logging.info(f"Attempting to resume training from checkpoint: {args.checkpoint_path}")
            # Load checkpoint for resuming (including optimizers and schedulers)
            trainer.load_checkpoint(args.checkpoint_path, load_optimizers_schedulers=True)
        
        try:
            trainer.train()
        except Exception as e:
            logging.error(f"An error occurred during training: {e}", exc_info=True)
            # Optionally save a crash checkpoint
            trainer.save_checkpoint(epoch=trainer.current_epoch, is_best=False) # Save current state
            logging.info("Saved a crash checkpoint.")
            
    elif args.mode == 'evaluate':
        if not args.checkpoint_path:
            logging.error("Evaluation mode requires a --checkpoint_path.")
            # Try to find a 'best_model.pth' in the experiment's checkpoint directory as a fallback
            best_model_path = os.path.join(trainer.checkpoints_dir, "best_model.pth")
            if os.path.exists(best_model_path):
                logging.info(f"No checkpoint path provided, attempting to use 'best_model.pth' from experiment dir: {best_model_path}")
                args.checkpoint_path = best_model_path
            else:
                logging.error(f"'best_model.pth' not found in {trainer.checkpoints_dir}. Please provide a checkpoint.")
                return
        
        try:
            # Load checkpoint for evaluation (only model weights needed by default)
            # The evaluate_model method itself handles loading if path is passed.
            trainer.evaluate_model(checkpoint_path=args.checkpoint_path, on_test_set=True)
        except Exception as e:
            logging.error(f"An error occurred during evaluation: {e}", exc_info=True)
    else:
        logging.error(f"Invalid mode: {args.mode}")

if __name__ == '__main__':
    main()
