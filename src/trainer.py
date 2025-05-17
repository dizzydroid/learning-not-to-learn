import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple

from .utils import ensure_dir # Assuming utils.py is in the same directory (src)

class Trainer:
    """
    Handles the training and evaluation of the 'Learning Not To Learn' model.
    """
    def __init__(self,
                 config: Dict[str, Any],
                 models: Dict[str, nn.Module],
                 optimizers: Dict[str, optim.Optimizer],
                 lr_schedulers: Dict[str, Optional[optim.lr_scheduler._LRScheduler]],
                 dataloaders: Dict[str, DataLoader],
                 device: torch.device):
        """
        Initializes the Trainer.

        Args:
            config: The global configuration dictionary.
            models: Dictionary containing 'feature_extractor_f', 'task_classifier_g', 
                    'bias_predictor_h', and 'grl'.
            optimizers: Dictionary containing 'optimizer_fg' and 'optimizer_h'.
            lr_schedulers: Dictionary containing 'scheduler_fg' and 'scheduler_h'.
            dataloaders: Dictionary containing 'train', 'val' (optional), and 'test' DataLoaders.
            device: The torch.device to use for training.
        """
        self.config = config
        self.models = models
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.dataloaders = dataloaders
        self.device = device

        self.feature_extractor_f = self.models['feature_extractor_f']
        self.task_classifier_g = self.models['task_classifier_g']
        self.bias_predictor_h = self.models['bias_predictor_h']
        self.grl = self.models['grl'] # Gradient Reversal Layer

        self.optimizer_fg = self.optimizers['optimizer_fg']
        self.optimizer_h = self.optimizers['optimizer_h']
        
        self.scheduler_fg = self.lr_schedulers.get('scheduler_fg')
        self.scheduler_h = self.lr_schedulers.get('scheduler_h')

        # Loss functions
        self.main_criterion = self._get_loss_function(
            self.config['training'].get('main_task_loss_fn', 'CrossEntropyLoss')
        )
        self.bias_criterion = self._get_loss_function(
            self.config['training'].get('bias_prediction_loss_fn', 'CrossEntropyLoss'),
            ignore_index=255 # Crucial for bias map targets from data_loader
        )
        logging.info(f"Main task loss: {self.config['training'].get('main_task_loss_fn', 'CrossEntropyLoss')}")
        logging.info(f"Bias prediction loss: {self.config['training'].get('bias_prediction_loss_fn', 'CrossEntropyLoss')} with ignore_index=255")

        # Training parameters
        self.num_epochs = self.config['training']['num_epochs']
        self.adversarial_lambda = self.config['training']['adversarial_lambda']
        
        # GRL lambda setup
        grl_lambda_val = self.config['training'].get('gradient_reversal_layer', {}).get('grl_lambda_fixed', 1.0)
        self.grl.set_lambda(grl_lambda_val) # Set lambda for GRL
        logging.info(f"Gradient Reversal Layer lambda set to: {grl_lambda_val}")

        # Logging and Checkpointing
        self.output_dir = self.config['logging']['output_dir']
        if self.config['logging'].get('use_experiment_subfolder', True):
            self.experiment_dir = os.path.join(self.output_dir, self.config['project']['experiment_name'])
        else:
            self.experiment_dir = self.output_dir
        ensure_dir(self.experiment_dir)
        
        self.checkpoints_dir = os.path.join(self.experiment_dir, 'checkpoints')
        ensure_dir(self.checkpoints_dir)

        self.tensorboard_writer = None
        if self.config['logging'].get('use_tensorboard', False):
            tb_log_dir = os.path.join(self.experiment_dir, 'tensorboard_logs')
            ensure_dir(tb_log_dir)
            self.tensorboard_writer = SummaryWriter(log_dir=tb_log_dir)
            logging.info(f"TensorBoard logging enabled. Logs will be saved to: {tb_log_dir}")

        self.current_epoch = 0
        self.best_val_metric = -float('inf') if self.config['logging'].get('best_model_metric_mode', 'max') == 'max' else float('inf')
        
        # Training phases
        self.phases_config = self.config['training'].get('phases', {})
        self.pretrain_fg_epochs = self.phases_config.get('pretrain_fg_epochs', 0)
        self.pretrain_h_epochs = self.phases_config.get('pretrain_h_epochs', 0)
        logging.info(f"Training phases: Pretrain F&G for {self.pretrain_fg_epochs} epochs, Pretrain H for {self.pretrain_h_epochs} epochs.")

    def _get_loss_function(self, loss_name: str, **kwargs) -> nn.Module:
        if loss_name.lower() == 'crossentropyloss':
            return nn.CrossEntropyLoss(**kwargs).to(self.device)
        # Add other loss functions if needed
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")

    def _set_models_mode(self, mode: str = 'train'):
        if mode == 'train':
            self.feature_extractor_f.train()
            self.task_classifier_g.train()
            self.bias_predictor_h.train()
        elif mode == 'eval':
            self.feature_extractor_f.eval()
            self.task_classifier_g.eval()
            self.bias_predictor_h.eval()
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'train' or 'eval'.")

    def _train_step_pretrain_fg(self, images: torch.Tensor, main_labels: torch.Tensor) -> Tuple[float, float]:
        """One training step for pretraining F and G."""
        self.optimizer_fg.zero_grad()
        
        features = self.feature_extractor_f(images)
        task_predictions = self.task_classifier_g(features)
        
        loss_g = self.main_criterion(task_predictions, main_labels)
        loss_g.backward()
        self.optimizer_fg.step()
        
        # Calculate accuracy for logging
        _, predicted_labels = torch.max(task_predictions, 1)
        accuracy_g = (predicted_labels == main_labels).float().mean().item()
        
        return loss_g.item(), accuracy_g

    def _train_step_pretrain_h(self, images: torch.Tensor, bias_targets: torch.Tensor) -> Tuple[float, float]:
        """One training step for pretraining H."""
        self.optimizer_h.zero_grad()
        
        with torch.no_grad(): # Keep F fixed
            features = self.feature_extractor_f(images)
            
        bias_predictions = self.bias_predictor_h(features) # Output: (B, NumBins, C, H, W)
        loss_h = self.bias_criterion(bias_predictions, bias_targets) # Target: (B, C, H, W)
        loss_h.backward()
        self.optimizer_h.step()

        # Calculate accuracy for H (pixel-wise accuracy over non-ignored pixels)
        # bias_predictions: (B, NumBins, C, H, W)
        # bias_targets: (B, C, H, W)
        _, predicted_bias_bins = torch.max(bias_predictions, dim=1) # Max over NumBins dim -> (B, C, H, W)
        
        valid_mask = (bias_targets != 255) # Assuming 255 is ignore_index
        accuracy_h = (predicted_bias_bins[valid_mask] == bias_targets[valid_mask]).float().mean().item() if valid_mask.any() else 0.0
        
        return loss_h.item(), accuracy_h

    def _train_step_adversarial(self, images: torch.Tensor, main_labels: torch.Tensor, bias_targets: torch.Tensor) -> Dict[str, float]:
        """One adversarial training step for F, G, and H."""
        batch_losses = {}

        # --- Step 1: Update Bias Predictor H ---
        # Goal: H should accurately predict bias from F's features.
        self.optimizer_h.zero_grad()
        with torch.no_grad(): # Detach features so F is not updated by H's loss in this step
            features_for_h = self.feature_extractor_f(images)
        
        bias_predictions_h = self.bias_predictor_h(features_for_h)
        loss_h_direct = self.bias_criterion(bias_predictions_h, bias_targets)
        loss_h_direct.backward()
        self.optimizer_h.step()
        batch_losses['loss_h_direct'] = loss_h_direct.item()

        # --- Step 2: Update Feature Extractor F and Task Classifier G ---
        # Goal: G should classify main task well.
        # Goal: F should produce features good for G, but bad for H (adversarial).
        self.optimizer_fg.zero_grad()
        
        features_for_fg = self.feature_extractor_f(images)
        
        # Main task loss for G (and F)
        task_predictions_g = self.task_classifier_g(features_for_fg)
        loss_g = self.main_criterion(task_predictions_g, main_labels)
        batch_losses['loss_g'] = loss_g.item()
        
        # Adversarial loss for F (to fool H)
        # Apply GRL to features before feeding to H
        features_for_h_adv = self.grl(features_for_fg) 
        bias_predictions_adv = self.bias_predictor_h(features_for_h_adv)
        loss_adv_for_f = self.bias_criterion(bias_predictions_adv, bias_targets)
        batch_losses['loss_adv_for_f'] = loss_adv_for_f.item()
        
        # Total loss for F and G
        total_loss_fg = loss_g + self.adversarial_lambda * loss_adv_for_f
        total_loss_fg.backward()
        self.optimizer_fg.step()
        batch_losses['total_loss_fg'] = total_loss_fg.item()
        
        # Accuracies for logging
        with torch.no_grad():
            _, predicted_labels_g = torch.max(task_predictions_g, 1)
            batch_losses['accuracy_g'] = (predicted_labels_g == main_labels).float().mean().item()

            # Accuracy of H on non-GRL features (for direct H performance)
            _, predicted_bias_bins_h_direct = torch.max(bias_predictions_h, dim=1)
            valid_mask_h = (bias_targets != 255)
            batch_losses['accuracy_h_direct'] = (predicted_bias_bins_h_direct[valid_mask_h] == bias_targets[valid_mask_h]).float().mean().item() if valid_mask_h.any() else 0.0
            
            # Accuracy of H on GRL features (how well F is fooling H)
            _, predicted_bias_bins_h_adv = torch.max(bias_predictions_adv, dim=1)
            batch_losses['accuracy_h_adv'] = (predicted_bias_bins_h_adv[valid_mask_h] == bias_targets[valid_mask_h]).float().mean().item() if valid_mask_h.any() else 0.0

        return batch_losses

    def train_epoch(self, epoch_num: int, current_phase: str) -> Dict[str, float]:
        """Trains the model for one epoch."""
        self._set_models_mode('train')
        
        epoch_losses = {
            'loss_g': [], 'loss_h_direct': [], 'loss_adv_for_f': [], 'total_loss_fg': [],
            'accuracy_g': [], 'accuracy_h_direct': [], 'accuracy_h_adv': []
        }
        if current_phase == "pretrain_fg":
            epoch_losses = {'loss_g': [], 'accuracy_g': []}
        elif current_phase == "pretrain_h":
            epoch_losses = {'loss_h_direct': [], 'accuracy_h_direct': []}

        num_batches = len(self.dataloaders['train'])
        for batch_idx, (images, bias_targets, main_labels) in enumerate(self.dataloaders['train']):
            images = images.to(self.device)
            bias_targets = bias_targets.to(self.device)
            main_labels = main_labels.to(self.device)

            if current_phase == "pretrain_fg":
                loss_g_val, acc_g_val = self._train_step_pretrain_fg(images, main_labels)
                epoch_losses['loss_g'].append(loss_g_val)
                epoch_losses['accuracy_g'].append(acc_g_val)
                log_str_batch = f"Epoch [{epoch_num+1}/{self.num_epochs}] Phase [{current_phase}] Batch [{batch_idx+1}/{num_batches}] | Loss G: {loss_g_val:.4f}, Acc G: {acc_g_val:.4f}"
            elif current_phase == "pretrain_h":
                loss_h_val, acc_h_val = self._train_step_pretrain_h(images, bias_targets)
                epoch_losses['loss_h_direct'].append(loss_h_val)
                epoch_losses['accuracy_h_direct'].append(acc_h_val)
                log_str_batch = f"Epoch [{epoch_num+1}/{self.num_epochs}] Phase [{current_phase}] Batch [{batch_idx+1}/{num_batches}] | Loss H: {loss_h_val:.4f}, Acc H: {acc_h_val:.4f}"
            elif current_phase == "adversarial":
                batch_metrics = self._train_step_adversarial(images, main_labels, bias_targets)
                for key, val in batch_metrics.items():
                    epoch_losses[key].append(val)
                log_str_batch = (f"Epoch [{epoch_num+1}/{self.num_epochs}] Phase [{current_phase}] Batch [{batch_idx+1}/{num_batches}] | "
                                 f"LossG: {batch_metrics['loss_g']:.4f}, AccG: {batch_metrics['accuracy_g']:.4f} | "
                                 f"LossH: {batch_metrics['loss_h_direct']:.4f}, AccH_direct: {batch_metrics['accuracy_h_direct']:.4f} | "
                                 f"LossAdvF: {batch_metrics['loss_adv_for_f']:.4f}, AccH_adv: {batch_metrics['accuracy_h_adv']:.4f}")
            else:
                raise ValueError(f"Unknown training phase: {current_phase}")

            if (batch_idx + 1) % self.config['logging'].get('log_interval_batches', 50) == 0:
                logging.info(log_str_batch)
        
        avg_epoch_losses = {key: np.mean(val) for key, val in epoch_losses.items() if val}
        return avg_epoch_losses

    def _run_eval_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Runs evaluation for one epoch on the given data_loader."""
        self._set_models_mode('eval')
        
        total_loss_g, total_loss_h = 0, 0
        correct_g, total_g = 0, 0
        correct_h_pixels, total_h_pixels = 0, 0
        
        with torch.no_grad():
            for images, bias_targets, main_labels in data_loader:
                images = images.to(self.device)
                bias_targets = bias_targets.to(self.device)
                main_labels = main_labels.to(self.device)

                features = self.feature_extractor_f(images)
                
                # Main task evaluation
                task_predictions = self.task_classifier_g(features)
                loss_g = self.main_criterion(task_predictions, main_labels)
                total_loss_g += loss_g.item() * images.size(0)
                _, predicted_g = torch.max(task_predictions.data, 1)
                total_g += main_labels.size(0)
                correct_g += (predicted_g == main_labels).sum().item()

                # Bias predictor evaluation
                bias_predictions = self.bias_predictor_h(features)
                loss_h = self.bias_criterion(bias_predictions, bias_targets)
                total_loss_h += loss_h.item() * images.size(0) # Consider how to scale loss for pixel-wise task
                
                _, predicted_bias_bins = torch.max(bias_predictions, dim=1)
                valid_mask = (bias_targets != 255)
                correct_h_pixels += (predicted_bias_bins[valid_mask] == bias_targets[valid_mask]).sum().item()
                total_h_pixels += valid_mask.sum().item()

        avg_loss_g = total_loss_g / len(data_loader.dataset)
        accuracy_g = correct_g / total_g if total_g > 0 else 0.0
        avg_loss_h = total_loss_h / len(data_loader.dataset) # Or total_h_pixels if loss is pixel-wise averaged
        accuracy_h = correct_h_pixels / total_h_pixels if total_h_pixels > 0 else 0.0
        
        return {
            'val_loss_g': avg_loss_g, 'val_accuracy_g': accuracy_g,
            'val_loss_h': avg_loss_h, 'val_accuracy_h': accuracy_h
        }

    def train(self):
        """Main training loop over epochs."""
        logging.info("Starting training...")
        start_total_time = time.time()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            start_epoch_time = time.time()
            
            # Determine current training phase
            if epoch < self.pretrain_fg_epochs:
                current_phase = "pretrain_fg"
            elif epoch < self.pretrain_fg_epochs + self.pretrain_h_epochs:
                current_phase = "pretrain_h"
            else:
                current_phase = "adversarial"

            train_metrics = self.train_epoch(epoch, current_phase)
            
            log_str_epoch = f"--- Epoch [{epoch+1}/{self.num_epochs}] Phase [{current_phase}] Summary --- | Time: {time.time() - start_epoch_time:.2f}s"
            for key, val in train_metrics.items():
                log_str_epoch += f" | Train {key}: {val:.4f}"
                if self.tensorboard_writer:
                    self.tensorboard_writer.add_scalar(f'Train/{key}', val, epoch)
            
            val_metrics = {}
            if self.dataloaders.get('val'):
                val_metrics = self._run_eval_epoch(self.dataloaders['val'])
                for key, val in val_metrics.items():
                    log_str_epoch += f" | {key}: {val:.4f}" # Val metrics already have 'val_' prefix
                    if self.tensorboard_writer:
                         self.tensorboard_writer.add_scalar(f'Validation/{key.replace("val_", "")}', val, epoch)
            
            logging.info(log_str_epoch)

            # Learning rate scheduling
            if self.scheduler_fg: self.scheduler_fg.step()
            if self.scheduler_h: self.scheduler_h.step()
            if self.tensorboard_writer:
                if self.scheduler_fg: self.tensorboard_writer.add_scalar('LR/optimizer_fg', self.optimizer_fg.param_groups[0]['lr'], epoch)
                if self.scheduler_h: self.tensorboard_writer.add_scalar('LR/optimizer_h', self.optimizer_h.param_groups[0]['lr'], epoch)

            # Checkpointing
            if (epoch + 1) % self.config['logging'].get('checkpoint_interval_epochs', 10) == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            if self.dataloaders.get('val') and self.config['logging'].get('save_best_model', True):
                metric_to_monitor = self.config['logging'].get('best_model_metric', 'val_accuracy_g')
                current_metric_val = val_metrics.get(metric_to_monitor)
                if current_metric_val is not None:
                    mode = self.config['logging'].get('best_model_metric_mode', 'max')
                    if (mode == 'max' and current_metric_val > self.best_val_metric) or \
                       (mode == 'min' and current_metric_val < self.best_val_metric):
                        self.best_val_metric = current_metric_val
                        logging.info(f"New best validation metric ({metric_to_monitor}): {self.best_val_metric:.4f}. Saving best model...")
                        self.save_checkpoint(epoch, is_best=True)
        
        logging.info(f"Training finished. Total time: {(time.time() - start_total_time)/3600:.2f} hours.")
        if self.tensorboard_writer:
            self.tensorboard_writer.close()

    def evaluate_model(self, checkpoint_path: Optional[str] = None, on_test_set: bool = True) -> Dict[str, float]:
        """Evaluates the model on the test set (or validation set)."""
        if checkpoint_path:
            if not self.load_checkpoint(checkpoint_path):
                logging.error("Failed to load checkpoint for evaluation. Aborting.")
                return {}
        elif not on_test_set and not self.dataloaders.get('val'):
             logging.warning("Validation set not available for evaluation.")
             return {}
        elif on_test_set and not self.dataloaders.get('test'):
            logging.error("Test set not available for evaluation.")
            return {}

        data_loader_to_use = self.dataloaders['test'] if on_test_set else self.dataloaders['val']
        set_name = "Test" if on_test_set else "Validation"
        
        logging.info(f"Starting evaluation on {set_name} set...")
        eval_metrics = self._run_eval_epoch(data_loader_to_use)
        
        log_str_eval = f"--- Evaluation Summary on {set_name} Set ---"
        for key, val in eval_metrics.items():
            log_str_eval += f" | {key.replace('val_', '')}: {val:.4f}" # Remove 'val_' prefix for general eval
        logging.info(log_str_eval)
        
        return eval_metrics

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Saves a model checkpoint."""
        checkpoint_data = {
            'epoch': epoch + 1,
            'config': self.config, # Save config for reference
            'feature_extractor_f_state_dict': self.feature_extractor_f.state_dict(),
            'task_classifier_g_state_dict': self.task_classifier_g.state_dict(),
            'bias_predictor_h_state_dict': self.bias_predictor_h.state_dict(),
            'optimizer_fg_state_dict': self.optimizer_fg.state_dict(),
            'optimizer_h_state_dict': self.optimizer_h.state_dict(),
            'best_val_metric': self.best_val_metric
        }
        if self.scheduler_fg:
            checkpoint_data['scheduler_fg_state_dict'] = self.scheduler_fg.state_dict()
        if self.scheduler_h:
            checkpoint_data['scheduler_h_state_dict'] = self.scheduler_h.state_dict()

        filename = f"checkpoint_epoch_{epoch+1}.pth"
        if is_best:
            filename = "best_model.pth"
        
        filepath = os.path.join(self.checkpoints_dir, filename)
        torch.save(checkpoint_data, filepath)
        logging.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, checkpoint_path: str, load_optimizers_schedulers: bool = True) -> bool:
        """Loads a model checkpoint."""
        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint file not found: {checkpoint_path}")
            return False
        
        try:
            logging.info(f"Loading checkpoint from {checkpoint_path}...")
            checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
            
            self.feature_extractor_f.load_state_dict(checkpoint_data['feature_extractor_f_state_dict'])
            self.task_classifier_g.load_state_dict(checkpoint_data['task_classifier_g_state_dict'])
            self.bias_predictor_h.load_state_dict(checkpoint_data['bias_predictor_h_state_dict'])
            
            if load_optimizers_schedulers:
                self.optimizer_fg.load_state_dict(checkpoint_data['optimizer_fg_state_dict'])
                self.optimizer_h.load_state_dict(checkpoint_data['optimizer_h_state_dict'])
                if self.scheduler_fg and 'scheduler_fg_state_dict' in checkpoint_data:
                    self.scheduler_fg.load_state_dict(checkpoint_data['scheduler_fg_state_dict'])
                if self.scheduler_h and 'scheduler_h_state_dict' in checkpoint_data:
                    self.scheduler_h.load_state_dict(checkpoint_data['scheduler_h_state_dict'])
                self.current_epoch = checkpoint_data.get('epoch', 0)
                self.best_val_metric = checkpoint_data.get('best_val_metric', self.best_val_metric)
            
            logging.info(f"Checkpoint loaded successfully. Resuming from epoch {self.current_epoch}.")
            return True
        except Exception as e:
            logging.error(f"Error loading checkpoint: {e}", exc_info=True)
            return False

if __name__ == '__main__':
    # This block is for conceptual testing or direct invocation if ever needed.
    # Full testing requires setting up dummy config, models, optimizers, dataloaders.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Trainer class defined. To use, instantiate and call .train() or .evaluate_model()")
    logging.info("Example: python main.py --config configs/your_config.yaml --mode train")
