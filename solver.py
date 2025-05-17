import torch
import torch.optim as optim
import torch.nn as nn
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np

from network import grad_reverse 
from utils import calculate_accuracy, get_grl_alpha_dann
import torch.nn.functional as F # For softmax if needed in test

class Solver:
    def __init__(self, args, data_loader_train, data_loader_val, model_f, model_g, model_h, device):
        self.args = args
        self.data_loader_train = data_loader_train
        self.data_loader_val = data_loader_val # This can also be the test loader for 'test' mode
        self.model_f = model_f
        self.model_g = model_g
        self.model_h = model_h
        self.device = device

        params_fg = list(self.model_f.parameters()) + list(self.model_g.parameters())
        self.optimizer_fg = optim.Adam(params_fg, lr=args.lr_fg, betas=(args.beta1, args.beta2))
        self.optimizer_h = optim.Adam(self.model_h.parameters(), lr=args.lr_h, betas=(args.beta1, args.beta2))

        self.criterion_main_task = nn.CrossEntropyLoss()
        self.criterion_bias = nn.CrossEntropyLoss()

        log_dir_base = os.path.join(args.output_dir, args.experiment_name, 'logs')
        os.makedirs(log_dir_base, exist_ok=True)
        current_time_log = time.strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(log_dir_base, current_time_log if args.mode == 'train' else "test_run")
        self.writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")

        self.checkpoint_dir = os.path.join(args.output_dir, args.experiment_name, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.total_train_iters_adversarial = len(self.data_loader_train) * args.epochs if self.data_loader_train else 0
        self.current_iter_adversarial = 0
        self.pretrain_iters = len(self.data_loader_train) * args.pretrain_fg_epochs if self.data_loader_train else 0

    def _train_one_epoch_pretrain(self, epoch_num):
        # ... (same as before)
        self.model_f.train()
        self.model_g.train()
        self.model_h.eval() 

        running_loss_lc_pre = 0.0
        running_main_acc_pre = 0.0
        
        progress_bar = tqdm(enumerate(self.data_loader_train), total=len(self.data_loader_train),
                            desc=f"Pretrain Epoch {epoch_num+1}/{self.args.pretrain_fg_epochs}")

        for i, (images, main_labels, _) in progress_bar: 
            images, main_labels = images.to(self.device), main_labels.to(self.device)
            self.optimizer_fg.zero_grad()
            features = self.model_f(images)
            main_task_preds = self.model_g(features)
            loss_lc = self.criterion_main_task(main_task_preds, main_labels)
            loss_lc.backward()
            self.optimizer_fg.step()
            running_loss_lc_pre += loss_lc.item()
            running_main_acc_pre += calculate_accuracy(main_task_preds, main_labels)
            progress_bar.set_postfix(pre_loss_lc=f"{running_loss_lc_pre/(i+1):.3f}",
                                     pre_acc_main=f"{running_main_acc_pre/(i+1):.3f}")
        
        avg_loss_lc_pre = running_loss_lc_pre / len(self.data_loader_train)
        avg_main_acc_pre = running_main_acc_pre / len(self.data_loader_train)
        self.writer.add_scalar('PretrainEpoch/Loss_Lc', avg_loss_lc_pre, epoch_num)
        self.writer.add_scalar('PretrainEpoch/Accuracy_Main', avg_main_acc_pre, epoch_num)
        print(f"Pretrain Epoch {epoch_num+1} Summary: Avg Lc={avg_loss_lc_pre:.4f}, Avg MainAcc={avg_main_acc_pre:.4f}")


    def _train_one_epoch_adversarial(self, epoch_num):
        # ... (same as before)
        self.model_f.train()
        self.model_g.train()
        self.model_h.train()

        epoch_loss_lc = 0.0; epoch_loss_h = 0.0; epoch_loss_adv_f = 0.0
        epoch_loss_entropy_f = 0.0; epoch_main_acc = 0.0; epoch_bias_acc_h_train = 0.0

        # Adjust epoch display for adversarial phase
        adv_epoch_display = epoch_num - self.args.pretrain_fg_epochs + 1
        total_adv_epochs_display = self.args.epochs - self.args.pretrain_fg_epochs
        
        progress_bar = tqdm(enumerate(self.data_loader_train), total=len(self.data_loader_train),
                            desc=f"Adv. Epoch {adv_epoch_display}/{total_adv_epochs_display}")

        for i, (images, main_labels, bias_labels) in progress_bar:
            # global_step for TB logging, relative to start of adversarial phase
            global_step = (epoch_num - self.args.pretrain_fg_epochs) * len(self.data_loader_train) + i
            images,main_labels,bias_labels = images.to(self.device),main_labels.to(self.device),bias_labels.to(self.device)

            alpha_grl = get_grl_alpha_dann(self.current_iter_adversarial,
                                           self.total_train_iters_adversarial,
                                           gamma=self.args.grl_gamma, adapt_iter_offset=0)
            self.current_iter_adversarial +=1
            self.writer.add_scalar('Params/Alpha_GRL', alpha_grl, global_step)

            self.optimizer_h.zero_grad()
            with torch.no_grad(): features_for_h = self.model_f(images).detach()
            bias_preds_for_h = self.model_h(features_for_h)
            loss_h = self.criterion_bias(bias_preds_for_h, bias_labels)
            loss_h.backward(); self.optimizer_h.step()
            
            self.optimizer_fg.zero_grad()
            features = self.model_f(images)
            main_task_preds = self.model_g(features)
            reversed_features = grad_reverse(features, alpha_grl)
            bias_preds_adv = self.model_h(reversed_features)
            
            loss_lc = self.criterion_main_task(main_task_preds, main_labels)
            loss_adv_for_f = self.criterion_bias(bias_preds_adv, bias_labels) * self.args.adv_mu
            
            loss_entropy_for_f = torch.tensor(0.0).to(self.device); current_entropy_val = 0.0
            if self.args.adv_lambda > 0:
                softmax_bias_preds_adv = F.softmax(bias_preds_adv, dim=1)
                current_entropy_val = -torch.sum(softmax_bias_preds_adv * torch.log(softmax_bias_preds_adv + 1e-8), dim=1).mean()
                loss_entropy_for_f = -self.args.adv_lambda * current_entropy_val
                
            total_loss_fg = loss_lc + loss_adv_for_f + loss_entropy_for_f
            total_loss_fg.backward(); self.optimizer_fg.step()

            epoch_loss_lc += loss_lc.item(); epoch_loss_h += loss_h.item()
            epoch_loss_adv_f += loss_adv_for_f.item(); epoch_loss_entropy_f += loss_entropy_for_f.item()
            epoch_main_acc += calculate_accuracy(main_task_preds, main_labels)
            epoch_bias_acc_h_train += calculate_accuracy(bias_preds_for_h, bias_labels)
            # ... (TensorBoard logging for iter can be added back if desired)
            progress_bar.set_postfix(Lc=f"{epoch_loss_lc/(i+1):.3f}",Lh=f"{epoch_loss_h/(i+1):.3f}",AccM=f"{epoch_main_acc/(i+1):.3f}")
        
        self.writer.add_scalar('TrainEpochAdv/AvgLoss_Lc', epoch_loss_lc / len(self.data_loader_train), epoch_num)
        # ...

    def train(self):
        # ... (same as before)
        print("--- Starting Training ---")
        if self.args.pretrain_fg_epochs > 0:
            print(f"--- Phase 1: Pre-training f and g for {self.args.pretrain_fg_epochs} epochs ---")
            for epoch in range(self.args.pretrain_fg_epochs):
                self._train_one_epoch_pretrain(epoch)
                if self.data_loader_val and self.args.do_eval:
                    self.evaluate(epoch, is_pretrain=True)
        
        print(f"--- Phase 2: Adversarial training for {self.args.epochs - self.args.pretrain_fg_epochs} epochs ---")
        self.current_iter_adversarial = 0 
        self.total_train_iters_adversarial = len(self.data_loader_train) * (self.args.epochs - self.args.pretrain_fg_epochs)

        for epoch in range(self.args.pretrain_fg_epochs, self.args.epochs):
            self._train_one_epoch_adversarial(epoch)
            if self.data_loader_val and self.args.do_eval:
                self.evaluate(epoch, is_pretrain=False)
            if (epoch + 1) % self.args.save_every_epochs == 0 or (epoch + 1) == self.args.epochs:
                self.save_model(epoch + 1)
        self.writer.close()
        print("--- Training Finished ---")


    def evaluate(self, epoch_num, is_pretrain=False, is_test_mode=False):
        # ... (same as before, but consider is_test_mode for logging prefix)
        self.model_f.eval(); self.model_g.eval(); self.model_h.eval()
        val_loss_lc_epoch = 0.0; val_main_acc_epoch = 0.0
        val_loss_bias_epoch = 0.0; val_bias_acc_epoch = 0.0
        
        # Use self.data_loader_val (which could be train, val, or test loader depending on mode)
        loader_to_eval = self.data_loader_val # Or pass specific loader if 'test' mode uses a different one
        if not loader_to_eval:
            print("Warning: No data loader provided for evaluation/test.")
            return None, None, None, None # Return Nones if no loader

        with torch.no_grad():
            for images_val, main_labels_val, bias_labels_val in loader_to_eval:
                images_val, main_labels_val, bias_labels_val = images_val.to(self.device), main_labels_val.to(self.device), bias_labels_val.to(self.device)
                features_val = self.model_f(images_val)
                main_task_preds_val = self.model_g(features_val)
                loss_lc_val = self.criterion_main_task(main_task_preds_val, main_labels_val)
                val_loss_lc_epoch += loss_lc_val.item()
                val_main_acc_epoch += calculate_accuracy(main_task_preds_val, main_labels_val)

                if not is_pretrain or is_test_mode: # Evaluate bias predictor in adv phase or test mode
                    if bias_labels_val is not None and -1 not in bias_labels_val : # Ensure bias labels are valid
                        bias_preds_val = self.model_h(features_val)
                        loss_bias_val = self.criterion_bias(bias_preds_val, bias_labels_val)
                        val_loss_bias_epoch += loss_bias_val.item()
                        val_bias_acc_epoch += calculate_accuracy(bias_preds_val, bias_labels_val)
        
        avg_val_loss_lc = val_loss_lc_epoch / len(loader_to_eval)
        avg_val_main_acc = val_main_acc_epoch / len(loader_to_eval)
        
        log_prefix = "Test" if is_test_mode else ("ValEpochPretrain" if is_pretrain else "ValEpochAdv")
        self.writer.add_scalar(f'{log_prefix}/Loss_Lc', avg_val_loss_lc, epoch_num)
        self.writer.add_scalar(f'{log_prefix}/Accuracy_Main', avg_val_main_acc, epoch_num)

        if not is_pretrain or is_test_mode:
            avg_val_loss_bias = val_loss_bias_epoch / len(loader_to_eval) if len(loader_to_eval) > 0 and val_loss_bias_epoch > 0 else 0.0
            avg_val_bias_acc = val_bias_acc_epoch / len(loader_to_eval) if len(loader_to_eval) > 0 and val_bias_acc_epoch > 0 else 0.0
            self.writer.add_scalar(f'{log_prefix}/Loss_Bias_h_on_f', avg_val_loss_bias, epoch_num)
            self.writer.add_scalar(f'{log_prefix}/Accuracy_Bias_h_on_f', avg_val_bias_acc, epoch_num)
            print(f"Epoch {epoch_num+1} {log_prefix}: MainLoss={avg_val_loss_lc:.4f}, MainAcc={avg_val_main_acc:.4f}, "
                  f"BiasLossH(f)={avg_val_loss_bias:.4f}, BiasAccH(f)={avg_val_bias_acc:.4f}")
        else:
            print(f"Epoch {epoch_num+1} {log_prefix} (Pretrain): MainLoss={avg_val_loss_lc:.4f}, MainAcc={avg_val_main_acc:.4f}")
        return avg_val_main_acc, avg_val_loss_lc, avg_val_bias_acc, avg_val_loss_bias


    def test_and_save_results(self, output_npy_file="test_results.npy"):
        """Runs evaluation on self.data_loader_val (assumed to be test loader) and saves results."""
        print(f"--- Running Test Mode & Saving Results to {output_npy_file} ---")
        self.model_f.eval()
        self.model_g.eval()
        self.model_h.eval()

        all_main_preds_probs = []
        all_main_preds_labels = []
        all_main_true_labels = []
        all_bias_preds_probs = []
        all_bias_preds_labels = []
        all_bias_true_labels = []

        if not self.data_loader_val:
            print("Error: No data loader available for testing (self.data_loader_val is None).")
            return

        with torch.no_grad():
            for images, main_labels, bias_labels in tqdm(self.data_loader_val, desc="Testing"):
                images = images.to(self.device)
                main_labels_cpu = main_labels.cpu().numpy()
                bias_labels_cpu = bias_labels.cpu().numpy()

                features = self.model_f(images)
                main_task_logits = self.model_g(features)
                main_task_probs = F.softmax(main_task_logits, dim=1).cpu().numpy()
                main_task_pred_labels = torch.argmax(main_task_logits, dim=1).cpu().numpy()

                all_main_preds_probs.append(main_task_probs)
                all_main_preds_labels.append(main_task_pred_labels)
                all_main_true_labels.append(main_labels_cpu)

                if bias_labels is not None and -1 not in bias_labels: # If bias labels are valid
                    bias_logits = self.model_h(features)
                    bias_probs = F.softmax(bias_logits, dim=1).cpu().numpy()
                    bias_pred_labels = torch.argmax(bias_logits, dim=1).cpu().numpy()
                    all_bias_preds_probs.append(bias_probs)
                    all_bias_preds_labels.append(bias_pred_labels)
                    all_bias_true_labels.append(bias_labels_cpu)

        # Concatenate results from all batches
        results = {
            'main_preds_probs': np.concatenate(all_main_preds_probs, axis=0),
            'main_preds_labels': np.concatenate(all_main_preds_labels, axis=0),
            'main_true_labels': np.concatenate(all_main_true_labels, axis=0)
        }
        if all_bias_true_labels: # If bias was evaluated
            results['bias_preds_probs'] = np.concatenate(all_bias_preds_probs, axis=0)
            results['bias_preds_labels'] = np.concatenate(all_bias_preds_labels, axis=0)
            results['bias_true_labels'] = np.concatenate(all_bias_true_labels, axis=0)

        # Calculate and print overall accuracies
        main_acc = calculate_accuracy(torch.from_numpy(results['main_preds_labels']), torch.from_numpy(results['main_true_labels']))
        print(f"Overall Test Main Accuracy: {main_acc:.4f}")
        if 'bias_true_labels' in results:
            bias_acc = calculate_accuracy(torch.from_numpy(results['bias_preds_labels']), torch.from_numpy(results['bias_true_labels']))
            print(f"Overall Test Bias Accuracy (h on f(x)): {bias_acc:.4f}")

        # Save to .npy file
        output_path = os.path.join(self.args.output_dir, self.args.experiment_name, output_npy_file)
        np.save(output_path, results)
        print(f"Test results saved to: {output_path}")
        print("--- Test Mode Finished ---")


    def save_model(self, epoch_num):
        # ... (same as before)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch_num}.pth")
        torch.save({
            'epoch': epoch_num,
            'model_f_state_dict': self.model_f.state_dict(),
            'model_g_state_dict': self.model_g.state_dict(),
            'model_h_state_dict': self.model_h.state_dict(),
            'optimizer_fg_state_dict': self.optimizer_fg.state_dict(),
            'optimizer_h_state_dict': self.optimizer_h.state_dict(),
            'args': self.args 
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


    def load_model(self, checkpoint_path):
        # ... (same as before)
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}"); return None
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model_f.load_state_dict(checkpoint['model_f_state_dict'])
        self.model_g.load_state_dict(checkpoint['model_g_state_dict'])
        self.model_h.load_state_dict(checkpoint['model_h_state_dict'])
        # Load optimizers only if continuing training, not strictly needed for eval/test
        if self.args.mode == 'train' and 'optimizer_fg_state_dict' in checkpoint:
            self.optimizer_fg.load_state_dict(checkpoint['optimizer_fg_state_dict'])
            self.optimizer_h.load_state_dict(checkpoint['optimizer_h_state_dict'])
        
        # It's good practice to also load args from checkpoint if resuming training
        # to ensure consistency, or at least log/compare them.
        # For now, we just print the epoch.
        loaded_epoch = checkpoint.get('epoch', 0) # Default to 0 if epoch not in checkpoint
        print(f"Loaded checkpoint from {checkpoint_path}, trained up to epoch {loaded_epoch}")
        return loaded_epoch

