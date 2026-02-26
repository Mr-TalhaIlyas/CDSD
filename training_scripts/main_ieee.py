#!/usr/bin/env python
"""
Simple supervised training for action classification.
Fine-tunes NTU-RGBD pretrained HyperGCN on custom seizure dataset.
"""
#%%
import os

os.chdir(os.path.dirname(__file__))
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import yaml
import time
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torchmetrics.classification import MultilabelExactMatch
from torchmetrics.classification import MulticlassConfusionMatrix
import wandb

# For confusion matrix plotting in wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from feeders.feeder_ieee import Feeder, custom_collate_fn, LABEL_NAMES
from model.model_sup_logic import SkeletonACL_CLIP_Logic


def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class Trainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.args = yaml.safe_load(f)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        init_seed(self.args.get('seed', 1))

        self.n_classes = self.args['model_args']['num_class']
        self.n_concepts = self.args['model_args']['spatial_concepts'] + self.args['model_args']['temporal_concepts']
        self.USE_BINARY_CLASSES = self.args['USE_BINARY_CLASSES']
        if self.USE_BINARY_CLASSES:
            TOP_K = 1
        else: 
            TOP_K = 5
        # Data loaders
        self.data_loader = {}
        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.args['train_feeder_args']),
            batch_size=self.args['batch_size'],
            shuffle=True,
            num_workers=self.args['num_worker'],
            drop_last=True,
            collate_fn=custom_collate_fn,
            worker_init_fn=lambda _: init_seed(self.args.get('seed', 1)),
            pin_memory=True,
            persistent_workers=True if self.args['num_worker'] > 0 else False
        )
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.args['test_feeder_args']),
            batch_size=self.args['test_batch_size'],
            shuffle=False,
            num_workers=self.args['num_worker'],
            drop_last=False,
            collate_fn=custom_collate_fn,
            worker_init_fn=lambda _: init_seed(self.args.get('seed', 1)),
            pin_memory=True,
            persistent_workers=True if self.args['num_worker'] > 0 else False
        )
        
        # Model
        self.model = SkeletonACL_CLIP_Logic(self.args, self.device).to(self.device)
        
        # Optimizer and scheduler
        self.lr = float(self.args.get('base_lr', 1e-4))
        self.num_epochs = int(self.args.get('num_epoch', 110))
        self.warm_up_epochs = int(self.args.get('warm_up_epoch', 5))
        self.steps_per_epoch = len(self.data_loader['train'])
        self.total_steps = self.num_epochs * self.steps_per_epoch
        wd = float(self.args.get('weight_decay', 0.0005))
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=wd
        )
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.lr,
            total_steps=self.total_steps,
            epochs=self.num_epochs,
            steps_per_epoch=self.steps_per_epoch,
            pct_start=self.warm_up_epochs / self.num_epochs,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4
        )

        # Metrics
        self.train_acc = Accuracy(task='multiclass', num_classes=self.n_classes).to(self.device)
        self.train_top5 = Accuracy(task='multiclass', num_classes=self.n_classes, top_k=TOP_K).to(self.device)
        self.val_acc = Accuracy(task='multiclass', num_classes=self.n_classes).to(self.device)
        self.val_top5 = Accuracy(task='multiclass', num_classes=self.n_classes, top_k=TOP_K).to(self.device)

        self.train_concept_exact = MultilabelExactMatch(num_labels=self.n_concepts).to(self.device)
        self.val_concept_exact = MultilabelExactMatch(num_labels=self.n_concepts).to(self.device)
        # Bookkeeping
        self.work_dir = Path(self.args['work_dir'])
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.best_acc = 0.0
        self.best_epoch = 0

        self._print_init_info()
        
        # Initialize WandB
        self.init_wandb()

    def _print_init_info(self):
        print("\n" + "="*80)
        print("✅ TRAINER INITIALIZED")
        print("="*80)
        print(f"Work directory: {self.work_dir}")
        print(f"Device: {self.device}")
        print(f"Total epochs: {self.num_epochs}")
        print(f"Steps per epoch: {self.steps_per_epoch}")
        print(f"Batch size: {self.args['batch_size']}")
        print(f"Learning rate: {self.lr:.2e}")
        print(f"Weight decay: {self.args.get('weight_decay', 0.0005):.2e}")
        print(f"Warmup epochs: {self.warm_up_epochs}")
        print(f"Training samples: {len(self.data_loader['train'].dataset)}")
        print(f"Test samples: {len(self.data_loader['test'].dataset)}")
        print(f"Action classes: {self.n_classes}")
        print("="*80 + "\n")

    def init_wandb(self):
        if self.args.get('use_wandb', False):
            # Get action class names
            if self.USE_BINARY_CLASSES:
                self.action_names = ['Normal', 'Seizure']
            else:
                self.action_names = [LABEL_NAMES.get(i, f"class_{i}") for i in range(self.n_classes)]
            
            # Build comprehensive config
            wandb_config = {
                **self.args,
                'system': {
                    'device': str(self.device),
                    'cuda_available': torch.cuda.is_available(),
                    'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A',
                },
                'model': {
                    'total_params': sum(p.numel() for p in self.model.parameters()),
                    'trainable_params': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                },
                'dataset': {
                    'train_samples': len(self.data_loader['train'].dataset),
                    'test_samples': len(self.data_loader['test'].dataset),
                    'n_classes': self.n_classes,
                    'action_names': self.action_names,
                },
                'training': {
                    'optimizer': 'AdamW',
                    'scheduler': 'OneCycleLR',
                    'lr': self.lr,
                    'weight_decay': self.args.get('weight_decay', 0.0005),
                    'batch_size': self.args['batch_size'],
                    'num_epochs': self.num_epochs,
                    'warmup_epochs': self.warm_up_epochs,
                    'total_steps': self.total_steps,
                }
            }
            
            wandb.init(
                project=self.args.get('wandb_project', 'SkeletonACL'),
                entity=self.args.get('wandb_entity', None),
                name=self.args.get('wandb_name', 'action_classification'),
                config=wandb_config,
                resume='allow',
                save_code=True,
            )
            
            # Watch model for gradient tracking
            wandb.watch(self.model, log='gradients', log_freq=100)
            
            print("✅ WandB initialized")
        else:
            self.action_names = [LABEL_NAMES.get(i, f"class_{i}") for i in range(self.n_classes)]
            print("⚠️  WandB logging disabled")

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        self.train_acc.reset()
        self.train_top5.reset()
        self.train_concept_exact.reset()
        
        epoch_loss = 0.0
        batch_losses = []
        grad_norms = []
        
        # Per-class tracking
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        pbar = tqdm(
            enumerate(self.data_loader['train']),
            total=len(self.data_loader['train']),
            desc=f"Epoch {epoch}/{self.num_epochs} [TRAIN]",
            ncols=110
        )
        
        for step, (batch_data, batch_label, batch_concept_vecs, _) in pbar:
            batch_data = batch_data.to(self.device)
            batch_label = batch_label.to(self.device)
            concepts_gt = torch.cat(
                            (batch_concept_vecs['full_body'],
                            batch_concept_vecs['temporal']),
                            dim=1).float().to(self.device)
            if self.USE_BINARY_CLASSES:
                batch_label = (batch_label == 8).long().to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            y_pred, concept_probs, loss = self.model(batch_data, batch_label, concepts_gt)
            
            # Check for NaN loss
            if torch.isnan(loss):
                print(f"\n⚠️  NaN loss detected at step {step}!")
                if self.args.get('use_wandb', False):
                    wandb.alert(
                        title="NaN Loss Detected",
                        text=f"NaN loss at epoch {epoch}, step {step}",
                        level=wandb.AlertLevel.ERROR
                    )
                continue
            
            # Backward pass
            loss.backward()
            
            # Compute gradient norm before clipping
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            with torch.no_grad():
                self.model.logic_layers.clip_weights()
            # Accumulate loss
            epoch_loss += loss.item()
            batch_losses.append(loss.item())
            
            # Per-class accuracy tracking
            preds = y_pred.argmax(dim=1)
            for pred, label in zip(preds.cpu().numpy(), batch_label.cpu().numpy()):
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

            # Update metrics
            self.train_acc.update(y_pred, batch_label)
            self.train_top5.update(y_pred, batch_label)
            self.train_concept_exact.update((concept_probs > 0.5).long(), concepts_gt)

            # Progress display
            acc = self.train_acc.compute().item() * 100
            top5 = self.train_top5.compute().item() * 100
            c_acc = self.train_concept_exact.compute().item() * 100
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc:.1f}%",
                'top5': f"{top5:.1f}%",
                'c_acc': f"{c_acc:.1f}%"
            })
            
            # Log to WandB (every 50 steps)
            if self.args.get('use_wandb', False) and step % 50 == 0:
                global_step = epoch * self.steps_per_epoch + step
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_acc': acc,
                    'train/batch_top5': top5,
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'train/grad_norm': total_norm,
                    'train/step': global_step,
                }, step=global_step)

        # Compute final metrics
        epoch_loss /= len(self.data_loader['train'])
        acc = self.train_acc.compute().item() * 100
        top5 = self.train_top5.compute().item() * 100
        c_acc = self.train_concept_exact.compute().item() * 100
        
        # Per-class accuracy
        per_class_acc = {}
        for cls_idx in range(self.n_classes):
            if class_total[cls_idx] > 0:
                per_class_acc[self.action_names[cls_idx]] = class_correct[cls_idx] / class_total[cls_idx] * 100
            else:
                per_class_acc[self.action_names[cls_idx]] = 0.0

        print(f"\n{'='*70}")
        print(f"EPOCH {epoch} TRAIN: Loss={epoch_loss:.4f}, Acc={acc:.2f}%, Top5={top5:.2f}%, C_ACC={c_acc:.2f}%")
        print(f"{'='*70}")
        
        # Log epoch-level metrics to WandB
        if self.args.get('use_wandb', False):
            log_dict = {
                'train/epoch_loss': epoch_loss,
                'train/epoch_acc': acc,
                'train/epoch_top5': top5,
                'train/loss_std': np.std(batch_losses),
                'train/grad_norm_mean': np.mean(grad_norms),
                'train/grad_norm_max': np.max(grad_norms),
                'epoch': epoch,
            }
            # Log per-class accuracy
            for cls_name, cls_acc in per_class_acc.items():
                log_dict[f'train/class_acc/{cls_name}'] = cls_acc
            
            wandb.log(log_dict, step=epoch * self.steps_per_epoch)

        return epoch_loss, acc

    @torch.no_grad()
    def evaluate(self, epoch, split='test'):
        """Evaluate on test set with detailed metrics."""
        self.model.eval()
        self.val_acc.reset()
        self.val_top5.reset()
        self.val_concept_exact.reset()
        
        # Collect all predictions and labels for confusion matrix
        all_preds = []
        all_labels = []
        all_probs = []

        # Per-class tracking
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        pbar = tqdm(
            self.data_loader[split],
            desc=f"Epoch {epoch} [EVAL]",
            ncols=110
        )
        
        for batch_data, batch_label, batch_concept_vecs, _ in pbar:
            batch_data = batch_data.to(self.device)
            batch_label = batch_label.to(self.device)
            concepts_gt = torch.cat(
                            (batch_concept_vecs['full_body'],
                            batch_concept_vecs['temporal']),
                            dim=1).float().to(self.device)
            
            if self.USE_BINARY_CLASSES:
                batch_label = (batch_label == 8).long().to(self.device)
            # Forward pass (no loss computation needed)
            y_pred, cocnept_pred = self.model(batch_data, batch_label, concepts_gt)
            
            # Store predictions
            # probs = F.softmax(y_pred, dim=1)
            probs = y_pred # already done in model
            preds = y_pred.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            # Per-class tracking
            for pred, label in zip(preds.cpu().numpy(), batch_label.cpu().numpy()):
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1

            # Update metrics
            self.val_acc.update(y_pred, batch_label)
            self.val_top5.update(y_pred, batch_label)
            self.val_concept_exact.update(cocnept_pred, concepts_gt)
            
            acc = self.val_acc.compute().item() * 100
            top5 = self.val_top5.compute().item() * 100
            c_acc = self.val_concept_exact.compute().item() * 100
            
            pbar.set_postfix({
                'acc': f"{acc:.1f}%",
                'top5': f"{top5:.1f}%",
                'c_acc': f"{c_acc:.1f}%"
            })
            
        # Final metrics
        acc = self.val_acc.compute().item() * 100
        top5 = self.val_top5.compute().item() * 100
        c_acc = self.val_concept_exact.compute().item() * 100
        
        # Per-class accuracy
        per_class_acc = {}
        for cls_idx in range(self.n_classes):
            if class_total[cls_idx] > 0:
                per_class_acc[self.action_names[cls_idx]] = class_correct[cls_idx] / class_total[cls_idx] * 100
            else:
                per_class_acc[self.action_names[cls_idx]] = 0.0
        print(70*'*')
        print(f"EPOCH {epoch} EVAL: Acc={acc:.2f}%, Top5={top5:.2f}%, C_ACC={c_acc:.2f}%")
        print("Per-class accuracy:")
        for cls_name, cls_acc in per_class_acc.items():
            print(f"  {cls_name}: {cls_acc:.1f}%")
        print(70*'*')
        if self.args.get('use_wandb', False):
            log_dict = {
                'test/acc': acc,
                'test/top5': top5,
                'epoch': epoch
            }
            
            # Log per-class accuracy
            for cls_name, cls_acc in per_class_acc.items():
                log_dict[f'test/class_acc/{cls_name}'] = cls_acc
            
            # Compute and log confusion matrix
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=int)
            for pred, label in zip(all_preds, all_labels):
                conf_matrix[label, pred] += 1
            
            # Log confusion matrix as image
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(conf_matrix, cmap='Blues')
            
            # Add labels
            ax.set_xticks(range(self.n_classes))
            ax.set_yticks(range(self.n_classes))
            ax.set_xticklabels(self.action_names, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(self.action_names, fontsize=8)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.set_title(f'Confusion Matrix - Epoch {epoch}')
            
            # Add text annotations
            for i in range(self.n_classes):
                for j in range(self.n_classes):
                    text = ax.text(j, i, conf_matrix[i, j],
                                   ha="center", va="center", 
                                   color="white" if conf_matrix[i, j] > conf_matrix.max()/2 else "black",
                                   fontsize=7)
            
            plt.colorbar(im)
            plt.tight_layout()
            
            log_dict['test/confusion_matrix'] = wandb.Image(fig)
            plt.close(fig)
            
            # Log prediction confidence distribution
            all_probs = np.array(all_probs)
            max_probs = all_probs.max(axis=1)
            log_dict['test/confidence_mean'] = np.mean(max_probs)
            log_dict['test/confidence_std'] = np.std(max_probs)
            
            # Confidence for correct vs incorrect predictions
            correct_mask = all_preds == all_labels
            if correct_mask.sum() > 0:
                log_dict['test/confidence_correct'] = np.mean(max_probs[correct_mask])
            if (~correct_mask).sum() > 0:
                log_dict['test/confidence_incorrect'] = np.mean(max_probs[~correct_mask])
            
            wandb.log(log_dict)
        
        return acc

    def save_checkpoint(self, epoch, is_best=False):
        if is_best:
            best_path = self.work_dir / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'best_acc': self.best_acc,
            }, best_path)
            print(f"✅ New best model saved! Accuracy: {self.best_acc:.2f}%")

    def train(self):
        """Complete training loop."""
        print("\n" + "="*80)
        print("🚀 STARTING TRAINING")
        print("="*80 + "\n")
        
        for epoch in range(self.num_epochs):
            t0 = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Evaluation
            if (epoch + 1) % self.args.get('eval_interval', 1) == 0:
                test_acc = self.evaluate(epoch)
                
                is_best = test_acc > self.best_acc
                if is_best:
                    self.best_acc = test_acc
                    self.best_epoch = epoch
                    self.save_checkpoint(epoch, is_best=True)
            
            t1 = time.time()
            print(f"⏱️  Epoch {epoch} done in {t1-t0:.1f}s. Best: {self.best_acc:.2f}% (epoch {self.best_epoch})\n")
        
        print("\n" + "="*80)
        print("🎉 TRAINING COMPLETE!")
        print(f"Best Test Accuracy: {self.best_acc:.2f}% (Epoch {self.best_epoch})")
        print("="*80 + "\n")
        
        if self.args.get('use_wandb', False):
            wandb.finish()


def main():
    cfg = os.path.join(os.path.dirname(__file__), 'config', 'szr', 'config_skeleton_ieee.yaml')
    print(f"Using config: {cfg}")
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == '__main__':
    main()
#%%