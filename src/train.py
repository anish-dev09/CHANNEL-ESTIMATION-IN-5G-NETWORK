"""
Training script for AI-assisted channel estimation models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
from typing import Dict, Tuple
from torch.utils.tensorboard import SummaryWriter

from ai_models import get_model, ChannelEstimationLoss
from utils import (load_config, set_seed, get_device, save_checkpoint,
                   load_checkpoint, count_parameters)


class ChannelDataset(Dataset):
    """PyTorch dataset for channel estimation."""
    
    def __init__(self, data_path: str, normalize: bool = True):
        """Load dataset from file."""
        data = np.load(data_path)
        
        # Extract data
        self.rx_symbols = data['rx_symbols']
        self.H_ls = data['H_ls']
        self.H_true = data['H_true']
        self.pilot_mask = data['pilot_mask']
        self.snr_db = data['snr_db']
        
        self.normalize = normalize
        self.num_samples = len(self.rx_symbols)
        
        # Precompute normalization stats
        if self.normalize:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """Compute mean and std for normalization."""
        # Use first RX and TX for simplicity
        rx = self.rx_symbols[:, :, 0, :]
        H_ls = self.H_ls[:, :, 0, 0, :]
        H_true = self.H_true[:, :, 0, 0, :]
        
        # Compute stats on real and imaginary separately
        self.rx_mean = np.mean([rx.real.mean(), rx.imag.mean()])
        self.rx_std = np.mean([rx.real.std(), rx.imag.std()])
        
        self.H_ls_mean = np.mean([H_ls.real.mean(), H_ls.imag.mean()])
        self.H_ls_std = np.mean([H_ls.real.std(), H_ls.imag.std()])
        
        self.H_true_mean = np.mean([H_true.real.mean(), H_true.imag.mean()])
        self.H_true_std = np.mean([H_true.real.std(), H_true.imag.std()])
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sample."""
        # Take first RX and TX antenna (can be extended)
        rx = self.rx_symbols[idx, :, 0, :]  # (num_symbols, num_subcarriers)
        H_ls = self.H_ls[idx, :, 0, 0, :]  # (num_symbols, num_subcarriers)
        H_true = self.H_true[idx, :, 0, 0, :]  # (num_symbols, num_subcarriers)
        pilot_mask = self.pilot_mask[idx]  # (num_symbols, num_subcarriers)
        
        # Convert to real representation (separate real and imaginary)
        rx_real = np.stack([rx.real, rx.imag], axis=0)  # (2, num_symbols, num_subcarriers)
        H_ls_real = np.stack([H_ls.real, H_ls.imag], axis=0)
        H_true_real = np.stack([H_true.real, H_true.imag], axis=0)
        
        # Normalize
        if self.normalize:
            rx_real = (rx_real - self.rx_mean) / (self.rx_std + 1e-8)
            H_ls_real = (H_ls_real - self.H_ls_mean) / (self.H_ls_std + 1e-8)
            H_true_real = (H_true_real - self.H_true_mean) / (self.H_true_std + 1e-8)
        
        # Combine inputs: [rx_real, rx_imag, H_ls_real, H_ls_imag, pilot_mask]
        inputs = np.concatenate([
            rx_real,
            H_ls_real,
            pilot_mask[np.newaxis, :, :]
        ], axis=0)  # (5, num_symbols, num_subcarriers)
        
        # Convert to torch tensors
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(H_true_real).float()
        pilot_mask = torch.from_numpy(pilot_mask).float()
        
        return inputs, targets, pilot_mask


class Trainer:
    """Training manager."""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 val_loader: DataLoader, config: Dict, device: torch.device):
        """Initialize trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        loss_config = config['training']['loss_weights']
        self.criterion = ChannelEstimationLoss(
            loss_type=config['training']['loss'],
            channel_weight=loss_config['channel_mse'],
            pilot_weight=loss_config.get('ber_penalty', 0.0)
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Logging
        log_dir = Path(config['paths']['log_dir']) / 'tensorboard'
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def _create_optimizer(self):
        """Create optimizer."""
        opt_name = self.config['training']['optimizer'].lower()
        lr = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if opt_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif opt_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, 
                           momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_name = self.config['training']['lr_scheduler'].lower()
        
        if scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config['training']['epochs']
            )
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=30, 
                gamma=0.1
            )
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                patience=10
            )
        else:
            return None
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch+1}')
        for batch_idx, (inputs, targets, pilot_mask) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            pilot_mask = pilot_mask.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets, pilot_mask)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['gradient_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['training']['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            if batch_idx % self.config['logging']['log_interval'] == 0:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), step)
        
        return total_loss / len(self.train_loader)
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets, pilot_mask in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                pilot_mask = pilot_mask.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets, pilot_mask)
                
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config['training']['epochs']}")
        print("="*60 + "\n")
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Log
            self.writer.add_scalar('train/epoch_loss', train_loss, epoch)
            self.writer.add_scalar('val/loss', val_loss, epoch)
            self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"\nEpoch {epoch+1}/{self.config['training']['epochs']}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss: {val_loss:.6f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                
                if self.config['training']['checkpoint']['save_best']:
                    save_path = Path(self.config['paths']['model_dir']) / 'best_model.pth'
                    save_checkpoint(self.model, self.optimizer, epoch, val_loss, save_path)
                    print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
            else:
                self.epochs_without_improvement += 1
            
            # Periodic checkpoint
            if (epoch + 1) % self.config['training']['checkpoint']['save_freq'] == 0:
                save_path = Path(self.config['paths']['model_dir']) / f'checkpoint_epoch_{epoch+1}.pth'
                save_checkpoint(self.model, self.optimizer, epoch, val_loss, save_path)
            
            # Early stopping
            if self.config['training']['early_stopping']['enabled']:
                patience = self.config['training']['early_stopping']['patience']
                if self.epochs_without_improvement >= patience:
                    print(f"\nEarly stopping triggered after {patience} epochs without improvement")
                    break
        
        print("\n" + "="*60)
        print("Training Completed!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print("="*60)
        
        self.writer.close()


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train channel estimation model')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, default='cnn', 
                       choices=['cnn', 'lstm', 'hybrid', 'resnet'],
                       help='Model architecture')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Override config with CLI args
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    
    # Create directories
    Path(config['paths']['model_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['log_dir']).mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device(args.device)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = ChannelDataset(
        Path(args.data_dir) / 'train.npz',
        normalize=config['dataset']['normalize']
    )
    val_dataset = ChannelDataset(
        Path(args.data_dir) / 'val.npz',
        normalize=config['dataset']['normalize']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['compute']['num_workers'],
        pin_memory=config['compute']['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['compute']['num_workers'],
        pin_memory=config['compute']['pin_memory']
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    model_config = config['model'][args.model] if args.model in config['model'] else {}
    model = get_model(args.model, model_config)
    
    # Create trainer and train
    trainer = Trainer(model, train_loader, val_loader, config, device)
    trainer.train()
    
    # Save final model
    final_path = Path(config['paths']['model_dir']) / 'final_model.pth'
    save_checkpoint(model, trainer.optimizer, trainer.current_epoch, 
                   trainer.best_val_loss, final_path)
    print(f"\nFinal model saved to {final_path}")


if __name__ == '__main__':
    main()
