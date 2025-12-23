"""
Phase 6-7: Advanced Model Training (LSTM, Hybrid, ResNet)
==========================================================

Trains all advanced model architectures and compares them.

Usage:
    python run_phase6_advanced_training.py --model lstm --epochs 50
    python run_phase6_advanced_training.py --model hybrid --epochs 50
    python run_phase6_advanced_training.py --model resnet --epochs 50
    python run_phase6_advanced_training.py --all --epochs 50       # Train all
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import json
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_models import get_model, ChannelEstimationLoss
from utils import load_config, set_seed, linear2db


class ChannelEstimationDataset(Dataset):
    """PyTorch Dataset for channel estimation."""
    
    def __init__(self, data_path: str, normalize: bool = True, 
                 model_type: str = 'cnn'):
        """
        Load dataset from NPZ file.
        
        Args:
            data_path: Path to .npz file
            normalize: Whether to normalize inputs
            model_type: Model type (affects input preparation)
        """
        print(f"Loading dataset from {data_path}...")
        data = np.load(data_path, allow_pickle=True)
        
        self.rx_symbols = data['rx_symbols']
        self.H_ls = data['H_ls']
        self.H_true = data['H_true']
        self.pilot_mask = data['pilot_mask']
        self.snr_db = data['snr_db']
        self.model_type = model_type
        
        self.num_samples = self.rx_symbols.shape[0]
        self.normalize = normalize
        
        # Compute normalization statistics
        if normalize:
            self._compute_stats()
        
        print(f"  Loaded {self.num_samples} samples for {model_type}")
    
    def _compute_stats(self):
        """Compute normalization statistics."""
        rx = self.rx_symbols[:, :, 0, :]
        H_ls = self.H_ls[:, :, 0, 0, :]
        
        self.rx_std = np.std(np.abs(rx)) + 1e-8
        self.H_ls_std = np.std(np.abs(H_ls)) + 1e-8
        self.H_true_std = np.std(np.abs(self.H_true[:, :, 0, 0, :])) + 1e-8
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a single sample."""
        rx = self.rx_symbols[idx, :, 0, :]  # (14, 599)
        H_ls = self.H_ls[idx, :, 0, 0, :]   # (14, 599)
        H_true = self.H_true[idx, :, 0, 0, :] # (14, 599)
        pilot_mask = self.pilot_mask[idx]   # (14, 599)
        
        # Convert complex to real
        rx_real = np.stack([rx.real, rx.imag], axis=0)
        H_ls_real = np.stack([H_ls.real, H_ls.imag], axis=0)
        H_true_real = np.stack([H_true.real, H_true.imag], axis=0)
        
        if self.normalize:
            rx_real = rx_real / self.rx_std
            H_ls_real = H_ls_real / self.H_ls_std
            H_true_real = H_true_real / self.H_true_std
        
        # Different input formats for different models
        if self.model_type == 'lstm':
            # LSTM: (sequence_length, features)
            # Reshape to (num_subcarriers, features_per_subcarrier)
            # Features: [rx_real, rx_imag, H_ls_real, H_ls_imag] for each symbol
            inputs = np.concatenate([
                rx_real.reshape(2, -1).T,    # (14*599, 2) 
                H_ls_real.reshape(2, -1).T,  # (14*599, 2)
            ], axis=1).astype(np.float32)  # (8386, 4)
            
            targets = H_true_real.reshape(2, -1).T.astype(np.float32)  # (8386, 2)
            
        else:  # CNN, Hybrid, ResNet
            inputs = np.concatenate([
                rx_real,
                H_ls_real,
                pilot_mask[np.newaxis, :, :]
            ], axis=0).astype(np.float32)  # (5, 14, 599)
            
            targets = H_true_real.astype(np.float32)  # (2, 14, 599)
        
        pilot_mask_tensor = pilot_mask.astype(np.float32)
        
        return (
            torch.from_numpy(inputs),
            torch.from_numpy(targets),
            torch.from_numpy(pilot_mask_tensor)
        )


class AdvancedTrainer:
    """Training manager for advanced models."""
    
    def __init__(self, model, train_loader, val_loader, config, device):
        """Initialize trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        self.criterion = ChannelEstimationLoss(loss_type='mse')
        
        # Different optimizers for different models
        model_type = config.get('model_type', 'cnn')
        lr = config.get('learning_rate', 0.001)
        
        if model_type == 'lstm':
            self.optimizer = optim.Adam(
                model.parameters(), lr=lr * 0.5,  # Lower LR for LSTM
                weight_decay=1e-5
            )
        elif model_type == 'hybrid':
            self.optimizer = optim.AdamW(
                model.parameters(), lr=lr,
                weight_decay=1e-4
            )
        else:
            self.optimizer = optim.Adam(
                model.parameters(), lr=lr,
                weight_decay=1e-5
            )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        for inputs, targets, pilot_mask in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            pilot_mask = pilot_mask.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Handle different output shapes
            if outputs.shape != targets.shape:
                # Reshape if needed (for LSTM)
                outputs = outputs.view(targets.shape)
            
            loss = self.criterion(outputs, targets, pilot_mask)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for inputs, targets, pilot_mask in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                pilot_mask = pilot_mask.to(self.device)
                
                outputs = self.model(inputs)
                
                if outputs.shape != targets.shape:
                    outputs = outputs.view(targets.shape)
                
                loss = self.criterion(outputs, targets, pilot_mask)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, epochs, save_dir='models', patience=10):
        """Full training loop with early stopping."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        model_type = self.config.get('model_type', 'model')
        
        print(f"\nStarting {model_type.upper()} training for {epochs} epochs...")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Early stopping patience: {patience}")
        print()
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"LR: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                self.patience_counter = 0
                
                model_path = save_path / f'{model_type}_best.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, model_path)
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch+1}")
                    break
        
        # Save training history
        history_path = save_path / f'{model_type}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"\nTraining complete!")
        print(f"  Best epoch: {self.best_epoch}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  NMSE: {linear2db(self.best_val_loss):.2f} dB")
        
        return self.history


def train_model(model_type: str, epochs: int, batch_size: int = 32,
                data_dir: str = 'data', save_dir: str = 'models',
                device: str = None, patience: int = 10):
    """Train a specific model type."""
    
    # Setup device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
    # Load datasets
    train_dataset = ChannelEstimationDataset(
        f'{data_dir}/train.npz', model_type=model_type
    )
    val_dataset = ChannelEstimationDataset(
        f'{data_dir}/val.npz', model_type=model_type
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    # Get model
    config = {
        'model_type': model_type,
        'input_channels': 5,
        'learning_rate': 0.001,
        'epochs': epochs
    }
    
    # Adjust config for different models
    if model_type == 'lstm':
        config['input_size'] = 4  # 4 features per timestep
        config['hidden_size'] = 256
        config['num_layers'] = 2
    elif model_type == 'hybrid':
        config['cnn_channels'] = [32, 64, 128]
        config['lstm_hidden'] = 128
    elif model_type == 'resnet':
        config['num_blocks'] = 4
        config['hidden_channels'] = 64
    
    model = get_model(model_type, config)
    
    # Create trainer and train
    trainer = AdvancedTrainer(
        model, train_loader, val_loader, config, device
    )
    
    history = trainer.train(epochs, save_dir, patience)
    
    return history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Phase 6-7: Advanced Model Training'
    )
    
    parser.add_argument('--model', type=str, default='lstm',
                       choices=['lstm', 'hybrid', 'resnet'],
                       help='Model architecture to train')
    parser.add_argument('--all', action='store_true',
                       help='Train all advanced models')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Model save directory')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("   PHASE 6-7: ADVANCED MODEL TRAINING")
    print("="*60)
    
    set_seed(42)
    
    models_to_train = ['lstm', 'hybrid', 'resnet'] if args.all else [args.model]
    
    for model_type in models_to_train:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} model")
        print(f"{'='*60}")
        
        try:
            train_model(
                model_type=model_type,
                epochs=args.epochs,
                batch_size=args.batch_size,
                data_dir=args.data_dir,
                save_dir=args.save_dir,
                device=args.device,
                patience=args.patience
            )
        except Exception as e:
            print(f"Error training {model_type}: {e}")
            continue
    
    print("\n" + "="*60)
    print("PHASE 6-7 COMPLETE: Advanced Model Training")
    print("="*60)
    print(f"\nNext step: Run Phase 5 evaluation to compare all models")
    print(f"  python run_phase5_evaluation.py --all --snr-sweep")


if __name__ == '__main__':
    main()
