"""
Phase 4: CNN Model Training Script
===================================

Trains CNN-based channel estimator and compares with baselines.

Usage:
    python run_phase4_training.py --model cnn --epochs 50
    python run_phase4_training.py --model cnn --epochs 10 --quick  # Quick test
    python run_phase4_training.py --model resnet --epochs 100
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
    
    def __init__(self, data_path: str, normalize: bool = True):
        """
        Load dataset from NPZ file.
        
        Args:
            data_path: Path to .npz file
            normalize: Whether to normalize inputs
        """
        print(f"Loading dataset from {data_path}...")
        data = np.load(data_path, allow_pickle=True)
        
        self.rx_symbols = data['rx_symbols']
        self.H_ls = data['H_ls']
        self.H_true = data['H_true']
        self.pilot_mask = data['pilot_mask']
        self.snr_db = data['snr_db']
        
        self.num_samples = self.rx_symbols.shape[0]
        self.normalize = normalize
        
        # Compute normalization statistics
        if normalize:
            self._compute_stats()
        
        print(f"  Loaded {self.num_samples} samples")
    
    def _compute_stats(self):
        """Compute normalization statistics."""
        # Use first antenna pair for stats
        rx = self.rx_symbols[:, :, 0, :]  # (N, 14, 599)
        H_ls = self.H_ls[:, :, 0, 0, :]  # (N, 14, 599)
        
        # Compute std for normalization
        self.rx_std = np.std(np.abs(rx)) + 1e-8
        self.H_ls_std = np.std(np.abs(H_ls)) + 1e-8
        self.H_true_std = np.std(np.abs(self.H_true[:, :, 0, 0, :])) + 1e-8
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a single sample."""
        # Take first RX and TX antenna (2x2 MIMO -> use first pair)
        rx = self.rx_symbols[idx, :, 0, :]  # (14, 599)
        H_ls = self.H_ls[idx, :, 0, 0, :]  # (14, 599)
        H_true = self.H_true[idx, :, 0, 0, :]  # (14, 599)
        pilot_mask = self.pilot_mask[idx]  # (14, 599)
        
        # Convert complex to real: [real, imag]
        rx_real = np.stack([rx.real, rx.imag], axis=0)  # (2, 14, 599)
        H_ls_real = np.stack([H_ls.real, H_ls.imag], axis=0)  # (2, 14, 599)
        H_true_real = np.stack([H_true.real, H_true.imag], axis=0)  # (2, 14, 599)
        
        # Normalize
        if self.normalize:
            rx_real = rx_real / self.rx_std
            H_ls_real = H_ls_real / self.H_ls_std
            H_true_real = H_true_real / self.H_true_std
        
        # Combine inputs: [rx_real, rx_imag, H_ls_real, H_ls_imag, pilot_mask]
        inputs = np.concatenate([
            rx_real,              # (2, 14, 599)
            H_ls_real,            # (2, 14, 599)
            pilot_mask[np.newaxis, :, :]  # (1, 14, 599)
        ], axis=0).astype(np.float32)  # (5, 14, 599)
        
        # Target: true channel [real, imag]
        targets = H_true_real.astype(np.float32)  # (2, 14, 599)
        
        # Pilot mask for loss weighting
        pilot_mask_tensor = pilot_mask.astype(np.float32)  # (14, 599)
        
        return (
            torch.from_numpy(inputs),
            torch.from_numpy(targets),
            torch.from_numpy(pilot_mask_tensor)
        )


class Trainer:
    """Training manager for channel estimation models."""
    
    def __init__(self, model, train_loader, val_loader, config, device):
        """Initialize trainer."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = ChannelEstimationLoss(loss_type='mse')
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 50)
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
    
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
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets, pilot_mask)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
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
                loss = self.criterion(outputs, targets, pilot_mask)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, epochs, save_dir='models'):
        """Full training loop."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nStarting training for {epochs} epochs...")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {current_lr:.6f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch + 1
                
                model_path = save_path / f'{self.config["model_type"]}_best.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, model_path)
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save final model
        final_path = save_path / f'{self.config["model_type"]}_final.pth'
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }, final_path)
        
        # Save history
        history_path = save_path / f'{self.config["model_type"]}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history


def evaluate_model(model, test_loader, device):
    """Evaluate trained model on test set."""
    model.eval()
    
    all_mse = []
    all_nmse = []
    inference_times = []
    
    with torch.no_grad():
        for inputs, targets, pilot_mask in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Time inference
            start_time = time.time()
            outputs = model(inputs)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            inference_time = time.time() - start_time
            inference_times.append(inference_time / inputs.shape[0])
            
            # Compute MSE per sample
            for i in range(outputs.shape[0]):
                pred = outputs[i].cpu().numpy()
                true = targets[i].cpu().numpy()
                
                mse = np.mean((pred - true) ** 2)
                nmse = mse / (np.mean(true ** 2) + 1e-8)
                
                all_mse.append(mse)
                all_nmse.append(linear2db(nmse))
    
    results = {
        'mse_mean': float(np.mean(all_mse)),
        'mse_std': float(np.std(all_mse)),
        'nmse_mean_db': float(np.mean(all_nmse)),
        'nmse_std_db': float(np.std(all_nmse)),
        'latency_mean_ms': float(np.mean(inference_times) * 1000),
        'latency_std_ms': float(np.std(inference_times) * 1000)
    }
    
    return results


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Phase 4: Train Channel Estimation Model')
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'lstm', 'hybrid', 'resnet'],
                       help='Model type (default: cnn)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--save-dir', type=str, default='models',
                       help='Model save directory (default: models)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test mode (fewer epochs, smaller batch)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cpu, or cuda')
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.epochs = min(args.epochs, 5)
        args.batch_size = 8
        print("Quick mode enabled: 5 epochs, batch size 8")
    
    # Set seed
    set_seed(42)
    
    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print("\n" + "="*60)
    print("   PHASE 4: AI MODEL TRAINING")
    print("   AI-Assisted Channel Estimation in 5G")
    print("="*60)
    print(f"\n  Model: {args.model.upper()}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    
    # Load datasets
    data_dir = Path(args.data_dir)
    
    train_dataset = ChannelEstimationDataset(data_dir / 'train.npz')
    val_dataset = ChannelEstimationDataset(data_dir / 'val.npz')
    test_dataset = ChannelEstimationDataset(data_dir / 'test.npz')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    
    print(f"\n  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Create model
    model_config = {
        'input_channels': 5,
        'hidden_channels': [64, 128, 256, 128, 64],
        'kernel_size': 3,
        'dropout': 0.1
    }
    
    model = get_model(args.model, model_config)
    print(f"\n  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training config
    train_config = {
        'model_type': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': 1e-5
    }
    
    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, train_config, device)
    
    # Train
    print("\n" + "-"*60)
    history = trainer.train(args.epochs, args.save_dir)
    print("-"*60)
    
    # Training summary
    print(f"\nTraining completed!")
    print(f"  Best validation loss: {trainer.best_val_loss:.4f} (epoch {trainer.best_epoch})")
    print(f"  Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final validation loss: {history['val_loss'][-1]:.4f}")
    
    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    best_model_path = Path(args.save_dir) / f'{args.model}_best.pth'
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    results = evaluate_model(model, test_loader, device)
    
    print("\n" + "="*60)
    print("PHASE 4 COMPLETE: Model Training")
    print("="*60)
    print(f"\nTest Set Results:")
    print(f"  NMSE: {results['nmse_mean_db']:.2f} ± {results['nmse_std_db']:.2f} dB")
    print(f"  MSE: {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
    print(f"  Inference latency: {results['latency_mean_ms']:.2f} ± {results['latency_std_ms']:.2f} ms")
    
    # Save results
    results_path = Path(args.save_dir) / f'{args.model}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results saved to {results_path}")
    
    print(f"\nSaved files:")
    print(f"  ✓ {args.save_dir}/{args.model}_best.pth (best model)")
    print(f"  ✓ {args.save_dir}/{args.model}_final.pth (final model)")
    print(f"  ✓ {args.save_dir}/{args.model}_history.json (training history)")
    print(f"  ✓ {args.save_dir}/{args.model}_results.json (test results)")
    
    print(f"\nNext step: Run Phase 5 - Advanced Models")
    print(f"  python run_phase4_training.py --model lstm --epochs 50")
    print(f"  python run_phase4_training.py --model hybrid --epochs 50")
    print("="*60 + "\n")
    
    return results


if __name__ == '__main__':
    main()
