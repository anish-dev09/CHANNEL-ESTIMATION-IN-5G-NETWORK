"""
Phase 9: Hyperparameter Tuning
===============================

Systematic hyperparameter optimization for best model performance.

Usage:
    python run_phase9_hyperparameter_tuning.py --model cnn --trials 20
    python run_phase9_hyperparameter_tuning.py --model cnn --quick  # Quick test
"""

import sys
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import itertools

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from ai_models import CNNChannelEstimator, ChannelEstimationLoss
from utils import load_config, set_seed, linear2db


class QuickDataset(Dataset):
    """Lightweight dataset for hyperparameter tuning."""
    
    def __init__(self, data_path: str, max_samples: int = 2000):
        """Load subset of data."""
        data = np.load(data_path, allow_pickle=True)
        
        n = min(max_samples, len(data['rx_symbols']))
        self.rx_symbols = data['rx_symbols'][:n]
        self.H_ls = data['H_ls'][:n]
        self.H_true = data['H_true'][:n]
        self.pilot_mask = data['pilot_mask'][:n]
        
        # Normalization
        self.rx_std = np.std(np.abs(self.rx_symbols)) + 1e-8
        self.H_ls_std = np.std(np.abs(self.H_ls)) + 1e-8
        self.H_true_std = np.std(np.abs(self.H_true)) + 1e-8
    
    def __len__(self):
        return len(self.rx_symbols)
    
    def __getitem__(self, idx):
        rx = self.rx_symbols[idx, :, 0, :]
        H_ls = self.H_ls[idx, :, 0, 0, :]
        H_true = self.H_true[idx, :, 0, 0, :]
        pilot_mask = self.pilot_mask[idx]
        
        rx_real = np.stack([rx.real, rx.imag], axis=0) / self.rx_std
        H_ls_real = np.stack([H_ls.real, H_ls.imag], axis=0) / self.H_ls_std
        H_true_real = np.stack([H_true.real, H_true.imag], axis=0) / self.H_true_std
        
        inputs = np.concatenate([
            rx_real, H_ls_real, pilot_mask[np.newaxis, :, :]
        ], axis=0).astype(np.float32)
        
        return (
            torch.from_numpy(inputs),
            torch.from_numpy(H_true_real.astype(np.float32)),
            torch.from_numpy(pilot_mask.astype(np.float32))
        )


class HyperparameterTuner:
    """Hyperparameter tuning using grid/random search."""
    
    def __init__(self, data_dir: str = 'data', results_dir: str = 'results'):
        """Initialize tuner."""
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir) / 'hyperparameter_tuning'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Load datasets
        self.train_dataset = QuickDataset(str(self.data_dir / 'train.npz'), max_samples=2000)
        self.val_dataset = QuickDataset(str(self.data_dir / 'val.npz'), max_samples=500)
        
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")
    
    def create_model(self, config: dict) -> nn.Module:
        """Create model with specific hyperparameters."""
        model = CNNChannelEstimator(
            input_channels=config.get('input_channels', 5),
            hidden_channels=config.get('hidden_channels', [64, 128, 256, 128, 64]),
            kernel_size=config.get('kernel_size', 3),
            dropout=config.get('dropout', 0.1)
        )
        return model.to(self.device)
    
    def train_and_evaluate(self, config: dict, epochs: int = 15) -> dict:
        """Train model with config and return validation loss."""
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=config.get('batch_size', 32),
            shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.get('batch_size', 32),
            shuffle=False, num_workers=0
        )
        
        # Create model
        model = self.create_model(config)
        criterion = ChannelEstimationLoss(loss_type='mse')
        
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            model.train()
            for inputs, targets, masks in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                masks = masks.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            scheduler.step()
            
            # Validate
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets, masks in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    masks = masks.to(self.device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, targets, masks)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            best_val_loss = min(best_val_loss, val_loss)
        
        return {
            'val_loss': best_val_loss,
            'nmse_db': linear2db(best_val_loss),
            'config': config
        }
    
    def grid_search(self, search_space: dict, epochs: int = 15) -> list:
        """Perform grid search over hyperparameter space."""
        
        # Generate all combinations
        keys = list(search_space.keys())
        values = list(search_space.values())
        combinations = list(itertools.product(*values))
        
        print(f"\nGrid search: {len(combinations)} configurations")
        
        results = []
        
        for combo in tqdm(combinations, desc="Grid search"):
            config = dict(zip(keys, combo))
            
            try:
                result = self.train_and_evaluate(config, epochs)
                results.append(result)
                
                tqdm.write(f"  Config: lr={config.get('learning_rate')}, "
                          f"dropout={config.get('dropout')}, "
                          f"NMSE={result['nmse_db']:.2f} dB")
                
            except Exception as e:
                tqdm.write(f"  Failed: {e}")
        
        # Sort by validation loss
        results.sort(key=lambda x: x['val_loss'])
        
        return results
    
    def random_search(self, search_space: dict, num_trials: int = 20, 
                     epochs: int = 15) -> list:
        """Perform random search over hyperparameter space."""
        
        print(f"\nRandom search: {num_trials} trials")
        
        results = []
        
        for trial in tqdm(range(num_trials), desc="Random search"):
            # Sample random config
            config = {}
            for key, values in search_space.items():
                if isinstance(values, list):
                    config[key] = np.random.choice(values)
                elif isinstance(values, tuple):  # (min, max) for continuous
                    config[key] = np.random.uniform(values[0], values[1])
            
            try:
                result = self.train_and_evaluate(config, epochs)
                results.append(result)
                
                tqdm.write(f"  Trial {trial+1}: NMSE={result['nmse_db']:.2f} dB")
                
            except Exception as e:
                tqdm.write(f"  Trial {trial+1} failed: {e}")
        
        # Sort by validation loss
        results.sort(key=lambda x: x['val_loss'])
        
        return results
    
    def save_results(self, results: list, filename: str):
        """Save tuning results."""
        # Convert numpy types for JSON serialization
        clean_results = []
        for r in results:
            clean_r = {
                'val_loss': float(r['val_loss']),
                'nmse_db': float(r['nmse_db']),
                'config': {k: (v.tolist() if isinstance(v, np.ndarray) else 
                              float(v) if isinstance(v, (np.floating, np.integer)) else v)
                          for k, v in r['config'].items()}
            }
            clean_results.append(clean_r)
        
        path = self.results_dir / filename
        with open(path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        print(f"✓ Saved results to {path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Phase 9: Hyperparameter Tuning'
    )
    
    parser.add_argument('--model', type=str, default='cnn',
                       choices=['cnn', 'lstm', 'hybrid'],
                       help='Model to tune')
    parser.add_argument('--trials', type=int, default=20,
                       help='Number of random search trials')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Training epochs per trial')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test with fewer trials')
    parser.add_argument('--method', type=str, default='random',
                       choices=['grid', 'random'],
                       help='Search method')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory')
    
    args = parser.parse_args()
    
    if args.quick:
        args.trials = 5
        args.epochs = 5
    
    print("\n" + "="*60)
    print("   PHASE 9: HYPERPARAMETER TUNING")
    print("="*60)
    
    set_seed(42)
    
    tuner = HyperparameterTuner(args.data_dir, args.results_dir)
    
    # Define search space
    if args.model == 'cnn':
        search_space = {
            'learning_rate': [0.0005, 0.001, 0.002],
            'dropout': [0.05, 0.1, 0.2],
            'kernel_size': [3, 5],
            'batch_size': [16, 32, 64],
            'weight_decay': [1e-5, 1e-4],
            'hidden_channels': [
                [32, 64, 128, 64, 32],
                [64, 128, 256, 128, 64],
                [64, 128, 128, 64],
            ]
        }
    else:
        search_space = {
            'learning_rate': [0.0005, 0.001],
            'dropout': [0.1, 0.2],
            'batch_size': [16, 32],
        }
    
    # Run search
    if args.method == 'grid':
        # Use subset for grid search
        small_space = {k: v[:2] if isinstance(v, list) else v 
                       for k, v in search_space.items()}
        results = tuner.grid_search(small_space, args.epochs)
    else:
        results = tuner.random_search(search_space, args.trials, args.epochs)
    
    # Save results
    tuner.save_results(results, f'{args.model}_tuning_results.json')
    
    # Print best results
    print("\n" + "="*60)
    print("TOP 5 CONFIGURATIONS")
    print("="*60)
    
    for i, r in enumerate(results[:5]):
        print(f"\n{i+1}. NMSE: {r['nmse_db']:.2f} dB")
        print(f"   Config: {r['config']}")
    
    print("\n" + "="*60)
    print("PHASE 9 COMPLETE: Hyperparameter Tuning")
    print("="*60)
    print(f"\nBest configuration saved. Use these parameters for final training.")


if __name__ == '__main__':
    main()
