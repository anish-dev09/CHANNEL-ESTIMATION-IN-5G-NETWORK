"""
Phase 3: Dataset Generation Runner Script
==========================================

This script generates training, validation, and test datasets for 
AI-assisted channel estimation in 5G MIMO-OFDM systems.

Usage:
    python run_phase3_dataset_generation.py --samples 1000   # Quick test
    python run_phase3_dataset_generation.py --samples 10000  # Full dataset
    python run_phase3_dataset_generation.py --split train --samples 5000  # Single split
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from channel_simulator import simulate_transmission
from baseline_estimators import LSEstimator
from utils import load_config, set_seed


class DatasetGenerator:
    """Simplified dataset generator for Phase 3."""
    
    def __init__(self, config_path: str = 'configs/experiment_config.yaml'):
        """Initialize generator with configuration."""
        self.config = load_config(config_path)
        print(f"✓ Configuration loaded from {config_path}")
        
    def generate_sample(self, channel_type: str, doppler_hz: float,
                       snr_db: float, pilot_density: float) -> dict:
        """Generate a single training sample."""
        
        # Simulate MIMO-OFDM transmission
        sim_data = simulate_transmission(
            self.config,
            channel_type=channel_type,
            doppler_hz=doppler_hz,
            snr_db=snr_db,
            pilot_density=pilot_density
        )
        
        # Extract components
        rx_symbols = sim_data['rx_symbols']  # (14, 2, 599)
        tx_symbols = sim_data['tx_symbols']  # (14, 2, 599)
        H_true = sim_data['channel']  # (14, 2, 2, 599)
        pilot_pattern = sim_data['pilot_pattern']
        pilot_symbols = sim_data['pilot_symbols']
        
        # Compute LS estimate as input feature
        num_symbols, num_rx, num_sc = rx_symbols.shape
        num_tx = tx_symbols.shape[1]
        
        # Create 4D array for LS estimator
        rx_4d = rx_symbols.reshape(num_symbols, num_rx, 1, num_sc)
        rx_4d = np.repeat(rx_4d, num_tx, axis=2)
        
        ls_estimator = LSEstimator(interpolation_method='linear')
        H_ls = ls_estimator.estimate(
            rx_4d, pilot_symbols,
            pilot_pattern.pilot_mask,
            pilot_pattern.pilot_positions
        )
        
        return {
            'rx_symbols': rx_symbols,
            'tx_symbols': tx_symbols,
            'H_ls': H_ls,
            'H_true': H_true,
            'pilot_mask': pilot_pattern.pilot_mask.astype(np.float32),
            'snr_db': snr_db,
            'channel_type': channel_type,
            'doppler_hz': doppler_hz,
            'pilot_density': pilot_density
        }
    
    def generate_dataset(self, num_samples: int, split: str = 'train',
                        seed: int = None) -> dict:
        """
        Generate complete dataset with memory-efficient processing.
        
        Args:
            num_samples: Number of samples to generate
            split: Dataset split name ('train', 'val', 'test')
            seed: Random seed (different for each split)
            
        Returns:
            Dictionary with stacked arrays
        """
        # Set seed based on split for reproducibility
        if seed is None:
            seed_map = {'train': 42, 'val': 123, 'test': 456}
            seed = seed_map.get(split, 42)
        set_seed(seed)
        
        # Parameter options
        channel_types = ['EPA', 'EVA', 'ETU']
        doppler_values = [10, 50, 100, 200]
        snr_values = [-5, 0, 5, 10, 15, 20, 25, 30]
        pilot_densities = [0.05, 0.10]  # 5% and 10%
        
        print(f"\n{'='*60}")
        print(f"Generating {split.upper()} dataset: {num_samples} samples")
        print(f"{'='*60}")
        print(f"  Channel types: {channel_types}")
        print(f"  Doppler (Hz): {doppler_values}")
        print(f"  SNR (dB): {snr_values}")
        print(f"  Pilot density: {[f'{d*100:.0f}%' for d in pilot_densities]}")
        print(f"  Random seed: {seed}")
        print(f"  Memory-efficient mode: complex64")
        print()
        
        # Pre-allocate arrays for memory efficiency (use complex64 to save memory)
        # Get sample shapes from first sample
        test_sample = self.generate_sample('EPA', 50.0, 10.0, 0.1)
        
        rx_shape = (num_samples,) + test_sample['rx_symbols'].shape
        tx_shape = (num_samples,) + test_sample['tx_symbols'].shape
        H_ls_shape = (num_samples,) + test_sample['H_ls'].shape
        H_true_shape = (num_samples,) + test_sample['H_true'].shape
        pilot_shape = (num_samples,) + test_sample['pilot_mask'].shape
        
        print(f"  Pre-allocating arrays...")
        print(f"    rx_symbols: {rx_shape}")
        print(f"    H_true: {H_true_shape}")
        
        # Use complex64 instead of complex128 to halve memory usage
        rx_symbols = np.zeros(rx_shape, dtype=np.complex64)
        tx_symbols = np.zeros(tx_shape, dtype=np.complex64)
        H_ls = np.zeros(H_ls_shape, dtype=np.complex64)
        H_true = np.zeros(H_true_shape, dtype=np.complex64)
        pilot_mask = np.zeros(pilot_shape, dtype=np.float32)
        snr_db = np.zeros(num_samples, dtype=np.float32)
        channel_type = np.empty(num_samples, dtype='U3')
        doppler_hz = np.zeros(num_samples, dtype=np.float32)
        pilot_density_arr = np.zeros(num_samples, dtype=np.float32)
        
        failed_count = 0
        valid_idx = 0
        
        # Generate samples directly into pre-allocated arrays
        for i in tqdm(range(num_samples), desc=f"Generating {split}"):
            # Randomly select parameters
            ch_type = np.random.choice(channel_types)
            doppler = float(np.random.choice(doppler_values))
            snr = float(np.random.choice(snr_values))
            pilot_dens = float(np.random.choice(pilot_densities))
            
            try:
                sample = self.generate_sample(ch_type, doppler, snr, pilot_dens)
                
                # Store directly in pre-allocated arrays (convert to complex64)
                rx_symbols[valid_idx] = sample['rx_symbols'].astype(np.complex64)
                tx_symbols[valid_idx] = sample['tx_symbols'].astype(np.complex64)
                H_ls[valid_idx] = sample['H_ls'].astype(np.complex64)
                H_true[valid_idx] = sample['H_true'].astype(np.complex64)
                pilot_mask[valid_idx] = sample['pilot_mask']
                snr_db[valid_idx] = snr
                channel_type[valid_idx] = ch_type
                doppler_hz[valid_idx] = doppler
                pilot_density_arr[valid_idx] = pilot_dens
                
                valid_idx += 1
                
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:
                    print(f"\n  Warning: Sample {i} failed: {e}")
                continue
        
        if failed_count > 0:
            print(f"\n  Total failed samples: {failed_count}/{num_samples}")
            # Trim arrays to valid samples
            rx_symbols = rx_symbols[:valid_idx]
            tx_symbols = tx_symbols[:valid_idx]
            H_ls = H_ls[:valid_idx]
            H_true = H_true[:valid_idx]
            pilot_mask = pilot_mask[:valid_idx]
            snr_db = snr_db[:valid_idx]
            channel_type = channel_type[:valid_idx]
            doppler_hz = doppler_hz[:valid_idx]
            pilot_density_arr = pilot_density_arr[:valid_idx]
        
        print(f"\n  Generated {valid_idx} samples successfully")
        
        dataset = {
            'rx_symbols': rx_symbols,
            'tx_symbols': tx_symbols,
            'H_ls': H_ls,
            'H_true': H_true,
            'pilot_mask': pilot_mask,
            'snr_db': snr_db,
            'channel_type': channel_type,
            'doppler_hz': doppler_hz,
            'pilot_density': pilot_density_arr,
        }
        
        return dataset
    
    def save_dataset(self, dataset: dict, filepath: str):
        """Save dataset to NPZ file."""
        np.savez_compressed(filepath, **dataset)
        
        # Get file size
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"✓ Saved to {filepath} ({file_size:.1f} MB)")
    
    def print_dataset_info(self, dataset: dict, name: str):
        """Print dataset information."""
        print(f"\n{name} Dataset Info:")
        print(f"  Samples: {dataset['rx_symbols'].shape[0]}")
        print(f"  RX symbols shape: {dataset['rx_symbols'].shape}")
        print(f"  TX symbols shape: {dataset['tx_symbols'].shape}")
        print(f"  H_ls shape: {dataset['H_ls'].shape}")
        print(f"  H_true shape: {dataset['H_true'].shape}")
        print(f"  Pilot mask shape: {dataset['pilot_mask'].shape}")
        print(f"  SNR range: [{dataset['snr_db'].min():.0f}, {dataset['snr_db'].max():.0f}] dB")
        print(f"  Channel types: {np.unique(dataset['channel_type'])}")
        print(f"  Doppler range: [{dataset['doppler_hz'].min():.0f}, {dataset['doppler_hz'].max():.0f}] Hz")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Phase 3: Generate Channel Estimation Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Quick test (100 samples each):
    python run_phase3_dataset_generation.py --samples 100
    
  Medium dataset (1000 samples each):
    python run_phase3_dataset_generation.py --samples 1000
    
  Full dataset (10000 train, 1000 val, 2000 test):
    python run_phase3_dataset_generation.py --train 10000 --val 1000 --test 2000
    
  Single split:
    python run_phase3_dataset_generation.py --split train --samples 5000
        """
    )
    
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples per split (default: 1000)')
    parser.add_argument('--train', type=int, default=None,
                       help='Training samples (overrides --samples)')
    parser.add_argument('--val', type=int, default=None,
                       help='Validation samples (overrides --samples)')
    parser.add_argument('--test', type=int, default=None,
                       help='Test samples (overrides --samples)')
    parser.add_argument('--split', type=str, default=None,
                       choices=['train', 'val', 'test'],
                       help='Generate only this split')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory (default: data)')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Determine sample counts
    train_samples = args.train if args.train else args.samples
    val_samples = args.val if args.val else max(args.samples // 10, 100)
    test_samples = args.test if args.test else max(args.samples // 5, 200)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("   PHASE 3: DATASET GENERATION")
    print("   AI-Assisted Channel Estimation in 5G")
    print("="*60)
    
    # Initialize generator
    generator = DatasetGenerator(args.config)
    
    # Determine which splits to generate
    if args.split:
        splits = [args.split]
        sample_counts = {args.split: args.samples}
    else:
        splits = ['train', 'val', 'test']
        sample_counts = {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
    
    print(f"\nDatasets to generate:")
    for split in splits:
        print(f"  {split}: {sample_counts[split]} samples")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Generate each split
    generated_datasets = {}
    
    for split in splits:
        dataset = generator.generate_dataset(
            num_samples=sample_counts[split],
            split=split
        )
        
        # Save dataset
        filepath = output_dir / f'{split}.npz'
        generator.save_dataset(dataset, str(filepath))
        
        # Print info
        generator.print_dataset_info(dataset, split.capitalize())
        
        generated_datasets[split] = dataset
    
    # Final summary
    print("\n" + "="*60)
    print("PHASE 3 COMPLETE: Dataset Generation")
    print("="*60)
    print(f"\nGenerated files:")
    for split in splits:
        filepath = output_dir / f'{split}.npz'
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filepath} ({size_mb:.1f} MB)")
    
    print(f"\nNext step: Run Phase 4 - CNN Model Training")
    print(f"  python run_phase4_training.py --model cnn --epochs 50")
    print("="*60 + "\n")
    
    return generated_datasets


if __name__ == '__main__':
    main()
