"""
Dataset generation for AI-assisted channel estimation.

Generates training, validation, and test datasets with various:
- SNR conditions
- Channel models (EPA, EVA, ETU)
- Doppler effects
- Pilot densities
"""

import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse

from channel_simulator import simulate_transmission
from baseline_estimators import LSEstimator
from utils import load_config, create_directories, set_seed, complex_to_real


class ChannelEstimationDataset:
    """Dataset generator for channel estimation."""
    
    def __init__(self, config: Dict):
        """Initialize dataset generator."""
        self.config = config
        self.ofdm_config = config['ofdm']
        self.mimo_config = config['mimo']
        self.channel_config = config['channel']
        self.dataset_config = config['dataset']
        
    def generate_sample(self, channel_type: str, doppler_hz: float,
                       snr_db: float, pilot_density: float) -> Dict:
        """
        Generate a single training sample.
        
        Returns:
            Dictionary with inputs, targets, and metadata
        """
        # Simulate transmission
        sim_data = simulate_transmission(
            self.config,
            channel_type=channel_type,
            doppler_hz=doppler_hz,
            snr_db=snr_db,
            pilot_density=pilot_density
        )
        
        # Extract key components
        rx_symbols = sim_data['rx_symbols']  # (num_symbols, num_rx, num_subcarriers)
        tx_symbols = sim_data['tx_symbols']  # (num_symbols, num_tx, num_subcarriers)
        H_true = sim_data['channel']  # (num_symbols, num_rx, num_tx, num_subcarriers)
        pilot_pattern = sim_data['pilot_pattern']
        pilot_symbols = sim_data['pilot_symbols']
        
        # Get LS estimate as additional input feature
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
        
        # Prepare inputs and targets
        # Input: received signal + LS estimate + pilot mask
        # Target: true channel
        
        sample = {
            'rx_symbols': rx_symbols,  # (num_symbols, num_rx, num_subcarriers)
            'tx_symbols': tx_symbols,  # (num_symbols, num_tx, num_subcarriers)
            'H_ls': H_ls,  # (num_symbols, num_rx, num_tx, num_subcarriers)
            'H_true': H_true,  # (num_symbols, num_rx, num_tx, num_subcarriers)
            'pilot_mask': pilot_pattern.pilot_mask,  # (num_symbols, num_subcarriers)
            'snr_db': snr_db,
            'channel_type': channel_type,
            'doppler_hz': doppler_hz,
            'pilot_density': pilot_density
        }
        
        return sample
    
    def generate_dataset(self, num_samples: int, split: str = 'train') -> List[Dict]:
        """
        Generate dataset with diverse conditions.
        
        Args:
            num_samples: Number of samples to generate
            split: 'train', 'val', or 'test'
            
        Returns:
            List of samples
        """
        dataset = []
        
        # Define parameter ranges
        channel_types = self.channel_config['models']
        doppler_values = self.channel_config['doppler_hz']
        snr_values = self.config['simulation']['snr_range']
        pilot_densities = self.config['pilots']['density']
        
        print(f"Generating {split} dataset with {num_samples} samples...")
        
        for i in tqdm(range(num_samples)):
            # Randomly select parameters
            channel_type = np.random.choice(channel_types)
            doppler = np.random.choice(doppler_values)
            snr = np.random.choice(snr_values)
            pilot_density = np.random.choice(pilot_densities)
            
            # Generate sample
            try:
                sample = self.generate_sample(channel_type, doppler, snr, pilot_density)
                dataset.append(sample)
            except Exception as e:
                print(f"Warning: Failed to generate sample {i}: {e}")
                continue
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filepath: str, format: str = 'npz'):
        """
        Save dataset to file.
        
        Args:
            dataset: List of samples
            filepath: Output file path
            format: 'npz' or 'h5'
        """
        if format == 'npz':
            self._save_npz(dataset, filepath)
        elif format == 'h5':
            self._save_h5(dataset, filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _save_npz(self, dataset: List[Dict], filepath: str):
        """Save dataset as NPZ file."""
        # Stack all samples
        stacked = {
            'rx_symbols': np.stack([s['rx_symbols'] for s in dataset]),
            'tx_symbols': np.stack([s['tx_symbols'] for s in dataset]),
            'H_ls': np.stack([s['H_ls'] for s in dataset]),
            'H_true': np.stack([s['H_true'] for s in dataset]),
            'pilot_mask': np.stack([s['pilot_mask'] for s in dataset]),
            'snr_db': np.array([s['snr_db'] for s in dataset]),
            'channel_type': np.array([s['channel_type'] for s in dataset]),
            'doppler_hz': np.array([s['doppler_hz'] for s in dataset]),
            'pilot_density': np.array([s['pilot_density'] for s in dataset]),
        }
        
        np.savez_compressed(filepath, **stacked)
        print(f"Dataset saved to {filepath}")
    
    def _save_h5(self, dataset: List[Dict], filepath: str):
        """Save dataset as HDF5 file."""
        with h5py.File(filepath, 'w') as f:
            # Create datasets
            f.create_dataset('rx_symbols', data=np.stack([s['rx_symbols'] for s in dataset]))
            f.create_dataset('tx_symbols', data=np.stack([s['tx_symbols'] for s in dataset]))
            f.create_dataset('H_ls', data=np.stack([s['H_ls'] for s in dataset]))
            f.create_dataset('H_true', data=np.stack([s['H_true'] for s in dataset]))
            f.create_dataset('pilot_mask', data=np.stack([s['pilot_mask'] for s in dataset]))
            f.create_dataset('snr_db', data=np.array([s['snr_db'] for s in dataset]))
            f.create_dataset('doppler_hz', data=np.array([s['doppler_hz'] for s in dataset]))
            f.create_dataset('pilot_density', data=np.array([s['pilot_density'] for s in dataset]))
            
            # Store channel types as strings
            channel_types = np.array([s['channel_type'] for s in dataset], dtype='S10')
            f.create_dataset('channel_type', data=channel_types)
        
        print(f"Dataset saved to {filepath}")


def prepare_ml_inputs(sample: Dict, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare inputs and targets for ML model.
    
    Args:
        sample: Dataset sample
        normalize: Whether to normalize inputs
        
    Returns:
        (inputs, targets) as real-valued arrays
    """
    # Input: concatenate RX symbols, LS estimate, and pilot mask
    # Shape: (num_symbols, num_subcarriers, features)
    
    num_symbols, num_rx, num_sc = sample['rx_symbols'].shape
    _, _, num_tx, _ = sample['H_ls'].shape
    
    # For simplicity, take first RX and TX antenna (can extend to all)
    rx = sample['rx_symbols'][:, 0, :]  # (num_symbols, num_subcarriers)
    H_ls = sample['H_ls'][:, 0, 0, :]  # (num_symbols, num_subcarriers)
    pilot_mask = sample['pilot_mask'].astype(float)  # (num_symbols, num_subcarriers)
    
    # Convert complex to real (separate real/imag channels)
    rx_real = complex_to_real(rx)  # (num_symbols, num_subcarriers, 2)
    H_ls_real = complex_to_real(H_ls)  # (num_symbols, num_subcarriers, 2)
    
    # Concatenate features
    inputs = np.concatenate([
        rx_real,
        H_ls_real,
        pilot_mask[..., np.newaxis]
    ], axis=-1)  # (num_symbols, num_subcarriers, 5)
    
    # Target: true channel
    H_true = sample['H_true'][:, 0, 0, :]  # (num_symbols, num_subcarriers)
    targets = complex_to_real(H_true)  # (num_symbols, num_subcarriers, 2)
    
    # Normalize if requested
    if normalize:
        # Normalize inputs (except pilot mask)
        inputs[..., :4] = inputs[..., :4] / (np.std(inputs[..., :4]) + 1e-8)
        # Normalize targets
        targets = targets / (np.std(targets) + 1e-8)
    
    return inputs, targets


def main():
    """Main dataset generation script."""
    parser = argparse.ArgumentParser(description='Generate channel estimation dataset')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory')
    parser.add_argument('--train-samples', type=int, default=None,
                       help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=None,
                       help='Number of validation samples')
    parser.add_argument('--test-samples', type=int, default=None,
                       help='Number of test samples')
    parser.add_argument('--format', type=str, default='npz', choices=['npz', 'h5'],
                       help='Output format')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Override sample counts if provided
    if args.train_samples is not None:
        config['dataset']['train_samples'] = args.train_samples
    if args.val_samples is not None:
        config['dataset']['val_samples'] = args.val_samples
    if args.test_samples is not None:
        config['dataset']['test_samples'] = args.test_samples
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    generator = ChannelEstimationDataset(config)
    
    # Generate datasets
    print("\n" + "="*60)
    print("Channel Estimation Dataset Generation")
    print("="*60)
    
    # Training set
    train_dataset = generator.generate_dataset(
        config['dataset']['train_samples'], 
        split='train'
    )
    generator.save_dataset(
        train_dataset, 
        output_dir / f'train.{args.format}',
        format=args.format
    )
    
    # Validation set
    val_dataset = generator.generate_dataset(
        config['dataset']['val_samples'],
        split='val'
    )
    generator.save_dataset(
        val_dataset,
        output_dir / f'val.{args.format}',
        format=args.format
    )
    
    # Test set
    test_dataset = generator.generate_dataset(
        config['dataset']['test_samples'],
        split='test'
    )
    generator.save_dataset(
        test_dataset,
        output_dir / f'test.{args.format}',
        format=args.format
    )
    
    print("\n" + "="*60)
    print("Dataset generation completed!")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
