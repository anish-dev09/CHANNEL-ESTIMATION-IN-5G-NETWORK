"""
Phase 3: Robust Dataset Generation with Checkpointing
======================================================

This improved version includes:
- Checkpointing: Saves progress every N samples
- Resume capability: Can continue from last checkpoint
- Memory efficient: Uses complex64 and batch processing
- Progress tracking: Detailed progress information

Usage:
    python run_phase3_robust.py --samples 1000             # Quick test
    python run_phase3_robust.py --train 10000 --checkpoint 500  # Full with checkpoints
    python run_phase3_robust.py --resume                   # Resume from checkpoint
"""

import sys
import os
import argparse
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from channel_simulator import simulate_transmission
from baseline_estimators import LSEstimator
from utils import load_config, set_seed


class RobustDatasetGenerator:
    """Robust dataset generator with checkpointing support."""
    
    def __init__(self, config_path: str = 'configs/experiment_config.yaml',
                 output_dir: str = 'data'):
        """Initialize generator with configuration."""
        self.config = load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
        rx_symbols = sim_data['rx_symbols']
        tx_symbols = sim_data['tx_symbols']
        H_true = sim_data['channel']
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
            'rx_symbols': rx_symbols.astype(np.complex64),
            'tx_symbols': tx_symbols.astype(np.complex64),
            'H_ls': H_ls.astype(np.complex64),
            'H_true': H_true.astype(np.complex64),
            'pilot_mask': pilot_pattern.pilot_mask.astype(np.float32),
            'snr_db': np.float32(snr_db),
            'channel_type': channel_type,
            'doppler_hz': np.float32(doppler_hz),
            'pilot_density': np.float32(pilot_density)
        }
    
    def get_checkpoint_path(self, split: str) -> Path:
        """Get checkpoint file path for a split."""
        return self.checkpoint_dir / f'{split}_checkpoint.json'
    
    def get_temp_data_path(self, split: str, chunk_id: int) -> Path:
        """Get temporary data file path for a chunk."""
        return self.checkpoint_dir / f'{split}_chunk_{chunk_id:04d}.npz'
    
    def save_checkpoint(self, split: str, completed: int, total: int, 
                       chunk_id: int, start_time: float):
        """Save generation progress checkpoint."""
        checkpoint = {
            'split': split,
            'completed': completed,
            'total': total,
            'chunk_id': chunk_id,
            'start_time': start_time,
            'timestamp': datetime.now().isoformat(),
            'samples_per_second': completed / (time.time() - start_time) if completed > 0 else 0
        }
        with open(self.get_checkpoint_path(split), 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_checkpoint(self, split: str) -> dict:
        """Load checkpoint if exists."""
        path = self.get_checkpoint_path(split)
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def generate_dataset_chunked(self, num_samples: int, split: str = 'train',
                                 seed: int = None, chunk_size: int = 500,
                                 resume: bool = False) -> dict:
        """
        Generate dataset with chunked saving and checkpointing.
        
        Args:
            num_samples: Total samples to generate
            split: Dataset split name
            seed: Random seed
            chunk_size: Samples per checkpoint
            resume: Whether to resume from checkpoint
        """
        # Set seed
        if seed is None:
            seed_map = {'train': 42, 'val': 123, 'test': 456}
            seed = seed_map.get(split, 42)
        
        # Check for existing checkpoint
        checkpoint = self.load_checkpoint(split) if resume else None
        start_idx = 0
        chunk_id = 0
        
        if checkpoint and checkpoint['completed'] < checkpoint['total']:
            start_idx = checkpoint['completed']
            chunk_id = checkpoint['chunk_id']
            print(f"\n📥 Resuming from checkpoint: {start_idx}/{num_samples} completed")
            print(f"   Last checkpoint: {checkpoint['timestamp']}")
        
        # Set seed with offset for resuming
        set_seed(seed + start_idx)
        
        # Parameter options
        channel_types = ['EPA', 'EVA', 'ETU']
        doppler_values = [10, 50, 100, 200]
        snr_values = [-5, 0, 5, 10, 15, 20, 25, 30]
        pilot_densities = [0.05, 0.10]
        
        print(f"\n{'='*60}")
        print(f"Generating {split.upper()} dataset: {num_samples} samples")
        print(f"{'='*60}")
        print(f"  Chunk size: {chunk_size} (checkpoint every {chunk_size} samples)")
        print(f"  Random seed: {seed}")
        print(f"  Memory mode: complex64")
        print()
        
        # Pre-allocate chunk buffer
        test_sample = self.generate_sample('EPA', 50.0, 10.0, 0.1)
        shapes = {
            'rx_symbols': test_sample['rx_symbols'].shape,
            'tx_symbols': test_sample['tx_symbols'].shape,
            'H_ls': test_sample['H_ls'].shape,
            'H_true': test_sample['H_true'].shape,
            'pilot_mask': test_sample['pilot_mask'].shape,
        }
        
        start_time = time.time()
        failed_count = 0
        
        # Create progress bar
        pbar = tqdm(total=num_samples, initial=start_idx, desc=f"Generating {split}")
        
        # Generate in chunks
        current_chunk = []
        
        for i in range(start_idx, num_samples):
            # Randomly select parameters
            ch_type = np.random.choice(channel_types)
            doppler = float(np.random.choice(doppler_values))
            snr = float(np.random.choice(snr_values))
            pilot_dens = float(np.random.choice(pilot_densities))
            
            try:
                sample = self.generate_sample(ch_type, doppler, snr, pilot_dens)
                current_chunk.append(sample)
                
            except Exception as e:
                failed_count += 1
                if failed_count <= 5:
                    tqdm.write(f"  ⚠ Sample {i} failed: {e}")
                continue
            
            pbar.update(1)
            
            # Save chunk checkpoint
            if len(current_chunk) >= chunk_size or i == num_samples - 1:
                # Stack chunk data
                chunk_data = {
                    'rx_symbols': np.stack([s['rx_symbols'] for s in current_chunk]),
                    'tx_symbols': np.stack([s['tx_symbols'] for s in current_chunk]),
                    'H_ls': np.stack([s['H_ls'] for s in current_chunk]),
                    'H_true': np.stack([s['H_true'] for s in current_chunk]),
                    'pilot_mask': np.stack([s['pilot_mask'] for s in current_chunk]),
                    'snr_db': np.array([s['snr_db'] for s in current_chunk]),
                    'channel_type': np.array([s['channel_type'] for s in current_chunk]),
                    'doppler_hz': np.array([s['doppler_hz'] for s in current_chunk]),
                    'pilot_density': np.array([s['pilot_density'] for s in current_chunk]),
                }
                
                # Save chunk
                chunk_path = self.get_temp_data_path(split, chunk_id)
                np.savez_compressed(chunk_path, **chunk_data)
                
                # Save checkpoint
                self.save_checkpoint(split, i + 1, num_samples, chunk_id + 1, start_time)
                
                elapsed = time.time() - start_time
                rate = (i + 1 - start_idx) / elapsed if elapsed > 0 else 0
                tqdm.write(f"  💾 Checkpoint saved: chunk {chunk_id}, {i+1}/{num_samples} ({rate:.2f} samples/s)")
                
                chunk_id += 1
                current_chunk = []
        
        pbar.close()
        
        # Merge all chunks
        print(f"\n📦 Merging {chunk_id} chunks...")
        merged_data = self._merge_chunks(split, chunk_id)
        
        # Save final dataset
        final_path = self.output_dir / f'{split}.npz'
        np.savez_compressed(str(final_path), **merged_data)
        
        # Cleanup checkpoints
        self._cleanup_chunks(split, chunk_id)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Generated {merged_data['rx_symbols'].shape[0]} samples in {elapsed/60:.1f} minutes")
        print(f"   Rate: {merged_data['rx_symbols'].shape[0]/elapsed:.2f} samples/second")
        
        if failed_count > 0:
            print(f"   Failed samples: {failed_count}")
        
        return merged_data
    
    def _merge_chunks(self, split: str, num_chunks: int) -> dict:
        """Merge all chunk files into single dataset."""
        all_data = {
            'rx_symbols': [],
            'tx_symbols': [],
            'H_ls': [],
            'H_true': [],
            'pilot_mask': [],
            'snr_db': [],
            'channel_type': [],
            'doppler_hz': [],
            'pilot_density': [],
        }
        
        for chunk_id in range(num_chunks):
            chunk_path = self.get_temp_data_path(split, chunk_id)
            if chunk_path.exists():
                chunk = np.load(chunk_path, allow_pickle=True)
                for key in all_data:
                    all_data[key].append(chunk[key])
        
        # Concatenate all chunks
        merged = {}
        for key in all_data:
            if len(all_data[key]) > 0:
                merged[key] = np.concatenate(all_data[key], axis=0)
        
        return merged
    
    def _cleanup_chunks(self, split: str, num_chunks: int):
        """Remove temporary chunk files."""
        for chunk_id in range(num_chunks):
            chunk_path = self.get_temp_data_path(split, chunk_id)
            if chunk_path.exists():
                chunk_path.unlink()
        
        checkpoint_path = self.get_checkpoint_path(split)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        print(f"   Cleaned up {num_chunks} chunk files")
    
    def print_dataset_info(self, dataset: dict, name: str):
        """Print dataset information."""
        print(f"\n{name} Dataset Info:")
        print(f"  Samples: {dataset['rx_symbols'].shape[0]}")
        print(f"  RX symbols shape: {dataset['rx_symbols'].shape}")
        print(f"  H_true shape: {dataset['H_true'].shape}")
        print(f"  Memory usage: {sum(v.nbytes for v in dataset.values()) / 1024**2:.1f} MB")
        print(f"  SNR range: [{dataset['snr_db'].min():.0f}, {dataset['snr_db'].max():.0f}] dB")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Phase 3: Robust Dataset Generation with Checkpointing'
    )
    
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples per split (default: 1000)')
    parser.add_argument('--train', type=int, default=None,
                       help='Training samples (overrides --samples)')
    parser.add_argument('--val', type=int, default=None,
                       help='Validation samples')
    parser.add_argument('--test', type=int, default=None,
                       help='Test samples')
    parser.add_argument('--checkpoint', type=int, default=500,
                       help='Samples per checkpoint (default: 500)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last checkpoint')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory (default: data)')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Determine sample counts
    train_samples = args.train if args.train else args.samples
    val_samples = args.val if args.val else max(args.samples // 10, 100)
    test_samples = args.test if args.test else max(args.samples // 5, 200)
    
    print("\n" + "="*60)
    print("   PHASE 3: ROBUST DATASET GENERATION")
    print("   With Checkpointing and Resume Support")
    print("="*60)
    
    # Initialize generator
    generator = RobustDatasetGenerator(args.config, args.output_dir)
    
    # Sample counts
    sample_counts = {
        'train': train_samples,
        'val': val_samples,
        'test': test_samples
    }
    
    print(f"\nDatasets to generate:")
    for split, count in sample_counts.items():
        print(f"  {split}: {count} samples")
    print(f"Checkpoint interval: {args.checkpoint} samples")
    
    # Generate each split
    for split, count in sample_counts.items():
        dataset = generator.generate_dataset_chunked(
            num_samples=count,
            split=split,
            chunk_size=args.checkpoint,
            resume=args.resume
        )
        generator.print_dataset_info(dataset, split.capitalize())
    
    # Final summary
    print("\n" + "="*60)
    print("✅ PHASE 3 COMPLETE: Dataset Generation")
    print("="*60)
    
    output_dir = Path(args.output_dir)
    print(f"\nGenerated files:")
    for split in ['train', 'val', 'test']:
        filepath = output_dir / f'{split}.npz'
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"  ✓ {filepath} ({size_mb:.1f} MB)")
    
    print(f"\nNext step: Run Phase 4 - Model Training")
    print(f"  python run_phase4_training.py --model cnn --epochs 50")


if __name__ == '__main__':
    main()
