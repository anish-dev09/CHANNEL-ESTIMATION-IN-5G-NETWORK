"""
Phase 3: Dataset Verification Script
=====================================

Verifies the integrity and statistics of generated datasets.

Usage:
    python verify_phase3_datasets.py
    python verify_phase3_datasets.py --data-dir data --detailed
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from utils import calculate_nmse, linear2db


def verify_dataset(filepath: str, detailed: bool = False) -> dict:
    """
    Verify a single dataset file.
    
    Args:
        filepath: Path to .npz file
        detailed: Print detailed statistics
        
    Returns:
        Verification results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Verifying: {filepath}")
    print('='*60)
    
    # Load dataset
    try:
        data = np.load(filepath, allow_pickle=True)
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        return {'status': 'failed', 'error': str(e)}
    
    results = {'status': 'success', 'filepath': filepath}
    
    # Check required keys
    required_keys = ['rx_symbols', 'tx_symbols', 'H_ls', 'H_true', 
                     'pilot_mask', 'snr_db', 'channel_type', 'doppler_hz']
    
    missing_keys = [k for k in required_keys if k not in data.keys()]
    if missing_keys:
        print(f"  ✗ Missing keys: {missing_keys}")
        results['status'] = 'incomplete'
        results['missing_keys'] = missing_keys
        return results
    
    print(f"  ✓ All required keys present")
    
    # Get sample count
    num_samples = data['rx_symbols'].shape[0]
    results['num_samples'] = num_samples
    print(f"  ✓ Number of samples: {num_samples}")
    
    # Verify shapes
    print(f"\n  Array Shapes:")
    expected_shapes = {
        'rx_symbols': (num_samples, 14, 2, 599),
        'tx_symbols': (num_samples, 14, 2, 599),
        'H_ls': (num_samples, 14, 2, 2, 599),
        'H_true': (num_samples, 14, 2, 2, 599),
        'pilot_mask': (num_samples, 14, 599),
    }
    
    shape_errors = []
    for key, expected in expected_shapes.items():
        actual = data[key].shape
        status = "✓" if actual == expected else "✗"
        print(f"    {status} {key}: {actual} (expected: {expected})")
        if actual != expected:
            shape_errors.append(key)
    
    if shape_errors:
        results['status'] = 'shape_mismatch'
        results['shape_errors'] = shape_errors
    
    # Check data types
    print(f"\n  Data Types:")
    print(f"    rx_symbols: {data['rx_symbols'].dtype}")
    print(f"    H_true: {data['H_true'].dtype}")
    print(f"    pilot_mask: {data['pilot_mask'].dtype}")
    print(f"    snr_db: {data['snr_db'].dtype}")
    
    # Check for NaN/Inf values
    print(f"\n  Data Integrity:")
    nan_count = 0
    inf_count = 0
    for key in ['rx_symbols', 'H_ls', 'H_true']:
        arr = data[key]
        nans = np.isnan(arr).sum()
        infs = np.isinf(arr).sum()
        nan_count += nans
        inf_count += infs
        if nans > 0 or infs > 0:
            print(f"    ✗ {key}: {nans} NaN, {infs} Inf values")
    
    if nan_count == 0 and inf_count == 0:
        print(f"    ✓ No NaN or Inf values found")
    else:
        results['status'] = 'data_errors'
        results['nan_count'] = nan_count
        results['inf_count'] = inf_count
    
    # Check parameter distributions
    print(f"\n  Parameter Distributions:")
    
    # SNR distribution
    snr_values = data['snr_db']
    snr_unique, snr_counts = np.unique(snr_values, return_counts=True)
    print(f"    SNR range: [{snr_values.min():.0f}, {snr_values.max():.0f}] dB")
    results['snr_range'] = [float(snr_values.min()), float(snr_values.max())]
    
    if detailed:
        print(f"    SNR distribution:")
        for snr, count in zip(snr_unique, snr_counts):
            pct = 100 * count / num_samples
            print(f"      {snr:5.0f} dB: {count:5d} samples ({pct:5.1f}%)")
    
    # Channel type distribution
    channel_types = data['channel_type']
    ch_unique, ch_counts = np.unique(channel_types, return_counts=True)
    print(f"    Channel types: {list(ch_unique)}")
    results['channel_types'] = list(ch_unique)
    
    if detailed:
        print(f"    Channel distribution:")
        for ch, count in zip(ch_unique, ch_counts):
            pct = 100 * count / num_samples
            print(f"      {ch}: {count:5d} samples ({pct:5.1f}%)")
    
    # Doppler distribution
    doppler_values = data['doppler_hz']
    dop_unique, dop_counts = np.unique(doppler_values, return_counts=True)
    print(f"    Doppler range: [{doppler_values.min():.0f}, {doppler_values.max():.0f}] Hz")
    results['doppler_range'] = [float(doppler_values.min()), float(doppler_values.max())]
    
    # Pilot density distribution
    pilot_density = data['pilot_density']
    pd_unique, pd_counts = np.unique(pilot_density, return_counts=True)
    print(f"    Pilot densities: {[f'{p*100:.0f}%' for p in pd_unique]}")
    results['pilot_densities'] = list(pd_unique)
    
    # Check LS estimation quality
    print(f"\n  LS Estimation Quality (sample check):")
    sample_indices = np.random.choice(num_samples, min(10, num_samples), replace=False)
    nmse_values = []
    
    for idx in sample_indices:
        H_true = data['H_true'][idx, :, 0, 0, :]
        H_ls = data['H_ls'][idx, :, 0, 0, :]
        nmse = calculate_nmse(H_true, H_ls)
        nmse_values.append(linear2db(nmse))
    
    avg_nmse = np.mean(nmse_values)
    print(f"    Average LS NMSE: {avg_nmse:.2f} dB")
    results['avg_ls_nmse_db'] = float(avg_nmse)
    
    # Pilot mask check
    print(f"\n  Pilot Mask Check:")
    pilot_densities_actual = []
    for idx in range(min(10, num_samples)):
        mask = data['pilot_mask'][idx]
        density = mask.sum() / mask.size
        pilot_densities_actual.append(density)
    avg_density = np.mean(pilot_densities_actual)
    print(f"    Average pilot density: {avg_density*100:.1f}%")
    results['avg_pilot_density'] = float(avg_density)
    
    # Final status
    print(f"\n  {'='*50}")
    if results['status'] == 'success':
        print(f"  ✓ VERIFICATION PASSED")
    else:
        print(f"  ✗ VERIFICATION FAILED: {results['status']}")
    
    return results


def main():
    """Main verification script."""
    parser = argparse.ArgumentParser(description='Verify Phase 3 Datasets')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory (default: data)')
    parser.add_argument('--detailed', action='store_true',
                       help='Print detailed statistics')
    parser.add_argument('--file', type=str, default=None,
                       help='Verify specific file')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("   PHASE 3: DATASET VERIFICATION")
    print("   AI-Assisted Channel Estimation in 5G")
    print("="*60)
    
    data_dir = Path(args.data_dir)
    
    if args.file:
        # Verify single file
        files = [Path(args.file)]
    else:
        # Find all .npz files
        files = list(data_dir.glob('*.npz'))
    
    if not files:
        print(f"\n  ✗ No .npz files found in {data_dir}")
        print(f"    Run: python run_phase3_dataset_generation.py --samples 1000")
        return
    
    print(f"\nFound {len(files)} dataset file(s) to verify:")
    for f in files:
        print(f"  - {f}")
    
    # Verify each file
    all_results = {}
    
    for filepath in files:
        results = verify_dataset(str(filepath), detailed=args.detailed)
        all_results[filepath.stem] = results
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    total_samples = 0
    all_passed = True
    
    for name, results in all_results.items():
        status = results.get('status', 'unknown')
        samples = results.get('num_samples', 0)
        total_samples += samples
        
        if status == 'success':
            print(f"  ✓ {name}: {samples:,} samples - PASSED")
        else:
            print(f"  ✗ {name}: {status}")
            all_passed = False
    
    print(f"\nTotal samples: {total_samples:,}")
    
    if all_passed:
        print("\n✓ All datasets verified successfully!")
        print("\nNext step: Run Phase 4 - CNN Model Training")
        print("  python run_phase4_training.py --model cnn --epochs 50")
    else:
        print("\n✗ Some datasets failed verification")
        print("  Please regenerate: python run_phase3_dataset_generation.py")
    
    print("="*60 + "\n")
    
    return all_results


if __name__ == '__main__':
    main()
