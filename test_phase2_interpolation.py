"""
PHASE 2.4 - DETAILED INTERPOLATION METHOD ANALYSIS
Test different interpolation methods across various conditions
"""

import numpy as np
import yaml
from src.channel_simulator import simulate_transmission
from src.baseline_estimators import LSEstimator
from src.utils import calculate_nmse, linear2db

def load_config():
    with open('configs/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return config

def test_interpolation(config, snr_db, pilot_density, interp_method, channel_type='EVA'):
    """Test single interpolation configuration"""
    # Simulate transmission
    output = simulate_transmission(
        config=config,
        channel_type=channel_type,
        snr_db=snr_db,
        pilot_density=pilot_density,
        doppler_hz=50
    )
    
    rx_symbols = output['rx_symbols']
    tx_symbols = output['tx_symbols']
    pilot_pattern = output['pilot_pattern']
    tx_pilots = output['pilot_symbols']
    H_true = output['channel'][:, 0, 0, :]  # Shape: (num_symbols, num_subcarriers)
    
    # Get pilot positions
    pilot_mask = pilot_pattern.pilot_mask
    pilot_positions = np.where(pilot_mask)
    
    # Reshape to 4D if needed
    if rx_symbols.ndim == 3:
        num_symbols, num_rx, num_subcarriers = rx_symbols.shape
        num_tx = tx_symbols.shape[1]
        rx_symbols_4d = np.zeros((num_symbols, num_rx, num_tx, num_subcarriers), dtype=complex)
        for tx_idx in range(num_tx):
            rx_symbols_4d[:, :, tx_idx, :] = rx_symbols
        rx_symbols = rx_symbols_4d
    
    # Run LS estimation
    ls = LSEstimator(interpolation_method=interp_method)
    H_est_full = ls.estimate(rx_symbols, tx_pilots, pilot_mask, pilot_positions)
    H_est = H_est_full[:, 0, 0, :]  # Extract first antenna pair
    
    # Calculate NMSE in dB
    nmse = calculate_nmse(H_true, H_est)
    nmse_db = linear2db(nmse)
    return nmse_db

def main():
    print("\n" + "="*70)
    print("PHASE 2.4 - DETAILED INTERPOLATION ANALYSIS")
    print("="*70)
    
    config = load_config()
    
    # Test configurations
    interpolation_methods = ['linear', 'cubic', 'nearest']
    snr_levels = [5, 10, 15, 20, 25]
    pilot_densities = [0.05, 0.10, 0.15, 0.20]
    channel_types = ['EPA', 'EVA', 'ETU']
    
    print("\n" + "="*70)
    print("TEST 1: Interpolation Methods Across SNR Levels")
    print("="*70)
    print(f"Configuration: EVA channel, 10% pilots\n")
    
    results_snr = {method: [] for method in interpolation_methods}
    
    for snr in snr_levels:
        print(f"\n--- SNR = {snr} dB ---")
        for method in interpolation_methods:
            nmse = test_interpolation(config, snr, 0.10, method, 'EVA')
            results_snr[method].append(nmse)
            print(f"  {method:8s}: {nmse:6.2f} dB")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE - Interpolation vs SNR")
    print("="*70)
    print(f"{'SNR (dB)':<10} {'Linear':>12} {'Cubic':>12} {'Nearest':>12} {'Best':>12}")
    print("-" * 70)
    for i, snr in enumerate(snr_levels):
        values = [results_snr[m][i] for m in interpolation_methods]
        best_idx = np.argmin(values)
        best_method = interpolation_methods[best_idx]
        print(f"{snr:<10} {results_snr['linear'][i]:12.2f} {results_snr['cubic'][i]:12.2f} "
              f"{results_snr['nearest'][i]:12.2f} {best_method:>12}")
    
    # Average performance
    print("\n" + "-" * 70)
    for method in interpolation_methods:
        avg_nmse = np.mean(results_snr[method])
        print(f"Average NMSE ({method}): {avg_nmse:.2f} dB")
    
    best_overall = min(interpolation_methods, key=lambda m: np.mean(results_snr[m]))
    print(f"\n✓ Best Overall Method: {best_overall.upper()}")
    
    # Test 2: Pilot Density Impact
    print("\n\n" + "="*70)
    print("TEST 2: Interpolation Methods Across Pilot Densities")
    print("="*70)
    print(f"Configuration: EVA channel, SNR = 15 dB\n")
    
    results_pilot = {method: [] for method in interpolation_methods}
    
    for density in pilot_densities:
        print(f"\n--- Pilot Density = {density*100:.0f}% ---")
        for method in interpolation_methods:
            nmse = test_interpolation(config, 15, density, method, 'EVA')
            results_pilot[method].append(nmse)
            print(f"  {method:8s}: {nmse:6.2f} dB")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE - Interpolation vs Pilot Density")
    print("="*70)
    print(f"{'Pilots':<10} {'Linear':>12} {'Cubic':>12} {'Nearest':>12} {'Best':>12}")
    print("-" * 70)
    for i, density in enumerate(pilot_densities):
        values = [results_pilot[m][i] for m in interpolation_methods]
        best_idx = np.argmin(values)
        best_method = interpolation_methods[best_idx]
        print(f"{density*100:<10.0f}% {results_pilot['linear'][i]:11.2f} {results_pilot['cubic'][i]:12.2f} "
              f"{results_pilot['nearest'][i]:12.2f} {best_method:>12}")
    
    # Average performance
    print("\n" + "-" * 70)
    for method in interpolation_methods:
        avg_nmse = np.mean(results_pilot[method])
        print(f"Average NMSE ({method}): {avg_nmse:.2f} dB")
    
    # Test 3: Channel Type Impact
    print("\n\n" + "="*70)
    print("TEST 3: Interpolation Methods Across Channel Types")
    print("="*70)
    print(f"Configuration: SNR = 15 dB, 10% pilots\n")
    
    results_channel = {method: [] for method in interpolation_methods}
    
    for ch_type in channel_types:
        print(f"\n--- Channel: {ch_type} ---")
        for method in interpolation_methods:
            nmse = test_interpolation(config, 15, 0.10, method, ch_type)
            results_channel[method].append(nmse)
            print(f"  {method:8s}: {nmse:6.2f} dB")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY TABLE - Interpolation vs Channel Type")
    print("="*70)
    print(f"{'Channel':<10} {'Linear':>12} {'Cubic':>12} {'Nearest':>12} {'Best':>12}")
    print("-" * 70)
    for i, ch_type in enumerate(channel_types):
        values = [results_channel[m][i] for m in interpolation_methods]
        best_idx = np.argmin(values)
        best_method = interpolation_methods[best_idx]
        print(f"{ch_type:<10} {results_channel['linear'][i]:12.2f} {results_channel['cubic'][i]:12.2f} "
              f"{results_channel['nearest'][i]:12.2f} {best_method:>12}")
    
    # Final recommendations
    print("\n\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Calculate win counts
    win_counts = {method: 0 for method in interpolation_methods}
    
    for i in range(len(snr_levels)):
        values = [results_snr[m][i] for m in interpolation_methods]
        best_method = interpolation_methods[np.argmin(values)]
        win_counts[best_method] += 1
    
    for i in range(len(pilot_densities)):
        values = [results_pilot[m][i] for m in interpolation_methods]
        best_method = interpolation_methods[np.argmin(values)]
        win_counts[best_method] += 1
    
    for i in range(len(channel_types)):
        values = [results_channel[m][i] for m in interpolation_methods]
        best_method = interpolation_methods[np.argmin(values)]
        win_counts[best_method] += 1
    
    print("\nWin Count (Best performance in each test):")
    for method, count in sorted(win_counts.items(), key=lambda x: x[1], reverse=True):
        total_tests = len(snr_levels) + len(pilot_densities) + len(channel_types)
        print(f"  {method:8s}: {count}/{total_tests} tests ({count*100/total_tests:.1f}%)")
    
    # Overall averages
    overall_avg = {}
    for method in interpolation_methods:
        all_nmse = (results_snr[method] + results_pilot[method] + results_channel[method])
        overall_avg[method] = np.mean(all_nmse)
    
    print("\nOverall Average NMSE:")
    for method in sorted(overall_avg, key=overall_avg.get):
        print(f"  {method:8s}: {overall_avg[method]:.2f} dB")
    
    best_method = min(overall_avg, key=overall_avg.get)
    print(f"\n✓ RECOMMENDED METHOD: {best_method.upper()}")
    print(f"  Reason: Best average NMSE across all conditions")
    
    print("\n" + "="*70)
    print("✅ INTERPOLATION ANALYSIS COMPLETE!")
    print("="*70)
    print("\n✓ Phase 2.4 Complete - Proceed to Task 2.5\n")

if __name__ == "__main__":
    main()
