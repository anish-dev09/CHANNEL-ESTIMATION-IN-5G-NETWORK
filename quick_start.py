"""
Quick start demo script for AI-Assisted Channel Estimation in 5G.

This script demonstrates the complete workflow:
1. Simulate MIMO-OFDM transmission
2. Apply baseline estimators (LS, MMSE)
3. Compare performance
4. Visualize results
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.append('src')

from channel_simulator import simulate_transmission
from baseline_estimators import LSEstimator, MMSEEstimator, evaluate_estimator
from utils import load_config, set_seed

# Set seed for reproducibility
set_seed(42)


def run_quick_demo():
    """Run a quick demonstration of the system."""
    
    print("="*70)
    print("AI-ASSISTED CHANNEL ESTIMATION IN 5G - QUICK DEMO")
    print("="*70)
    print()
    
    # Load configuration
    print("[1/5] Loading configuration...")
    config = load_config('configs/experiment_config.yaml')
    print("      ✓ Configuration loaded")
    
    # Simulate transmission
    print("\n[2/5] Simulating MIMO-OFDM transmission...")
    print("      Channel: EVA")
    print("      Doppler: 50 Hz (~60 km/h)")
    print("      SNR: 15 dB")
    print("      Pilot density: 10%")
    
    sim_data = simulate_transmission(
        config,
        channel_type='EVA',
        doppler_hz=50,
        snr_db=15,
        pilot_density=0.1
    )
    
    rx_symbols = sim_data['rx_symbols']
    H_true = sim_data['channel']
    pilot_pattern = sim_data['pilot_pattern']
    pilot_symbols = sim_data['pilot_symbols']
    
    print("      ✓ Transmission simulated")
    print(f"      Shape: {rx_symbols.shape}")
    
    # Prepare data for estimators
    num_symbols, num_rx, num_sc = rx_symbols.shape
    num_tx = sim_data['tx_symbols'].shape[1]
    
    # Create 4D arrays
    rx_4d = rx_symbols.reshape(num_symbols, num_rx, 1, num_sc)
    rx_4d = np.repeat(rx_4d, num_tx, axis=2)
    
    # LS Estimation
    print("\n[3/5] Running LS channel estimation...")
    ls_estimator = LSEstimator(interpolation_method='linear')
    H_ls = ls_estimator.estimate(
        rx_4d, pilot_symbols,
        pilot_pattern.pilot_mask,
        pilot_pattern.pilot_positions
    )
    
    ls_metrics = evaluate_estimator(H_true, H_ls)
    print(f"      ✓ LS NMSE: {ls_metrics['nmse_db']:.2f} dB")
    
    # MMSE Estimation
    print("\n[4/5] Running MMSE channel estimation...")
    mmse_estimator = MMSEEstimator(estimate_statistics=True)
    H_mmse = mmse_estimator.estimate(
        rx_4d, pilot_symbols,
        pilot_pattern.pilot_mask,
        pilot_pattern.pilot_positions,
        snr_db=15
    )
    
    mmse_metrics = evaluate_estimator(H_true, H_mmse)
    print(f"      ✓ MMSE NMSE: {mmse_metrics['nmse_db']:.2f} dB")
    
    # Visualization
    print("\n[5/5] Creating visualizations...")
    create_visualizations(H_true, H_ls, H_mmse, pilot_pattern, ls_metrics, mmse_metrics)
    print("      ✓ Plots saved to 'results/' directory")
    
    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"True Channel Power: {np.mean(np.abs(H_true)**2):.6f}")
    print()
    print("LS Estimator:")
    print(f"  NMSE: {ls_metrics['nmse_db']:.2f} dB")
    print(f"  MSE:  {ls_metrics['mse']:.6f}")
    print()
    print("MMSE Estimator:")
    print(f"  NMSE: {mmse_metrics['nmse_db']:.2f} dB")
    print(f"  MSE:  {mmse_metrics['mse']:.6f}")
    print()
    
    improvement = ls_metrics['nmse_db'] - mmse_metrics['nmse_db']
    print(f"MMSE Improvement over LS: {improvement:.2f} dB")
    print("="*70)
    
    print("\n✓ Demo completed successfully!")
    print("\nNext steps:")
    print("  1. Generate larger dataset: python src/dataset_generator.py")
    print("  2. Train AI model: python src/train.py --model cnn")
    print("  3. Evaluate: python src/evaluate.py --model-types cnn")


def create_visualizations(H_true, H_ls, H_mmse, pilot_pattern, ls_metrics, mmse_metrics):
    """Create and save visualization plots."""
    
    # Create results directory
    Path('results').mkdir(exist_ok=True)
    
    # Take first RX, TX antenna for visualization
    H_true_plot = H_true[:, 0, 0, :]
    H_ls_plot = H_ls[:, 0, 0, :]
    H_mmse_plot = H_mmse[:, 0, 0, :]
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. True channel magnitude
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(np.abs(H_true_plot), aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_title('True Channel Magnitude', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Subcarrier Index')
    ax1.set_ylabel('OFDM Symbol')
    plt.colorbar(im1, ax=ax1)
    
    # 2. LS estimate
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(np.abs(H_ls_plot), aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_title(f'LS Estimate (NMSE: {ls_metrics["nmse_db"]:.2f} dB)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Subcarrier Index')
    ax2.set_ylabel('OFDM Symbol')
    plt.colorbar(im2, ax=ax2)
    
    # 3. MMSE estimate
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(np.abs(H_mmse_plot), aspect='auto', cmap='viridis', interpolation='nearest')
    ax3.set_title(f'MMSE Estimate (NMSE: {mmse_metrics["nmse_db"]:.2f} dB)', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Subcarrier Index')
    ax3.set_ylabel('OFDM Symbol')
    plt.colorbar(im3, ax=ax3)
    
    # 4. LS error
    ax4 = plt.subplot(2, 3, 4)
    ls_error = np.abs(H_true_plot - H_ls_plot)
    im4 = ax4.imshow(ls_error, aspect='auto', cmap='hot', interpolation='nearest')
    ax4.set_title('LS Estimation Error', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Subcarrier Index')
    ax4.set_ylabel('OFDM Symbol')
    plt.colorbar(im4, ax=ax4)
    
    # 5. MMSE error
    ax5 = plt.subplot(2, 3, 5)
    mmse_error = np.abs(H_true_plot - H_mmse_plot)
    im5 = ax5.imshow(mmse_error, aspect='auto', cmap='hot', interpolation='nearest')
    ax5.set_title('MMSE Estimation Error', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Subcarrier Index')
    ax5.set_ylabel('OFDM Symbol')
    plt.colorbar(im5, ax=ax5)
    
    # 6. Pilot pattern
    ax6 = plt.subplot(2, 3, 6)
    im6 = ax6.imshow(pilot_pattern.pilot_mask, aspect='auto', cmap='binary', interpolation='nearest')
    ax6.set_title(f'Pilot Pattern ({np.sum(pilot_pattern.pilot_mask)/pilot_pattern.pilot_mask.size*100:.1f}% density)', 
                  fontsize=12, fontweight='bold')
    ax6.set_xlabel('Subcarrier Index')
    ax6.set_ylabel('OFDM Symbol')
    plt.colorbar(im6, ax=ax6)
    
    plt.tight_layout()
    plt.savefig('results/quick_demo_results.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Additional plot: Channel frequency response at one symbol
    fig2, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    symbol_idx = 7  # Middle symbol
    
    # Magnitude
    axes[0].plot(np.abs(H_true_plot[symbol_idx, :]), 'k-', linewidth=2, label='True', alpha=0.8)
    axes[0].plot(np.abs(H_ls_plot[symbol_idx, :]), 'b--', linewidth=1.5, label='LS Estimate')
    axes[0].plot(np.abs(H_mmse_plot[symbol_idx, :]), 'r:', linewidth=1.5, label='MMSE Estimate')
    axes[0].scatter(np.where(pilot_pattern.pilot_mask[symbol_idx])[0],
                    np.abs(H_true_plot[symbol_idx, pilot_pattern.pilot_mask[symbol_idx]]),
                    c='green', s=50, marker='x', label='Pilots', zorder=10)
    axes[0].set_xlabel('Subcarrier Index', fontsize=11)
    axes[0].set_ylabel('Channel Magnitude', fontsize=11)
    axes[0].set_title(f'Channel Frequency Response (Symbol {symbol_idx})', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Phase
    axes[1].plot(np.angle(H_true_plot[symbol_idx, :]), 'k-', linewidth=2, label='True', alpha=0.8)
    axes[1].plot(np.angle(H_ls_plot[symbol_idx, :]), 'b--', linewidth=1.5, label='LS Estimate')
    axes[1].plot(np.angle(H_mmse_plot[symbol_idx, :]), 'r:', linewidth=1.5, label='MMSE Estimate')
    axes[1].scatter(np.where(pilot_pattern.pilot_mask[symbol_idx])[0],
                    np.angle(H_true_plot[symbol_idx, pilot_pattern.pilot_mask[symbol_idx]]),
                    c='green', s=50, marker='x', label='Pilots', zorder=10)
    axes[1].set_xlabel('Subcarrier Index', fontsize=11)
    axes[1].set_ylabel('Channel Phase (rad)', fontsize=11)
    axes[1].set_title('Channel Phase Response', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/channel_response_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    run_quick_demo()
