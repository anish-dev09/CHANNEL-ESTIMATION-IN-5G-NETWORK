"""
Phase 1 - Visualization Script: Channel Characteristics
Creates visualizations of channel responses for different models
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
from src.channel_simulator import ChannelModel

def load_config():
    """Load experiment configuration"""
    with open('configs/experiment_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def visualize_channel_models():
    """Generate comprehensive channel visualizations"""
    print("\n" + "="*70)
    print("PHASE 1 - CHANNEL VISUALIZATION")
    print("="*70 + "\n")
    
    config = load_config()
    
    channel_types = ['EPA', 'EVA', 'ETU']
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle('Channel Model Comparison: EPA, EVA, ETU', fontsize=16, fontweight='bold')
    
    print("Generating channel responses using simulate_transmission...\n")
    
    for idx, channel_type in enumerate(channel_types):
        print(f"  Processing {channel_type} channel...")
        
        # Simulate transmission to get channel response
        from src.channel_simulator import simulate_transmission
        output = simulate_transmission(
            config=config,
            channel_type=channel_type,
            doppler_hz=50,
            snr_db=20,
            pilot_density=0.01
        )
        
        # Extract channel for first TX-RX pair
        H = output['channel'][:, 0, 0, :]  # Shape: (num_symbols, num_subcarriers)
        num_symbols, num_subcarriers = H.shape
        
        # Plot 1: Magnitude (Time-Frequency)
        ax1 = axes[idx, 0]
        im1 = ax1.imshow(np.abs(H).T, aspect='auto', cmap='viridis', origin='lower',
                         extent=[0, num_symbols, 0, num_subcarriers])
        ax1.set_title(f'{channel_type}: Channel Magnitude |H|', fontweight='bold')
        ax1.set_xlabel('OFDM Symbol Index')
        ax1.set_ylabel('Subcarrier Index')
        plt.colorbar(im1, ax=ax1, label='|H|')
        
        # Plot 2: Frequency Response (averaged over time)
        ax2 = axes[idx, 1]
        mean_magnitude = np.mean(np.abs(H), axis=0)
        std_magnitude = np.std(np.abs(H), axis=0)
        subcarrier_indices = np.arange(num_subcarriers)
        
        ax2.plot(subcarrier_indices, mean_magnitude, 'b-', linewidth=1.5, label='Mean')
        ax2.fill_between(subcarrier_indices, 
                         mean_magnitude - std_magnitude,
                         mean_magnitude + std_magnitude,
                         alpha=0.3, label='±1 Std Dev')
        ax2.set_title(f'{channel_type}: Frequency Response', fontweight='bold')
        ax2.set_xlabel('Subcarrier Index')
        ax2.set_ylabel('|H|')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Calculate and display statistics
        mean_power = np.mean(np.abs(H)**2)
        temporal_var = np.std(np.abs(H), axis=0).mean()
        frequency_var = np.std(np.abs(H), axis=1).mean()
        
        print(f"    Mean Power: {mean_power:.4f}")
        print(f"    Temporal Variation: {temporal_var:.4f}")
        print(f"    Frequency Variation: {frequency_var:.4f}")
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    output_path = 'results/phase1_channel_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved: {output_path}")
    
    plt.show()
    print("  ✓ Displaying plot...\n")

def visualize_doppler_effects():
    """Visualize impact of different Doppler frequencies"""
    print("\n" + "="*70)
    print("DOPPLER EFFECTS VISUALIZATION")
    print("="*70 + "\n")
    
    config = load_config()
    carrier_freq = float(config['channel']['carrier_freq'])
    
    doppler_frequencies = [10, 50, 100, 200]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Doppler Effect on EVA Channel (10-200 Hz)', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    print("Generating Doppler effect visualizations...\n")
    
    from src.channel_simulator import simulate_transmission
    
    for idx, doppler_hz in enumerate(doppler_frequencies):
        velocity_kmh = doppler_hz * 3e8 / carrier_freq / 1000 * 3.6
        print(f"  Doppler {doppler_hz} Hz (~{velocity_kmh:.0f} km/h)...")
        
        output = simulate_transmission(
            config=config,
            channel_type='EVA',
            doppler_hz=doppler_hz,
            snr_db=20,
            pilot_density=0.01
        )
        
        H = output['channel'][:, 0, 0, :]
        num_symbols, num_subcarriers = H.shape
        
        ax = axes[idx]
        im = ax.imshow(np.abs(H).T, aspect='auto', cmap='viridis', origin='lower',
                      extent=[0, num_symbols, 0, num_subcarriers])
        
        temporal_var = np.std(np.abs(H), axis=0).mean()
        
        ax.set_title(f'Doppler = {doppler_hz} Hz (~{velocity_kmh:.0f} km/h)\n' + 
                    f'Temporal Variation: {temporal_var:.4f}', fontweight='bold')
        ax.set_xlabel('OFDM Symbol Index')
        ax.set_ylabel('Subcarrier Index')
        plt.colorbar(im, ax=ax, label='|H|')
    
    plt.tight_layout()
    
    output_path = 'results/phase1_doppler_effects.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✓ Saved: {output_path}")
    
    plt.show()
    print("  ✓ Displaying plot...\n")

def visualize_temporal_variation():
    """Visualize channel temporal variation"""
    print("\n" + "="*70)
    print("TEMPORAL VARIATION ANALYSIS")
    print("="*70 + "\n")
    
    config = load_config()
    
    # Create modified config with more symbols
    extended_config = config.copy()
    extended_config['ofdm'] = config['ofdm'].copy()
    extended_config['ofdm']['num_symbols'] = 50
    
    print("Generating extended temporal observation...\n")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Channel Temporal Variation Analysis (EVA, 50 Hz Doppler)', 
                 fontsize=16, fontweight='bold')
    
    from src.channel_simulator import simulate_transmission
    output = simulate_transmission(
        config=extended_config,
        channel_type='EVA',
        doppler_hz=50,
        snr_db=20,
        pilot_density=0.01
    )
    
    H = output['channel'][:, 0, 0, :]
    num_symbols, num_subcarriers = H.shape
    
    # Plot 1: Time-frequency heatmap
    ax1 = axes[0]
    im1 = ax1.imshow(np.abs(H).T, aspect='auto', cmap='viridis', origin='lower',
                     extent=[0, num_symbols, 0, num_subcarriers])
    ax1.set_title('Channel Magnitude Variation', fontweight='bold')
    ax1.set_xlabel('OFDM Symbol Index')
    ax1.set_ylabel('Subcarrier Index')
    plt.colorbar(im1, ax=ax1, label='|H|')
    
    # Plot 2: Sample subcarrier evolution
    ax2 = axes[1]
    sample_subcarriers = [num_subcarriers//4, num_subcarriers//2, 3*num_subcarriers//4]
    for sc in sample_subcarriers:
        ax2.plot(np.abs(H[:, sc]), label=f'Subcarrier {sc}', linewidth=2)
    ax2.set_title('Channel Magnitude Evolution (Sample Subcarriers)', fontweight='bold')
    ax2.set_xlabel('OFDM Symbol Index')
    ax2.set_ylabel('|H|')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Temporal autocorrelation
    ax3 = axes[2]
    center_subcarrier = num_subcarriers // 2
    h_time = H[:, center_subcarrier]
    autocorr = np.correlate(np.abs(h_time), np.abs(h_time), mode='full')
    autocorr = autocorr[len(autocorr)//2:]  # Keep positive lags
    autocorr = autocorr / autocorr[0]  # Normalize
    
    lags = np.arange(len(autocorr))
    ax3.plot(lags, autocorr, 'b-', linewidth=2)
    ax3.axhline(y=0.5, color='r', linestyle='--', label='50% Correlation')
    ax3.set_title('Temporal Autocorrelation (Center Subcarrier)', fontweight='bold')
    ax3.set_xlabel('Symbol Lag')
    ax3.set_ylabel('Normalized Autocorrelation')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 20])
    
    plt.tight_layout()
    
    output_path = 'results/phase1_temporal_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    
    plt.show()
    print("  ✓ Displaying plot...\n")

if __name__ == '__main__':
    try:
        print("\nGenerating Phase 1 visualizations...")
        print("This will create 3 sets of plots:\n")
        print("  1. Channel Model Comparison (EPA, EVA, ETU)")
        print("  2. Doppler Effects (10-200 Hz)")
        print("  3. Temporal Variation Analysis\n")
        
        input("Press Enter to continue...")
        
        # Generate visualizations
        visualize_channel_models()
        visualize_doppler_effects()
        visualize_temporal_variation()
        
        print("\n" + "="*70)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("="*70)
        print("\nOutput files in 'results/' directory:")
        print("  • phase1_channel_visualization.png")
        print("  • phase1_doppler_effects.png")
        print("  • phase1_temporal_analysis.png")
        
        print("\n" + "🎉 "*20)
        print("PHASE 1.4 COMPLETED!")
        print("Next: Run quick_start.py for full system demo (Task 1.6)")
        print("🎉 "*20 + "\n")
        
    except Exception as e:
        print(f"\n❌ Visualization failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
