"""
Phase 1 - Test Script: OFDM Transmission Simulation
Tests complete OFDM transmission pipeline with MIMO
"""

import numpy as np
import yaml
from src.channel_simulator import simulate_transmission

def load_config():
    """Load experiment configuration"""
    with open('configs/experiment_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_ofdm_transmission():
    """Test complete OFDM transmission simulation"""
    print("\n" + "="*70)
    print("PHASE 1 - OFDM TRANSMISSION VALIDATION")
    print("="*70 + "\n")
    
    config = load_config()
    
    print("Configuration:")
    print(f"  FFT Size: {config['ofdm']['fft_size']}")
    print(f"  CP Length: {config['ofdm']['cp_length']}")
    print(f"  OFDM Symbols: {config['ofdm']['num_symbols']}")
    print(f"  Useful Subcarriers: {config['ofdm']['useful_subcarriers']}")
    print(f"  MIMO: {config['mimo']['num_tx_antennas']}x{config['mimo']['num_rx_antennas']}")
    
    # Get first pilot density value
    pilot_density = config['pilots']['density'][0] if isinstance(config['pilots']['density'], list) else config['pilots']['density']
    print(f"  Pilot Density: {pilot_density*100}%\n")
    
    # Test parameters
    test_cases = [
        {'channel': 'EPA', 'doppler': 10, 'snr': 20, 'scenario': 'Low mobility, high SNR'},
        {'channel': 'EVA', 'doppler': 50, 'snr': 15, 'scenario': 'Medium mobility, medium SNR'},
        {'channel': 'ETU', 'doppler': 100, 'snr': 10, 'scenario': 'High mobility, low SNR'},
    ]
    
    results = []
    
    for idx, test in enumerate(test_cases, 1):
        print(f"\n[Test {idx}/3] {test['scenario']}")
        print("-" * 70)
        print(f"  Channel: {test['channel']}")
        print(f"  Doppler: {test['doppler']} Hz")
        print(f"  SNR: {test['snr']} dB\n")
        
        try:
            # Simulate transmission
            output = simulate_transmission(
                config=config,
                channel_type=test['channel'],
                snr_db=test['snr'],
                doppler_hz=test['doppler'],
                pilot_density=pilot_density
            )
            
            # Validate outputs
            rx_symbols = output['rx_symbols']
            tx_symbols = output['tx_symbols']
            H_true = output['channel']
            pilot_pattern = output['pilot_pattern']
            pilot_mask = pilot_pattern.pilot_mask
            
            # Expected shapes (accounting for DC subcarrier removal)
            num_symbols = config['ofdm']['num_symbols']
            num_tx = config['mimo']['num_tx_antennas']
            num_rx = config['mimo']['num_rx_antennas']
            num_sc = config['ofdm']['useful_subcarriers'] - 1  # DC removed
            
            expected_rx_shape = (num_symbols, num_rx, num_sc)
            expected_tx_shape = (num_symbols, num_tx, num_sc)
            expected_H_shape = (num_symbols, num_rx, num_tx, num_sc)
            expected_pilot_shape = (num_symbols, num_sc)
            
            print(f"  Output Shapes:")
            print(f"    RX Symbols: {rx_symbols.shape} (expected: {expected_rx_shape})")
            print(f"    TX Symbols: {tx_symbols.shape} (expected: {expected_tx_shape})")
            print(f"    Channel H:  {H_true.shape} (expected: {expected_H_shape})")
            print(f"    Pilot Mask: {pilot_mask.shape} (expected: {expected_pilot_shape})")
            
            # Validate shapes
            assert rx_symbols.shape == expected_rx_shape, "RX shape mismatch"
            assert tx_symbols.shape == expected_tx_shape, "TX shape mismatch"
            assert H_true.shape == expected_H_shape, "Channel shape mismatch"
            assert pilot_mask.shape == expected_pilot_shape, "Pilot mask shape mismatch"
            print(f"  ✓ All shapes correct")
            
            # Validate pilot density
            actual_pilot_density = pilot_mask.sum() / pilot_mask.size
            expected_pilot_density = pilot_density
            pilot_density_error = abs(actual_pilot_density - expected_pilot_density)
            
            print(f"\n  Pilot Pattern:")
            print(f"    Expected Density: {expected_pilot_density*100:.1f}%")
            print(f"    Actual Density: {actual_pilot_density*100:.1f}%")
            print(f"    Error: {pilot_density_error*100:.2f}%")
            
            assert pilot_density_error < 0.05, f"Pilot density error too large: {pilot_density_error}"
            print(f"  ✓ Pilot density within tolerance")
            
            # Check for valid data
            assert not np.isnan(rx_symbols).any(), "NaN in RX symbols"
            assert not np.isinf(rx_symbols).any(), "Inf in RX symbols"
            print(f"  ✓ No NaN/Inf values")
            
            # Calculate SNR at receiver (rough estimate)
            signal_power = np.mean(np.abs(tx_symbols)**2)
            noise_estimate = np.var(rx_symbols - tx_symbols)  # Simplified
            estimated_snr_db = 10 * np.log10(signal_power / max(noise_estimate, 1e-10))
            
            print(f"\n  Signal Quality:")
            print(f"    TX Power: {10*np.log10(signal_power):.2f} dB")
            print(f"    Estimated SNR: {estimated_snr_db:.2f} dB")
            
            # Channel statistics
            h_magnitude = np.abs(H_true)
            print(f"\n  Channel Statistics:")
            print(f"    Mean |H|: {h_magnitude.mean():.4f}")
            print(f"    Std |H|: {h_magnitude.std():.4f}")
            print(f"    Max |H|: {h_magnitude.max():.4f}")
            print(f"    Min |H|: {h_magnitude.min():.4f}")
            
            results.append({
                'test': f"{test['channel']}-{test['doppler']}Hz-{test['snr']}dB",
                'status': 'PASS',
                'pilot_density': actual_pilot_density,
                'estimated_snr': estimated_snr_db,
                'mean_channel_magnitude': h_magnitude.mean()
            })
            
            print(f"\n  ✅ Test {idx} PASSED\n")
            
        except Exception as e:
            print(f"\n  ❌ Test {idx} FAILED: {str(e)}\n")
            results.append({
                'test': f"{test['channel']}-{test['doppler']}Hz-{test['snr']}dB",
                'status': 'FAIL',
                'error': str(e)
            })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Test Case':<30} {'Status':<10} {'Pilot Density':<15} {'Est. SNR (dB)':<15}")
    print("-" * 70)
    
    for res in results:
        status = res['status']
        test_name = res['test']
        if status == 'PASS':
            print(f"{test_name:<30} {'✅ PASS':<10} {res['pilot_density']*100:>6.2f}%        {res['estimated_snr']:>6.2f}")
        else:
            print(f"{test_name:<30} {'❌ FAIL':<10} {res.get('error', 'Unknown error')}")
    
    # Overall result
    all_passed = all(r['status'] == 'PASS' for r in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL TRANSMISSION TESTS PASSED!")
        print("="*70)
        print("\n✓ Phase 1.3 Complete - Proceed to Task 1.4 (Visualization)\n")
        return True
    else:
        print("❌ SOME TRANSMISSION TESTS FAILED - REVIEW ERRORS ABOVE")
        print("="*70)
        return False

def test_mimo_dimensions():
    """Test different MIMO configurations"""
    print("\n" + "="*70)
    print("TESTING MIMO CONFIGURATIONS")
    print("="*70 + "\n")
    
    config = load_config()
    
    mimo_configs = [
        (1, 1, "SISO"),
        (2, 2, "2x2 MIMO"),
        (4, 4, "4x4 MIMO"),
    ]
    
    print("Testing different MIMO antenna configurations:\n")
    
    for num_tx, num_rx, description in mimo_configs:
        # Create modified config
        test_config = config.copy()
        test_config['mimo'] = {'num_tx_antennas': num_tx, 'num_rx_antennas': num_rx}
        
        output = simulate_transmission(
            config=test_config,
            channel_type='EVA',
            snr_db=15,
            doppler_hz=50,
            pilot_density=0.1
        )
        
        H_shape = output['channel'].shape
        expected_mimo_shape = (num_rx, num_tx)
        
        print(f"  {description}:")
        print(f"    Channel shape: {H_shape}")
        print(f"    MIMO dimensions in channel: ({H_shape[1]}, {H_shape[2]})")
        print(f"    Expected: {expected_mimo_shape}")
        assert H_shape[1] == num_rx and H_shape[2] == num_tx, "MIMO dimension mismatch!"
        print(f"    ✓ Correct\n")
    
    print("✓ MIMO configurations test complete\n")

if __name__ == '__main__':
    try:
        # Test OFDM transmission
        success = test_ofdm_transmission()
        
        # Test MIMO configurations
        test_mimo_dimensions()
        
        if success:
            print("\n" + "🎉 "*20)
            print("PHASE 1.3 COMPLETED SUCCESSFULLY!")
            print("Next: Run visualize_channel_phase1.py")
            print("🎉 "*20 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
