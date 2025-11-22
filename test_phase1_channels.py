"""
Phase 1 - Test Script: Channel Models Validation
Tests EPA, EVA, ETU channel models with various configurations
"""

import numpy as np
import yaml
from src.channel_simulator import ChannelModel, OFDMConfig

def load_config():
    """Load experiment configuration"""
    with open('configs/experiment_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_channel_models():
    """Test all three channel models"""
    print("\n" + "="*70)
    print("PHASE 1 - CHANNEL MODELS VALIDATION")
    print("="*70 + "\n")
    
    config = load_config()
    
    # Test parameters
    num_tx = config['mimo']['num_tx_antennas']
    num_rx = config['mimo']['num_rx_antennas']
    doppler_hz = 50  # Moderate Doppler
    # Calculate sampling rate from FFT size and subcarrier spacing
    sampling_rate = config['ofdm']['fft_size'] * config['ofdm']['subcarrier_spacing']
    carrier_freq = float(config['channel']['carrier_freq'])
    
    # Time samples for channel generation
    symbol_duration = config['ofdm']['fft_size'] / sampling_rate
    num_samples = int(config['ofdm']['num_symbols'] * symbol_duration * sampling_rate)
    
    channel_types = ['EPA', 'EVA', 'ETU']
    
    print(f"Test Configuration:")
    print(f"  MIMO: {num_tx}x{num_rx}")
    print(f"  Time Samples: {num_samples}")
    print(f"  Doppler: {doppler_hz} Hz")
    print(f"  Sampling Rate: {sampling_rate/1e6:.2f} MHz")
    print(f"  Carrier Frequency: {carrier_freq/1e9:.2f} GHz\n")
    
    results = {}
    
    for channel_type in channel_types:
        print(f"\n[{channel_type}] Testing {channel_type} Channel Model...")
        print("-" * 70)
        
        try:
            # Create channel model
            channel_model = ChannelModel(
                model_type=channel_type,
                doppler_hz=doppler_hz,
                carrier_freq=carrier_freq,
                sampling_rate=sampling_rate
            )
            
            # Generate time-domain channel
            H = channel_model.generate_time_varying_channel(
                num_samples=num_samples,
                num_tx=num_tx,
                num_rx=num_rx
            )
            
            # Validate shape
            expected_shape = (num_samples, num_rx, num_tx, H.shape[-1])  # Last dim is max_delay+1
            assert H.shape == expected_shape, f"Shape mismatch: {H.shape} != {expected_shape}"
            
            # Calculate statistics (average over all dimensions except time)
            h_power = np.mean(np.abs(H)**2, axis=(1, 2, 3))  # Power over time
            mean_power = np.mean(h_power)
            mean_magnitude = np.mean(np.abs(H))
            temporal_variation = np.std(h_power)
            
            print(f"  ✓ Shape: {H.shape}")
            print(f"  ✓ Mean Power: {mean_power:.6f}")
            print(f"  ✓ Mean Magnitude: {mean_magnitude:.6f}")
            print(f"  ✓ Temporal Variation: {temporal_variation:.6f}")
            print(f"  ✓ Number of Paths: {channel_model.num_paths}")
            
            # Check for NaN or Inf
            assert not np.isnan(H).any(), "NaN values detected!"
            assert not np.isinf(H).any(), "Inf values detected!"
            print(f"  ✓ No NaN/Inf values")
            
            # Check channel activity (should have non-zero values)
            assert np.abs(H).max() > 0, "Channel has no energy!"
            print(f"  ✓ Max magnitude: {np.abs(H).max():.6f}")
            
            # Store results
            results[channel_type] = {
                'shape': H.shape,
                'mean_power': mean_power,
                'mean_magnitude': mean_magnitude,
                'temporal_variation': temporal_variation,
                'num_paths': channel_model.num_paths,
                'status': 'PASS'
            }
            
            print(f"  ✅ {channel_type} channel test PASSED\n")
            
        except Exception as e:
            print(f"  ❌ {channel_type} channel test FAILED: {str(e)}\n")
            results[channel_type] = {'status': 'FAIL', 'error': str(e)}
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Channel':<10} {'Status':<10} {'Mean Power':<15} {'Temp. Variation':<20}")
    print("-" * 70)
    
    for channel_type, res in results.items():
        status = res['status']
        if status == 'PASS':
            print(f"{channel_type:<10} {'✅ PASS':<10} {res['mean_power']:<15.4f} {res['temporal_variation']:<20.4f}")
        else:
            print(f"{channel_type:<10} {'❌ FAIL':<10} {res.get('error', 'Unknown error')}")
    
    # Overall result
    all_passed = all(r['status'] == 'PASS' for r in results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL CHANNEL MODELS VALIDATED SUCCESSFULLY!")
        print("="*70)
        print("\n✓ Phase 1.2 Complete - Proceed to Task 1.3\n")
        return True
    else:
        print("❌ SOME CHANNEL MODELS FAILED - REVIEW ERRORS ABOVE")
        print("="*70)
        return False

def test_doppler_effects():
    """Test channel behavior at different Doppler frequencies"""
    print("\n" + "="*70)
    print("TESTING DOPPLER EFFECTS")
    print("="*70 + "\n")
    
    config = load_config()
    num_tx = config['mimo']['num_tx_antennas']
    num_rx = config['mimo']['num_rx_antennas']
    sampling_rate = config['ofdm']['fft_size'] * config['ofdm']['subcarrier_spacing']
    carrier_freq = float(config['channel']['carrier_freq'])
    
    # Shorter test duration
    num_samples = 500
    
    doppler_frequencies = [10, 50, 100, 200]  # Low to high mobility
    
    print(f"Testing EVA channel with varying Doppler frequencies:")
    print(f"  10 Hz  ≈ 10 km/h (pedestrian)")
    print(f"  50 Hz  ≈ 60 km/h (urban vehicle)")
    print(f"  100 Hz ≈ 120 km/h (highway)")
    print(f"  200 Hz ≈ 240 km/h (high-speed train)\n")
    
    for doppler_hz in doppler_frequencies:
        channel_model = ChannelModel(
            model_type='EVA',
            doppler_hz=doppler_hz,
            carrier_freq=carrier_freq,
            sampling_rate=sampling_rate
        )
        
        H = channel_model.generate_time_varying_channel(
            num_samples=num_samples,
            num_tx=num_tx,
            num_rx=num_rx
        )
        
        # Measure temporal variation
        h_power = np.mean(np.abs(H)**2, axis=(1, 2, 3))
        temporal_std = np.std(h_power)
        
        print(f"  Doppler {doppler_hz:3d} Hz: Temporal Variation = {temporal_std:.6f}")
    
    print("\n✓ Doppler effects test complete\n")

if __name__ == '__main__':
    try:
        # Test channel models
        success = test_channel_models()
        
        # Test Doppler effects
        test_doppler_effects()
        
        if success:
            print("\n" + "🎉 "*20)
            print("PHASE 1.2 COMPLETED SUCCESSFULLY!")
            print("Next: Run test_phase1_transmission.py")
            print("🎉 "*20 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
