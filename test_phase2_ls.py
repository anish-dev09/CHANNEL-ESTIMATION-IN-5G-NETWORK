"""
Phase 2 - Test Script: LS Estimator Validation
Tests Least Squares channel estimation with different configurations
"""

import numpy as np
import yaml
from src.channel_simulator import simulate_transmission
from src.baseline_estimators import LSEstimator
from src.utils import calculate_nmse, linear2db

def load_config():
    """Load experiment configuration"""
    with open('configs/experiment_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def test_ls_estimator():
    """Test LS estimator with various configurations"""
    print("\n" + "="*70)
    print("PHASE 2 - LS ESTIMATOR VALIDATION")
    print("="*70 + "\n")
    
    config = load_config()
    
    # Test configurations
    test_cases = [
        {'snr': 20, 'pilot_density': 0.10, 'interpolation': 'linear', 'scenario': 'High SNR, 10% pilots, linear'},
        {'snr': 15, 'pilot_density': 0.10, 'interpolation': 'cubic', 'scenario': 'Medium SNR, 10% pilots, cubic'},
        {'snr': 10, 'pilot_density': 0.05, 'interpolation': 'linear', 'scenario': 'Low SNR, 5% pilots, linear'},
    ]
    
    results = []
    
    for idx, test in enumerate(test_cases, 1):
        print(f"\n[Test {idx}/3] {test['scenario']}")
        print("-" * 70)
        
        try:
            # Simulate transmission
            output = simulate_transmission(
                config=config,
                channel_type='EVA',
                snr_db=test['snr'],
                doppler_hz=50,
                pilot_density=test['pilot_density']
            )
            
            rx_symbols = output['rx_symbols']
            tx_symbols = output['tx_symbols']
            H_true = output['channel']
            pilot_pattern = output['pilot_pattern']
            
            print(f"  Configuration:")
            print(f"    SNR: {test['snr']} dB")
            print(f"    Pilot Density: {test['pilot_density']*100:.1f}%")
            print(f"    Interpolation: {test['interpolation']}")
            
            # Create LS estimator
            estimator = LSEstimator(interpolation_method=test['interpolation'])
            
            # Get pilot positions and symbols
            pilot_mask = pilot_pattern.pilot_mask
            pilot_positions = np.where(pilot_mask)
            tx_pilots = output['pilot_symbols']
            
            # Reshape rx_symbols to expected 4D format if needed
            if rx_symbols.ndim == 3:  # (num_symbols, num_rx, num_subcarriers)
                num_symbols, num_rx, num_subcarriers = rx_symbols.shape
                num_tx = tx_symbols.shape[1]
                # Create 4D array by expanding TX dimension
                rx_symbols_4d = np.zeros((num_symbols, num_rx, num_tx, num_subcarriers), dtype=complex)
                for tx_idx in range(num_tx):
                    rx_symbols_4d[:, :, tx_idx, :] = rx_symbols
                rx_symbols = rx_symbols_4d
            
            # Estimate channel
            H_est_full = estimator.estimate(
                rx_symbols=rx_symbols,
                tx_pilots=tx_pilots,
                pilot_mask=pilot_mask,
                pilot_positions=pilot_positions
            )
            
            # Extract first RX-TX pair for comparison
            H_est = H_est_full[:, 0, 0, :]
            
            # Calculate NMSE
            H_true_slice = H_true[:, 0, 0, :]  # First RX-TX pair
            nmse = calculate_nmse(H_true_slice, H_est)
            nmse_db = linear2db(nmse)
            
            print(f"\n  Results:")
            print(f"    H_est shape: {H_est.shape}")
            print(f"    H_true shape: {H_true_slice.shape}")
            print(f"    NMSE: {nmse_db:.2f} dB")
            print(f"    Mean |H_true|: {np.abs(H_true_slice).mean():.4f}")
            print(f"    Mean |H_est|: {np.abs(H_est).mean():.4f}")
            
            # Validation
            assert H_est.shape == H_true_slice.shape, "Shape mismatch!"
            assert not np.isnan(H_est).any(), "NaN in estimate!"
            assert not np.isinf(H_est).any(), "Inf in estimate!"
            
            # NMSE should be reasonable (< 10 dB for these SNRs)
            if nmse_db > 10:
                print(f"    ⚠️  Warning: High NMSE ({nmse_db:.2f} dB)")
            else:
                print(f"    ✓ NMSE within acceptable range")
            
            results.append({
                'test': test['scenario'],
                'snr': test['snr'],
                'pilot_density': test['pilot_density'],
                'interpolation': test['interpolation'],
                'nmse_db': nmse_db,
                'status': 'PASS'
            })
            
            print(f"\n  ✅ Test {idx} PASSED\n")
            
        except Exception as e:
            print(f"\n  ❌ Test {idx} FAILED: {str(e)}\n")
            results.append({
                'test': test['scenario'],
                'status': 'FAIL',
                'error': str(e)
            })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Test Case':<45} {'SNR (dB)':<10} {'NMSE (dB)':<12} {'Status':<10}")
    print("-" * 70)
    
    for res in results:
        if res['status'] == 'PASS':
            print(f"{res['test']:<45} {res['snr']:<10} {res['nmse_db']:<12.2f} {'✅ PASS':<10}")
        else:
            print(f"{res['test']:<45} {'N/A':<10} {'N/A':<12} {'❌ FAIL':<10}")
    
    # Overall result
    all_passed = all(r['status'] == 'PASS' for r in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✅ ALL LS ESTIMATOR TESTS PASSED!")
        print("="*70)
        print("\n✓ Phase 2.1 Complete - Proceed to Task 2.2 (MMSE Test)\n")
        return True, results
    else:
        print("❌ SOME LS TESTS FAILED - REVIEW ERRORS ABOVE")
        print("="*70)
        return False, results

def test_interpolation_methods():
    """Compare different interpolation methods"""
    print("\n" + "="*70)
    print("TESTING INTERPOLATION METHODS")
    print("="*70 + "\n")
    
    config = load_config()
    
    # Single test configuration
    output = simulate_transmission(
        config=config,
        channel_type='EVA',
        snr_db=15,
        doppler_hz=50,
        pilot_density=0.10
    )
    
    rx_symbols = output['rx_symbols']
    tx_symbols = output['tx_symbols']
    H_true = output['channel'][:, 0, 0, :]
    pilot_pattern = output['pilot_pattern']
    
    interpolation_methods = ['linear', 'cubic', 'nearest']
    
    # Get pilot info
    pilot_mask = pilot_pattern.pilot_mask
    pilot_positions = np.where(pilot_mask)
    tx_pilots = output['pilot_symbols']
    
    # Prepare 4D rx_symbols
    if rx_symbols.ndim == 3:
        num_symbols, num_rx, num_subcarriers = rx_symbols.shape
        num_tx = tx_symbols.shape[1]
        rx_symbols_4d = np.zeros((num_symbols, num_rx, num_tx, num_subcarriers), dtype=complex)
        for tx_idx in range(num_tx):
            rx_symbols_4d[:, :, tx_idx, :] = rx_symbols
        rx_symbols = rx_symbols_4d
    
    print("Comparing interpolation methods on same data:\n")
    
    for method in interpolation_methods:
        estimator = LSEstimator(interpolation_method=method)
        H_est_full = estimator.estimate(rx_symbols, tx_pilots, pilot_mask, pilot_positions)
        H_est = H_est_full[:, 0, 0, :]
        
        nmse = calculate_nmse(H_true, H_est)
        nmse_db = linear2db(nmse)
        
        print(f"  {method.capitalize():<10}: NMSE = {nmse_db:>7.2f} dB")
    
    print("\n✓ Interpolation comparison complete\n")

if __name__ == '__main__':
    try:
        # Test LS estimator
        success, results = test_ls_estimator()
        
        # Compare interpolation methods
        test_interpolation_methods()
        
        if success:
            print("\n" + "🎉 "*20)
            print("PHASE 2.1 COMPLETED SUCCESSFULLY!")
            print("Next: Run test_phase2_mmse.py")
            print("🎉 "*20 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
