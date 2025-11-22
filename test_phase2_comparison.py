"""
Phase 2 - Test Script: LS vs MMSE Comparison
Direct comparison of baseline estimators
"""

import numpy as np
import yaml
import time
from src.channel_simulator import simulate_transmission
from src.baseline_estimators import LSEstimator, MMSEEstimator
from src.utils import calculate_nmse, linear2db

def load_config():
    """Load experiment configuration"""
    with open('configs/experiment_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def compare_estimators():
    """Compare LS and MMSE estimators"""
    print("\n" + "="*70)
    print("PHASE 2 - LS VS MMSE COMPARISON")
    print("="*70 + "\n")
    
    config = load_config()
    
    # Test at different SNR levels
    snr_levels = [5, 10, 15, 20, 25]
    pilot_density = 0.10
    
    print(f"Configuration:")
    print(f"  Channel: EVA")
    print(f"  Doppler: 50 Hz")
    print(f"  Pilot Density: {pilot_density*100:.1f}%")
    print(f"  Interpolation: cubic (both methods)\n")
    
    ls_results = []
    mmse_results = []
    
    for snr in snr_levels:
        print(f"\n{'='*70}")
        print(f"SNR = {snr} dB")
        print(f"{'='*70}\n")
        
        # Simulate transmission
        output = simulate_transmission(
            config=config,
            channel_type='EVA',
            snr_db=snr,
            doppler_hz=50,
            pilot_density=pilot_density
        )
        
        rx_symbols = output['rx_symbols']
        tx_symbols = output['tx_symbols']
        H_true = output['channel'][:, 0, 0, :]
        pilot_pattern = output['pilot_pattern']
        
        # Get pilot info
        pilot_mask = pilot_pattern.pilot_mask
        pilot_positions = np.where(pilot_mask)
        tx_pilots = output['pilot_symbols']
        
        # Reshape rx_symbols to 4D if needed
        if rx_symbols.ndim == 3:
            num_symbols, num_rx, num_subcarriers = rx_symbols.shape
            num_tx = tx_symbols.shape[1]
            rx_symbols_4d = np.zeros((num_symbols, num_rx, num_tx, num_subcarriers), dtype=complex)
            for tx_idx in range(num_tx):
                rx_symbols_4d[:, :, tx_idx, :] = rx_symbols
            rx_symbols = rx_symbols_4d
        
        # LS Estimation
        print("  [LS Estimator]")
        ls_estimator = LSEstimator(interpolation_method='cubic')
        
        start_time = time.time()
        H_ls_full = ls_estimator.estimate(rx_symbols, tx_pilots, pilot_mask, pilot_positions)
        ls_time = (time.time() - start_time) * 1000  # Convert to ms
        H_ls = H_ls_full[:, 0, 0, :]
        
        nmse_ls = calculate_nmse(H_true, H_ls)
        nmse_ls_db = linear2db(nmse_ls)
        
        print(f"    NMSE: {nmse_ls_db:>7.2f} dB")
        print(f"    Time: {ls_time:>7.2f} ms")
        
        ls_results.append({
            'snr': snr,
            'nmse_db': nmse_ls_db,
            'time_ms': ls_time
        })
        
        # MMSE Estimation
        print("\n  [MMSE Estimator]")
        mmse_estimator = MMSEEstimator()
        
        start_time = time.time()
        H_mmse_full = mmse_estimator.estimate(rx_symbols, tx_pilots, pilot_mask, pilot_positions, snr_db=snr)
        mmse_time = (time.time() - start_time) * 1000  # Convert to ms
        H_mmse = H_mmse_full[:, 0, 0, :]
        
        nmse_mmse = calculate_nmse(H_true, H_mmse)
        nmse_mmse_db = linear2db(nmse_mmse)
        
        print(f"    NMSE: {nmse_mmse_db:>7.2f} dB")
        print(f"    Time: {mmse_time:>7.2f} ms")
        
        mmse_results.append({
            'snr': snr,
            'nmse_db': nmse_mmse_db,
            'time_ms': mmse_time
        })
        
        # Improvement
        improvement = nmse_ls_db - nmse_mmse_db
        print(f"\n  [Comparison]")
        print(f"    MMSE Improvement: {improvement:>7.2f} dB")
        print(f"    Time Ratio (MMSE/LS): {mmse_time/ls_time:>7.2f}x")
    
    # Final Summary Table
    print("\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"\n{'SNR (dB)':<10} {'LS NMSE (dB)':<15} {'MMSE NMSE (dB)':<17} {'Improvement (dB)':<18} {'Winner':<10}")
    print("-" * 70)
    
    for ls_res, mmse_res in zip(ls_results, mmse_results):
        improvement = ls_res['nmse_db'] - mmse_res['nmse_db']
        winner = "MMSE ✓" if improvement > 0 else "LS ✓" if improvement < 0 else "Tie"
        
        print(f"{ls_res['snr']:<10} {ls_res['nmse_db']:<15.2f} {mmse_res['nmse_db']:<17.2f} {improvement:<18.2f} {winner:<10}")
    
    # Performance Summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    
    avg_ls_nmse = np.mean([r['nmse_db'] for r in ls_results])
    avg_mmse_nmse = np.mean([r['nmse_db'] for r in mmse_results])
    avg_improvement = avg_ls_nmse - avg_mmse_nmse
    
    avg_ls_time = np.mean([r['time_ms'] for r in ls_results])
    avg_mmse_time = np.mean([r['time_ms'] for r in mmse_results])
    
    print(f"\nAverage NMSE:")
    print(f"  LS:   {avg_ls_nmse:.2f} dB")
    print(f"  MMSE: {avg_mmse_nmse:.2f} dB")
    print(f"  MMSE Advantage: {avg_improvement:.2f} dB")
    
    print(f"\nAverage Execution Time:")
    print(f"  LS:   {avg_ls_time:.2f} ms")
    print(f"  MMSE: {avg_mmse_time:.2f} ms")
    print(f"  Time Overhead: {((avg_mmse_time/avg_ls_time - 1) * 100):.1f}%")
    
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE!")
    print("="*70)
    print("\n✓ Phase 2.3 Complete - Proceed to Task 2.4\n")
    
    return ls_results, mmse_results

if __name__ == '__main__':
    try:
        ls_results, mmse_results = compare_estimators()
        
        print("\n" + "🎉 "*20)
        print("PHASE 2.3 COMPLETED SUCCESSFULLY!")
        print("Next: Run test_phase2_snr_sweep.py")
        print("🎉 "*20 + "\n")
        
    except Exception as e:
        print(f"\n❌ Comparison failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
