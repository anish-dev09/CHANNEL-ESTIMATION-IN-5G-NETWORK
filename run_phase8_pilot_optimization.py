"""
Phase 8: Pilot Optimization Analysis
=====================================

Analyzes the impact of pilot density and placement on channel estimation.

Usage:
    python run_phase8_pilot_optimization.py                    # Standard analysis
    python run_phase8_pilot_optimization.py --densities 0.05 0.10 0.15 0.20
"""

import sys
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from channel_simulator import simulate_transmission
from baseline_estimators import LSEstimator
from ai_models import get_model
from utils import load_config, set_seed, linear2db


def compute_nmse(H_est: np.ndarray, H_true: np.ndarray) -> float:
    """Compute NMSE."""
    error = H_est - H_true
    mse = np.mean(np.abs(error) ** 2)
    power = np.mean(np.abs(H_true) ** 2)
    return mse / (power + 1e-10)


class PilotOptimizer:
    """Analyze pilot density and placement optimization."""
    
    def __init__(self, config_path: str = 'configs/experiment_config.yaml',
                 model_path: str = 'models', results_path: str = 'results'):
        """Initialize optimizer."""
        self.config = load_config(config_path)
        self.model_path = Path(model_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
    
    def load_model(self, model_type: str = 'cnn') -> torch.nn.Module:
        """Load trained model."""
        model_file = self.model_path / f'{model_type}_best.pth'
        
        if not model_file.exists():
            print(f"Warning: Model {model_file} not found")
            return None
        
        checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
        config = checkpoint.get('config', {})
        model = get_model(model_type, config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def generate_test_sample(self, pilot_density: float, snr_db: float = 15.0,
                            channel_type: str = 'EVA', doppler_hz: float = 50.0) -> dict:
        """Generate a test sample with specific pilot density."""
        sim_data = simulate_transmission(
            self.config,
            channel_type=channel_type,
            doppler_hz=doppler_hz,
            snr_db=snr_db,
            pilot_density=pilot_density
        )
        
        rx_symbols = sim_data['rx_symbols']
        tx_symbols = sim_data['tx_symbols']
        H_true = sim_data['channel']
        pilot_pattern = sim_data['pilot_pattern']
        pilot_symbols = sim_data['pilot_symbols']
        
        # Compute LS estimate
        num_symbols, num_rx, num_sc = rx_symbols.shape
        num_tx = tx_symbols.shape[1]
        
        rx_4d = rx_symbols.reshape(num_symbols, num_rx, 1, num_sc)
        rx_4d = np.repeat(rx_4d, num_tx, axis=2)
        
        ls_estimator = LSEstimator(interpolation_method='linear')
        H_ls = ls_estimator.estimate(
            rx_4d, pilot_symbols,
            pilot_pattern.pilot_mask,
            pilot_pattern.pilot_positions
        )
        
        return {
            'rx_symbols': rx_symbols,
            'H_ls': H_ls,
            'H_true': H_true,
            'pilot_mask': pilot_pattern.pilot_mask,
            'snr_db': snr_db
        }
    
    def analyze_pilot_density(self, pilot_densities: list, 
                             snr_values: list = None,
                             num_samples: int = 100,
                             model_type: str = 'cnn') -> dict:
        """Analyze NMSE vs pilot density."""
        
        if snr_values is None:
            snr_values = [5, 10, 15, 20]
        
        model = self.load_model(model_type)
        
        results = {
            'pilot_densities': pilot_densities,
            'snr_values': snr_values,
            'LS': {},
            model_type: {} if model else None
        }
        
        for snr in snr_values:
            results['LS'][snr] = {d: [] for d in pilot_densities}
            if model:
                results[model_type][snr] = {d: [] for d in pilot_densities}
        
        print(f"\nAnalyzing pilot densities: {pilot_densities}")
        print(f"SNR values: {snr_values}")
        print(f"Samples per configuration: {num_samples}")
        
        total_iter = len(pilot_densities) * len(snr_values) * num_samples
        pbar = tqdm(total=total_iter, desc="Pilot analysis")
        
        for density in pilot_densities:
            for snr in snr_values:
                for _ in range(num_samples):
                    try:
                        sample = self.generate_test_sample(
                            pilot_density=density,
                            snr_db=snr
                        )
                        
                        H_true = sample['H_true'][:, 0, 0, :]
                        H_ls = sample['H_ls'][:, 0, 0, :]
                        
                        # LS NMSE
                        ls_nmse = compute_nmse(H_ls, H_true)
                        results['LS'][snr][density].append(ls_nmse)
                        
                        # Model NMSE (if available)
                        if model:
                            # Prepare input
                            rx = sample['rx_symbols'][:, 0, :]
                            rx_real = np.stack([rx.real, rx.imag], axis=0)
                            H_ls_real = np.stack([H_ls.real, H_ls.imag], axis=0)
                            pilot_mask = sample['pilot_mask']
                            
                            inputs = np.concatenate([
                                rx_real,
                                H_ls_real,
                                pilot_mask[np.newaxis, :, :]
                            ], axis=0).astype(np.float32)
                            
                            inputs = torch.from_numpy(inputs).unsqueeze(0).to(self.device)
                            
                            with torch.no_grad():
                                output = model(inputs).cpu().numpy()[0]
                            
                            H_est = output[0] + 1j * output[1]
                            model_nmse = compute_nmse(H_est, H_true)
                            results[model_type][snr][density].append(model_nmse)
                    
                    except Exception as e:
                        print(f"\nWarning: Sample failed - {e}")
                    
                    pbar.update(1)
        
        pbar.close()
        
        # Aggregate results
        summary = {
            'pilot_densities': pilot_densities,
            'snr_values': snr_values,
            'methods': {}
        }
        
        for method in ['LS', model_type] if model else ['LS']:
            if results.get(method) is None:
                continue
            summary['methods'][method] = {}
            for snr in snr_values:
                summary['methods'][method][snr] = {}
                for density in pilot_densities:
                    vals = results[method][snr][density]
                    if vals:
                        summary['methods'][method][snr][density] = {
                            'nmse_mean': float(np.mean(vals)),
                            'nmse_db': linear2db(np.mean(vals)),
                            'nmse_std': float(np.std(vals))
                        }
        
        return summary
    
    def plot_pilot_analysis(self, results: dict, save_path: str = None):
        """Plot pilot density analysis results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        pilot_densities = results['pilot_densities']
        snr_values = results['snr_values']
        methods = list(results['methods'].keys())
        
        colors = {'LS': 'blue', 'cnn': 'red', 'lstm': 'green', 'hybrid': 'orange'}
        markers = {'LS': 'o', 'cnn': '^', 'lstm': 's', 'hybrid': 'd'}
        
        # Plot 1: NMSE vs Pilot Density for different SNRs
        ax1 = axes[0]
        for method in methods:
            for snr in snr_values:
                nmse_vals = [results['methods'][method][snr].get(d, {}).get('nmse_db', np.nan) 
                            for d in pilot_densities]
                ax1.plot([d*100 for d in pilot_densities], nmse_vals,
                        marker=markers.get(method, 'o'),
                        color=colors.get(method, 'gray'),
                        label=f'{method.upper()} SNR={snr}dB',
                        alpha=0.7)
        
        ax1.set_xlabel('Pilot Density (%)', fontsize=11)
        ax1.set_ylabel('NMSE (dB)', fontsize=11)
        ax1.set_title('NMSE vs Pilot Density', fontsize=12)
        ax1.legend(fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Improvement over LS at different densities
        ax2 = axes[1]
        if len(methods) > 1:
            ai_method = [m for m in methods if m != 'LS'][0]
            for snr in snr_values:
                improvements = []
                for d in pilot_densities:
                    ls_nmse = results['methods']['LS'][snr].get(d, {}).get('nmse_db', 0)
                    ai_nmse = results['methods'][ai_method][snr].get(d, {}).get('nmse_db', 0)
                    improvements.append(ls_nmse - ai_nmse)
                
                ax2.bar([d*100 + snr*0.5 for d in pilot_densities], improvements,
                       width=2, label=f'SNR={snr}dB', alpha=0.7)
        
            ax2.set_xlabel('Pilot Density (%)', fontsize=11)
            ax2.set_ylabel('Improvement over LS (dB)', fontsize=11)
            ax2.set_title(f'{ai_method.upper()} Improvement vs Pilot Density', fontsize=12)
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        plt.close()
    
    def generate_report(self, results: dict) -> str:
        """Generate pilot optimization report."""
        lines = [
            "=" * 60,
            "PHASE 8: PILOT OPTIMIZATION ANALYSIS REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "## Analysis Configuration",
            f"- Pilot Densities: {results['pilot_densities']}",
            f"- SNR Values: {results['snr_values']}",
            "",
            "## Key Findings",
            ""
        ]
        
        # Find optimal density
        for method in results['methods']:
            lines.append(f"\n### {method.upper()}")
            for snr in results['snr_values']:
                best_density = min(
                    results['pilot_densities'],
                    key=lambda d: results['methods'][method][snr].get(d, {}).get('nmse_db', 0)
                )
                best_nmse = results['methods'][method][snr].get(best_density, {}).get('nmse_db', 'N/A')
                lines.append(f"- SNR {snr}dB: Best density = {best_density*100:.0f}% (NMSE = {best_nmse:.2f} dB)")
        
        lines.extend([
            "",
            "## Recommendations",
            "- Higher pilot density generally improves estimation accuracy",
            "- AI methods show greater improvements at lower pilot densities",
            "- Optimal pilot density depends on channel conditions and SNR",
            ""
        ])
        
        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Phase 8: Pilot Optimization Analysis'
    )
    
    parser.add_argument('--densities', type=float, nargs='+',
                       default=[0.05, 0.08, 0.10, 0.12, 0.15],
                       help='Pilot densities to analyze')
    parser.add_argument('--snr', type=float, nargs='+',
                       default=[5, 10, 15, 20],
                       help='SNR values to test')
    parser.add_argument('--samples', type=int, default=50,
                       help='Samples per configuration')
    parser.add_argument('--model', type=str, default='cnn',
                       help='Model to compare with LS')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("   PHASE 8: PILOT OPTIMIZATION ANALYSIS")
    print("="*60)
    
    set_seed(42)
    
    optimizer = PilotOptimizer(results_path=args.results_dir)
    
    # Run analysis
    results = optimizer.analyze_pilot_density(
        pilot_densities=args.densities,
        snr_values=args.snr,
        num_samples=args.samples,
        model_type=args.model
    )
    
    # Save results
    results_path = Path(args.results_dir) / 'pilot_optimization_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ Saved results to {results_path}")
    
    # Plot
    plot_path = Path(args.results_dir) / 'pilot_optimization.png'
    optimizer.plot_pilot_analysis(results, str(plot_path))
    
    # Report
    report = optimizer.generate_report(results)
    print("\n" + report)
    
    report_path = Path(args.results_dir) / 'pilot_optimization_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("PHASE 8 COMPLETE: Pilot Optimization Analysis")
    print("="*60)


if __name__ == '__main__':
    main()
