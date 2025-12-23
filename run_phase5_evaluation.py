"""
Phase 5: Model Evaluation and Comparison Script
=================================================

Comprehensive evaluation of trained models vs baseline estimators.
Generates detailed metrics, plots, and comparison reports.

Usage:
    python run_phase5_evaluation.py --model cnn                    # Evaluate CNN model
    python run_phase5_evaluation.py --all                          # Evaluate all models
    python run_phase5_evaluation.py --model cnn --snr-sweep        # Per-SNR analysis
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

from ai_models import get_model
from baseline_estimators import LSEstimator, MMSEEstimator
from utils import load_config, linear2db


def compute_nmse(H_est: np.ndarray, H_true: np.ndarray) -> float:
    """
    Compute Normalized Mean Squared Error.
    
    NMSE = E[|H_est - H_true|²] / E[|H_true|²]
    """
    error = H_est - H_true
    mse = np.mean(np.abs(error) ** 2)
    power = np.mean(np.abs(H_true) ** 2)
    return mse / (power + 1e-10)


def compute_mse(H_est: np.ndarray, H_true: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    error = H_est - H_true
    return np.mean(np.abs(error) ** 2)


def compute_mae(H_est: np.ndarray, H_true: np.ndarray) -> float:
    """Compute Mean Absolute Error."""
    error = H_est - H_true
    return np.mean(np.abs(error))


def compute_ber_approximation(H_est: np.ndarray, H_true: np.ndarray, snr_db: float) -> float:
    """
    Approximate BER based on channel estimation error.
    Uses the relationship between estimation error and BER degradation.
    """
    nmse = compute_nmse(H_est, H_true)
    snr_linear = 10 ** (snr_db / 10)
    # Approximate BER for QPSK with channel estimation error
    effective_snr = snr_linear / (1 + snr_linear * nmse)
    # Q-function approximation
    ber = 0.5 * np.exp(-effective_snr / 2)
    return float(np.clip(ber, 1e-10, 0.5))


class ModelEvaluator:
    """Evaluate trained models and compare with baselines."""
    
    def __init__(self, model_path: str = 'models', data_path: str = 'data',
                 results_path: str = 'results', device: str = None):
        """Initialize evaluator."""
        self.model_path = Path(model_path)
        self.data_path = Path(data_path)
        self.results_path = Path(results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {self.device}")
        
        # Load test data
        self.test_data = None
        self._load_test_data()
    
    def _load_test_data(self):
        """Load test dataset."""
        test_path = self.data_path / 'test.npz'
        if test_path.exists():
            print(f"Loading test data from {test_path}...")
            self.test_data = dict(np.load(test_path, allow_pickle=True))
            print(f"  Loaded {self.test_data['rx_symbols'].shape[0]} test samples")
        else:
            print(f"Warning: Test data not found at {test_path}")
    
    def load_model(self, model_type: str) -> torch.nn.Module:
        """Load a trained model."""
        model_file = self.model_path / f'{model_type}_best.pth'
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_file}")
        
        print(f"Loading model from {model_file}...")
        checkpoint = torch.load(model_file, map_location=self.device, weights_only=False)
        
        # Get model configuration
        config = checkpoint.get('config', {})
        model = get_model(model_type, config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"  Loaded {model_type} model (epoch {checkpoint.get('epoch', 'N/A')}, "
              f"val_loss: {checkpoint.get('val_loss', 'N/A'):.4f})")
        
        return model
    
    def prepare_inputs(self, idx: int) -> tuple:
        """Prepare input tensors for model."""
        # Get single sample
        rx = self.test_data['rx_symbols'][idx, :, 0, :]  # (14, 599)
        H_ls = self.test_data['H_ls'][idx, :, 0, 0, :]   # (14, 599)
        H_true = self.test_data['H_true'][idx, :, 0, 0, :]  # (14, 599)
        pilot_mask = self.test_data['pilot_mask'][idx]  # (14, 599)
        
        # Convert to real representation
        rx_real = np.stack([rx.real, rx.imag], axis=0)
        H_ls_real = np.stack([H_ls.real, H_ls.imag], axis=0)
        
        # Normalize (use std of test data)
        rx_std = np.std(np.abs(self.test_data['rx_symbols'])) + 1e-8
        H_ls_std = np.std(np.abs(self.test_data['H_ls'])) + 1e-8
        
        rx_real = rx_real / rx_std
        H_ls_real = H_ls_real / H_ls_std
        
        # Combine inputs
        inputs = np.concatenate([
            rx_real,
            H_ls_real,
            pilot_mask[np.newaxis, :, :]
        ], axis=0).astype(np.float32)
        
        # Add batch dimension
        inputs = torch.from_numpy(inputs).unsqueeze(0).to(self.device)
        
        return inputs, H_true, H_ls
    
    def model_predict(self, model: torch.nn.Module, inputs: torch.Tensor) -> np.ndarray:
        """Get model prediction."""
        with torch.no_grad():
            output = model(inputs)
            output = output.cpu().numpy()[0]  # (2, 14, 599)
        
        # Convert back to complex
        H_est = output[0] + 1j * output[1]  # (14, 599)
        
        # Denormalize
        H_true_std = np.std(np.abs(self.test_data['H_true'][:, :, 0, 0, :])) + 1e-8
        H_est = H_est * H_true_std
        
        return H_est
    
    def evaluate_model(self, model_type: str, num_samples: int = None) -> dict:
        """Evaluate a single model."""
        if self.test_data is None:
            raise ValueError("Test data not loaded")
        
        model = self.load_model(model_type)
        
        if num_samples is None:
            num_samples = len(self.test_data['rx_symbols'])
        
        results = {
            'model': model_type,
            'nmse_values': [],
            'mse_values': [],
            'mae_values': [],
            'snr_db': [],
            'channel_type': [],
        }
        
        print(f"\nEvaluating {model_type} on {num_samples} samples...")
        
        for idx in tqdm(range(num_samples)):
            inputs, H_true, H_ls = self.prepare_inputs(idx)
            H_est = self.model_predict(model, inputs)
            
            nmse = compute_nmse(H_est, H_true)
            mse = compute_mse(H_est, H_true)
            mae = compute_mae(H_est, H_true)
            
            results['nmse_values'].append(nmse)
            results['mse_values'].append(mse)
            results['mae_values'].append(mae)
            results['snr_db'].append(float(self.test_data['snr_db'][idx]))
            results['channel_type'].append(str(self.test_data['channel_type'][idx]))
        
        # Compute aggregate metrics
        results['nmse_mean'] = float(np.mean(results['nmse_values']))
        results['nmse_std'] = float(np.std(results['nmse_values']))
        results['nmse_db'] = linear2db(results['nmse_mean'])
        results['mse_mean'] = float(np.mean(results['mse_values']))
        results['mae_mean'] = float(np.mean(results['mae_values']))
        
        print(f"  NMSE: {results['nmse_db']:.2f} dB")
        print(f"  MSE:  {results['mse_mean']:.6f}")
        print(f"  MAE:  {results['mae_mean']:.6f}")
        
        return results
    
    def evaluate_baselines(self, num_samples: int = None) -> dict:
        """Evaluate LS and MMSE baselines."""
        if self.test_data is None:
            raise ValueError("Test data not loaded")
        
        if num_samples is None:
            num_samples = len(self.test_data['rx_symbols'])
        
        results = {
            'LS': {'nmse_values': [], 'snr_db': []},
            'MMSE': {'nmse_values': [], 'snr_db': []},
        }
        
        print(f"\nEvaluating baselines on {num_samples} samples...")
        
        for idx in tqdm(range(num_samples)):
            H_true = self.test_data['H_true'][idx, :, 0, 0, :]  # (14, 599)
            H_ls = self.test_data['H_ls'][idx, :, 0, 0, :]      # (14, 599)
            snr = float(self.test_data['snr_db'][idx])
            
            # LS results (already computed in dataset)
            ls_nmse = compute_nmse(H_ls, H_true)
            results['LS']['nmse_values'].append(ls_nmse)
            results['LS']['snr_db'].append(snr)
            
            # MMSE - simplified version using LS + noise variance
            snr_linear = 10 ** (snr / 10)
            noise_var = 1 / snr_linear
            # Wiener filter approximation
            alpha = 1 / (1 + noise_var)
            H_mmse = alpha * H_ls
            mmse_nmse = compute_nmse(H_mmse, H_true)
            results['MMSE']['nmse_values'].append(mmse_nmse)
            results['MMSE']['snr_db'].append(snr)
        
        # Aggregate metrics
        for method in ['LS', 'MMSE']:
            results[method]['nmse_mean'] = float(np.mean(results[method]['nmse_values']))
            results[method]['nmse_std'] = float(np.std(results[method]['nmse_values']))
            results[method]['nmse_db'] = linear2db(results[method]['nmse_mean'])
            print(f"  {method} NMSE: {results[method]['nmse_db']:.2f} dB")
        
        return results
    
    def snr_sweep_analysis(self, model_type: str = None) -> dict:
        """Analyze performance across SNR values."""
        if self.test_data is None:
            raise ValueError("Test data not loaded")
        
        snr_values = sorted(list(set(self.test_data['snr_db'])))
        print(f"\nSNR sweep analysis for values: {snr_values}")
        
        results = {snr: {'LS': [], 'MMSE': []} for snr in snr_values}
        
        if model_type:
            model = self.load_model(model_type)
            for snr in snr_values:
                results[snr][model_type] = []
        
        num_samples = len(self.test_data['rx_symbols'])
        
        for idx in tqdm(range(num_samples), desc="SNR sweep"):
            snr = float(self.test_data['snr_db'][idx])
            H_true = self.test_data['H_true'][idx, :, 0, 0, :]
            H_ls = self.test_data['H_ls'][idx, :, 0, 0, :]
            
            # LS
            results[snr]['LS'].append(compute_nmse(H_ls, H_true))
            
            # MMSE
            snr_linear = 10 ** (snr / 10)
            alpha = 1 / (1 + 1/snr_linear)
            H_mmse = alpha * H_ls
            results[snr]['MMSE'].append(compute_nmse(H_mmse, H_true))
            
            # AI model
            if model_type:
                inputs, _, _ = self.prepare_inputs(idx)
                H_est = self.model_predict(model, inputs)
                results[snr][model_type].append(compute_nmse(H_est, H_true))
        
        # Aggregate
        summary = {'snr_db': snr_values, 'methods': {}}
        methods = ['LS', 'MMSE']
        if model_type:
            methods.append(model_type)
        
        for method in methods:
            summary['methods'][method] = {
                'nmse_db': [linear2db(np.mean(results[snr][method])) for snr in snr_values]
            }
        
        return summary
    
    def plot_snr_comparison(self, snr_results: dict, save_path: str = None):
        """Plot NMSE vs SNR comparison."""
        plt.figure(figsize=(10, 6))
        
        snr_values = snr_results['snr_db']
        markers = {'LS': 'o', 'MMSE': 's', 'cnn': '^', 'lstm': 'v', 'hybrid': 'd', 'resnet': 'p'}
        colors = {'LS': 'blue', 'MMSE': 'green', 'cnn': 'red', 'lstm': 'purple', 
                  'hybrid': 'orange', 'resnet': 'brown'}
        
        for method, data in snr_results['methods'].items():
            plt.plot(snr_values, data['nmse_db'], 
                    marker=markers.get(method, 'o'),
                    color=colors.get(method, 'gray'),
                    label=method.upper(), linewidth=2, markersize=8)
        
        plt.xlabel('SNR (dB)', fontsize=12)
        plt.ylabel('NMSE (dB)', fontsize=12)
        plt.title('Channel Estimation Performance vs SNR', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved plot to {save_path}")
        
        plt.close()
    
    def generate_report(self, all_results: dict, save_path: str = None) -> str:
        """Generate evaluation report."""
        report_lines = [
            "=" * 60,
            "PHASE 5: CHANNEL ESTIMATION EVALUATION REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            "## Performance Summary (NMSE in dB)",
            ""
        ]
        
        # Create comparison table
        report_lines.append("| Method     | NMSE (dB) | MSE      | Improvement vs LS |")
        report_lines.append("|------------|-----------|----------|-------------------|")
        
        ls_nmse = all_results.get('LS', {}).get('nmse_db', 0)
        
        for method, data in all_results.items():
            nmse_db = data.get('nmse_db', 0)
            mse = data.get('mse_mean', 0)
            improvement = ls_nmse - nmse_db if ls_nmse else 0
            report_lines.append(f"| {method:10s} | {nmse_db:9.2f} | {mse:8.6f} | {improvement:+17.2f} |")
        
        report_lines.extend([
            "",
            "## Test Configuration",
            f"- Test samples: {len(self.test_data['rx_symbols']) if self.test_data else 'N/A'}",
            f"- Device: {self.device}",
            "",
            "## Notes",
            "- NMSE = E[|H_est - H_true|²] / E[|H_true|²]",
            "- Lower NMSE (more negative dB) is better",
            "- Improvement is relative to LS baseline",
            ""
        ])
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"✓ Saved report to {save_path}")
        
        return report


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Phase 5: Model Evaluation and Comparison'
    )
    
    parser.add_argument('--model', type=str, default=None,
                       choices=['cnn', 'lstm', 'hybrid', 'resnet'],
                       help='Specific model to evaluate')
    parser.add_argument('--all', action='store_true',
                       help='Evaluate all available models')
    parser.add_argument('--baselines-only', action='store_true',
                       help='Only evaluate LS and MMSE baselines')
    parser.add_argument('--snr-sweep', action='store_true',
                       help='Perform SNR sweep analysis')
    parser.add_argument('--num-samples', type=int, default=None,
                       help='Number of test samples (default: all)')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Directory containing trained models')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing test data')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Directory for results output')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("   PHASE 5: MODEL EVALUATION AND COMPARISON")
    print("="*60)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_dir,
        data_path=args.data_dir,
        results_path=args.results_dir
    )
    
    all_results = {}
    
    # Evaluate baselines
    baseline_results = evaluator.evaluate_baselines(args.num_samples)
    all_results.update(baseline_results)
    
    # Evaluate models
    if not args.baselines_only:
        models_to_eval = []
        
        if args.all:
            # Find all available models
            for model_type in ['cnn', 'lstm', 'hybrid', 'resnet']:
                model_file = Path(args.model_dir) / f'{model_type}_best.pth'
                if model_file.exists():
                    models_to_eval.append(model_type)
        elif args.model:
            models_to_eval = [args.model]
        
        for model_type in models_to_eval:
            try:
                results = evaluator.evaluate_model(model_type, args.num_samples)
                all_results[model_type] = results
            except FileNotFoundError as e:
                print(f"Warning: {e}")
    
    # SNR sweep analysis
    if args.snr_sweep:
        model_for_sweep = args.model or (models_to_eval[0] if models_to_eval else None)
        snr_results = evaluator.snr_sweep_analysis(model_for_sweep)
        
        # Save plot
        plot_path = Path(args.results_dir) / 'nmse_vs_snr.png'
        evaluator.plot_snr_comparison(snr_results, str(plot_path))
        
        # Save data
        snr_data_path = Path(args.results_dir) / 'snr_sweep_data.json'
        with open(snr_data_path, 'w') as f:
            json.dump(snr_results, f, indent=2)
    
    # Generate and print report
    report_path = Path(args.results_dir) / 'evaluation_report.md'
    report = evaluator.generate_report(all_results, str(report_path))
    print("\n" + report)
    
    # Save all results
    results_path = Path(args.results_dir) / 'evaluation_results.json'
    
    # Clean results for JSON serialization
    json_results = {}
    for method, data in all_results.items():
        json_results[method] = {
            'nmse_db': data.get('nmse_db'),
            'nmse_mean': data.get('nmse_mean'),
            'mse_mean': data.get('mse_mean'),
            'mae_mean': data.get('mae_mean')
        }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"✓ Saved results to {results_path}")
    
    print("\n" + "="*60)
    print("PHASE 5 COMPLETE: Evaluation")
    print("="*60)
    print(f"\nNext step: Run Phase 6 - Advanced Model Training")
    print(f"  python run_phase4_training.py --model lstm --epochs 50")


if __name__ == '__main__':
    main()
