"""
Evaluation script for trained channel estimation models.

Compares:
- LS baseline
- MMSE baseline  
- AI models (CNN, LSTM, Hybrid)

Metrics:
- MSE, NMSE
- BER after equalization
- Inference latency
- Model size
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import time
from typing import Dict, List

from ai_models import get_model
from baseline_estimators import LSEstimator, MMSEEstimator, evaluate_estimator, equalize_channel
from train import ChannelDataset
from utils import load_config, get_device, load_checkpoint, qam_demodulation, calculate_ber

sns.set_style('whitegrid')


def evaluate_baseline(estimator, test_loader, estimator_type='LS'):
    """Evaluate baseline estimator."""
    all_mse = []
    all_nmse = []
    inference_times = []
    
    print(f"\nEvaluating {estimator_type} Estimator...")
    
    for inputs, targets, pilot_mask in tqdm(test_loader):
        batch_size = inputs.shape[0]
        
        for i in range(batch_size):
            # Reconstruct complex format for baseline
            # Input format: [rx_real, rx_imag, H_ls_real, H_ls_imag, pilot_mask]
            rx_real = inputs[i, 0].numpy()
            rx_imag = inputs[i, 1].numpy()
            rx_complex = rx_real + 1j * rx_imag
            
            H_true_real = targets[i, 0].numpy()
            H_true_imag = targets[i, 1].numpy()
            H_true = H_true_real + 1j * H_true_imag
            
            pilot_mask_np = pilot_mask[i].numpy().astype(bool)
            
            # Create 4D arrays (num_symbols, num_rx=1, num_tx=1, num_sc)
            num_symbols, num_sc = rx_complex.shape
            rx_4d = rx_complex.reshape(num_symbols, 1, 1, num_sc)
            H_true_4d = H_true.reshape(num_symbols, 1, 1, num_sc)
            
            # Extract pilot positions
            pilot_positions = np.where(pilot_mask_np)
            pilot_symbols = np.ones(len(pilot_positions[0]))  # Assume known pilots
            
            # Time estimation
            start_time = time.time()
            
            if estimator_type == 'LS':
                H_est = estimator.estimate(rx_4d, pilot_symbols, pilot_mask_np, pilot_positions)
            else:  # MMSE
                H_est = estimator.estimate(rx_4d, pilot_symbols, pilot_mask_np, pilot_positions, snr_db=10)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Evaluate
            metrics = evaluate_estimator(H_true_4d, H_est)
            all_mse.append(metrics['mse'])
            all_nmse.append(metrics['nmse_db'])
    
    return {
        'mse_mean': np.mean(all_mse),
        'mse_std': np.std(all_mse),
        'nmse_mean': np.mean(all_nmse),
        'nmse_std': np.std(all_nmse),
        'latency_mean_ms': np.mean(inference_times) * 1000,
        'latency_std_ms': np.std(inference_times) * 1000
    }


def evaluate_ai_model(model, test_loader, device):
    """Evaluate AI model."""
    model.eval()
    all_mse = []
    all_nmse = []
    inference_times = []
    
    print(f"\nEvaluating AI Model...")
    
    with torch.no_grad():
        for inputs, targets, pilot_mask in tqdm(test_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Time inference
            start_time = time.time()
            outputs = model(inputs)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            inference_time = time.time() - start_time
            inference_times.append(inference_time / inputs.shape[0])  # Per sample
            
            # Convert to numpy for evaluation
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            for i in range(outputs.shape[0]):
                pred = outputs_np[i, 0] + 1j * outputs_np[i, 1]
                true = targets_np[i, 0] + 1j * targets_np[i, 1]
                
                # Reshape to 4D for evaluation
                pred_4d = pred.reshape(1, 1, 1, -1) if pred.ndim == 1 else pred.reshape(pred.shape[0], 1, 1, pred.shape[1])
                true_4d = true.reshape(1, 1, 1, -1) if true.ndim == 1 else true.reshape(true.shape[0], 1, 1, true.shape[1])
                
                metrics = evaluate_estimator(true_4d, pred_4d)
                all_mse.append(metrics['mse'])
                all_nmse.append(metrics['nmse_db'])
    
    return {
        'mse_mean': np.mean(all_mse),
        'mse_std': np.std(all_mse),
        'nmse_mean': np.mean(all_nmse),
        'nmse_std': np.std(all_nmse),
        'latency_mean_ms': np.mean(inference_times) * 1000,
        'latency_std_ms': np.std(inference_times) * 1000
    }


def plot_comparison(results: Dict, save_path: Path):
    """Plot comparison of all methods."""
    methods = list(results.keys())
    nmse_values = [results[m]['nmse_mean'] for m in methods]
    nmse_stds = [results[m]['nmse_std'] for m in methods]
    latencies = [results[m]['latency_mean_ms'] for m in methods]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # NMSE comparison
    ax = axes[0]
    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, nmse_values, yerr=nmse_stds, capsize=5, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('NMSE (dB)', fontsize=12)
    ax.set_title('Channel Estimation NMSE Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, nmse_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)
    
    # Latency comparison
    ax = axes[1]
    bars = ax.bar(x_pos, latencies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Inference Latency (ms)', fontsize=12)
    ax.set_title('Inference Latency Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.grid(axis='y', alpha=0.3)
    ax.set_yscale('log')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, latencies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path / 'comparison.png', dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {save_path / 'comparison.png'}")


def generate_report(results: Dict, config: Dict, save_path: Path):
    """Generate evaluation report."""
    report = {
        'configuration': {
            'ofdm': config['ofdm'],
            'mimo': config['mimo'],
            'channel_models': config['channel']['models']
        },
        'results': results
    }
    
    # Save JSON
    with open(save_path / 'evaluation_results.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate text report
    report_text = []
    report_text.append("="*70)
    report_text.append("CHANNEL ESTIMATION EVALUATION REPORT")
    report_text.append("="*70)
    report_text.append("")
    report_text.append("Configuration:")
    report_text.append(f"  FFT Size: {config['ofdm']['fft_size']}")
    report_text.append(f"  CP Length: {config['ofdm']['cp_length']}")
    report_text.append(f"  MIMO: {config['mimo']['num_tx_antennas']}x{config['mimo']['num_rx_antennas']}")
    report_text.append("")
    report_text.append("-"*70)
    report_text.append("Results Summary:")
    report_text.append("-"*70)
    report_text.append("")
    
    for method, metrics in results.items():
        report_text.append(f"{method}:")
        report_text.append(f"  NMSE: {metrics['nmse_mean']:.2f} ± {metrics['nmse_std']:.2f} dB")
        report_text.append(f"  MSE:  {metrics['mse_mean']:.6f} ± {metrics['mse_std']:.6f}")
        report_text.append(f"  Latency: {metrics['latency_mean_ms']:.2f} ± {metrics['latency_std_ms']:.2f} ms")
        report_text.append("")
    
    report_text.append("="*70)
    
    # Save text report
    with open(save_path / 'evaluation_report.txt', 'w') as f:
        f.write('\n'.join(report_text))
    
    # Print to console
    print('\n'.join(report_text))


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate channel estimation models')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml',
                       help='Configuration file')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--model-dir', type=str, default='models',
                       help='Model directory')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--model-types', type=str, nargs='+', default=['cnn'],
                       help='Model types to evaluate')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device(args.device)
    
    # Load test dataset
    print("Loading test dataset...")
    test_dataset = ChannelDataset(
        Path(args.data_dir) / 'test.npz',
        normalize=config['dataset']['normalize']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    print(f"Test samples: {len(test_dataset)}")
    
    # Evaluate all methods
    results = {}
    
    # Baseline: LS
    ls_estimator = LSEstimator(interpolation_method='linear')
    results['LS'] = evaluate_baseline(ls_estimator, test_loader, 'LS')
    
    # Baseline: MMSE
    mmse_estimator = MMSEEstimator(estimate_statistics=True)
    results['MMSE'] = evaluate_baseline(mmse_estimator, test_loader, 'MMSE')
    
    # AI models
    for model_type in args.model_types:
        print(f"\nLoading {model_type.upper()} model...")
        model_config = config['model'].get(model_type, {})
        model = get_model(model_type, model_config)
        
        # Load checkpoint
        checkpoint_path = Path(args.model_dir) / 'best_model.pth'
        if checkpoint_path.exists():
            epoch, loss = load_checkpoint(model, None, checkpoint_path, device)
            print(f"Loaded checkpoint from epoch {epoch} (loss: {loss:.6f})")
        else:
            print(f"Warning: No checkpoint found at {checkpoint_path}")
            continue
        
        model = model.to(device)
        results[model_type.upper()] = evaluate_ai_model(model, test_loader, device)
    
    # Generate plots and report
    plot_comparison(results, output_dir)
    generate_report(results, config, output_dir)
    
    print(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()
