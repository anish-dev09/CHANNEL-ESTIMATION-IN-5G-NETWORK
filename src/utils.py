"""
Utility functions for the 5G channel estimation project.
"""

import numpy as np
import torch
import random
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str = 'configs/experiment_config.yaml') -> Dict[str, Any]:
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_directories(config: Dict[str, Any]):
    """Create necessary directories for the project."""
    paths = config.get('paths', {})
    for path_key, path_value in paths.items():
        Path(path_value).mkdir(parents=True, exist_ok=True)


def db2linear(db_value: float) -> float:
    """Convert dB to linear scale."""
    return 10 ** (db_value / 10)


def linear2db(linear_value: float) -> float:
    """Convert linear scale to dB."""
    return 10 * np.log10(linear_value + 1e-12)


def awgn_noise(signal_shape: Tuple, snr_db: float, signal_power: float = 1.0) -> np.ndarray:
    """
    Generate AWGN noise for a given SNR.
    
    Args:
        signal_shape: Shape of the signal
        snr_db: Signal-to-noise ratio in dB
        signal_power: Average power of the signal
        
    Returns:
        Complex AWGN noise
    """
    snr_linear = db2linear(snr_db)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2)  # Divide by 2 for complex noise
    
    noise_real = np.random.randn(*signal_shape) * noise_std
    noise_imag = np.random.randn(*signal_shape) * noise_std
    
    return noise_real + 1j * noise_imag


def qam_modulation(bits: np.ndarray, M: int = 4) -> np.ndarray:
    """
    QAM modulation.
    
    Args:
        bits: Binary data
        M: Modulation order (4, 16, 64, 256)
        
    Returns:
        Modulated symbols
    """
    bits_per_symbol = int(np.log2(M))
    num_symbols = len(bits) // bits_per_symbol
    
    # Reshape bits into symbols
    bit_matrix = bits[:num_symbols * bits_per_symbol].reshape(-1, bits_per_symbol)
    
    # Convert to decimal
    decimal_values = np.dot(bit_matrix, 2 ** np.arange(bits_per_symbol)[::-1])
    
    # Gray coding mapping for QPSK and 16-QAM
    if M == 4:  # QPSK
        constellation = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
        gray_map = [0, 1, 3, 2]
    elif M == 16:  # 16-QAM
        constellation = np.array([
            -3-3j, -3-1j, -3+3j, -3+1j,
            -1-3j, -1-1j, -1+3j, -1+1j,
            3-3j, 3-1j, 3+3j, 3+1j,
            1-3j, 1-1j, 1+3j, 1+1j
        ]) / np.sqrt(10)
        gray_map = [0, 1, 3, 2, 4, 5, 7, 6, 12, 13, 15, 14, 8, 9, 11, 10]
    else:
        raise NotImplementedError(f"Modulation order {M} not implemented")
    
    # Map to constellation
    symbols = constellation[gray_map[decimal_values]]
    
    return symbols


def qam_demodulation(symbols: np.ndarray, M: int = 4) -> np.ndarray:
    """
    QAM demodulation using minimum distance.
    
    Args:
        symbols: Received symbols
        M: Modulation order
        
    Returns:
        Demodulated bits
    """
    bits_per_symbol = int(np.log2(M))
    
    # Constellation
    if M == 4:  # QPSK
        constellation = np.array([1+1j, -1+1j, 1-1j, -1-1j]) / np.sqrt(2)
        gray_map = [0, 1, 3, 2]
    elif M == 16:  # 16-QAM
        constellation = np.array([
            -3-3j, -3-1j, -3+3j, -3+1j,
            -1-3j, -1-1j, -1+3j, -1+1j,
            3-3j, 3-1j, 3+3j, 3+1j,
            1-3j, 1-1j, 1+3j, 1+1j
        ]) / np.sqrt(10)
        gray_map = [0, 1, 3, 2, 4, 5, 7, 6, 12, 13, 15, 14, 8, 9, 11, 10]
    else:
        raise NotImplementedError(f"Demodulation order {M} not implemented")
    
    # Minimum distance decoding
    distances = np.abs(symbols[:, np.newaxis] - constellation[np.newaxis, :])
    detected_indices = np.argmin(distances, axis=1)
    
    # Reverse gray mapping
    inverse_gray_map = np.argsort(gray_map)
    decimal_values = inverse_gray_map[detected_indices]
    
    # Convert to bits
    bits = np.zeros((len(symbols), bits_per_symbol), dtype=int)
    for i in range(bits_per_symbol):
        bits[:, bits_per_symbol - 1 - i] = (decimal_values >> i) & 1
    
    return bits.flatten()


def calculate_ber(transmitted_bits: np.ndarray, received_bits: np.ndarray) -> float:
    """Calculate bit error rate."""
    return np.sum(transmitted_bits != received_bits) / len(transmitted_bits)


def calculate_mse(true_channel: np.ndarray, estimated_channel: np.ndarray) -> float:
    """Calculate mean squared error between true and estimated channel."""
    return np.mean(np.abs(true_channel - estimated_channel) ** 2)


def calculate_nmse(true_channel: np.ndarray, estimated_channel: np.ndarray) -> float:
    """Calculate normalized mean squared error."""
    mse = calculate_mse(true_channel, estimated_channel)
    power = np.mean(np.abs(true_channel) ** 2)
    return mse / (power + 1e-12)


def complex_to_real(complex_array: np.ndarray) -> np.ndarray:
    """Convert complex array to real array with separate real and imaginary parts."""
    return np.stack([complex_array.real, complex_array.imag], axis=-1)


def real_to_complex(real_array: np.ndarray) -> np.ndarray:
    """Convert real array with separate real and imaginary parts to complex."""
    return real_array[..., 0] + 1j * real_array[..., 1]


def get_device(device_str: str = 'auto') -> torch.device:
    """Get PyTorch device."""
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device_str)


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    epoch: int, loss: float, path: str):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer], 
                    path: str, device: torch.device) -> Tuple[int, float]:
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
