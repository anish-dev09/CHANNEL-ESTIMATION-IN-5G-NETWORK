"""
OFDM and MIMO channel simulator for 5G systems.

Implements:
- OFDM modulation/demodulation
- MIMO transmission
- Channel models: EPA, EVA, ETU
- Doppler effects
- Pilot insertion and extraction
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class OFDMConfig:
    """OFDM system configuration."""
    fft_size: int = 1024
    cp_length: int = 72
    num_symbols: int = 14
    useful_subcarriers: int = 600
    subcarrier_spacing: float = 15000.0  # Hz
    

@dataclass
class MIMOConfig:
    """MIMO configuration."""
    num_tx: int = 2
    num_rx: int = 2


class ChannelModel:
    """
    Multipath fading channel models for 5G.
    Implements EPA, EVA, and ETU models.
    """
    
    # Channel model parameters (delay in ns, power in dB)
    CHANNEL_PROFILES = {
        'EPA': {  # Extended Pedestrian A
            'delays': np.array([0, 30, 70, 90, 110, 190, 410]) * 1e-9,
            'powers': np.array([0.0, -1.0, -2.0, -3.0, -8.0, -17.2, -20.8])
        },
        'EVA': {  # Extended Vehicular A
            'delays': np.array([0, 30, 150, 310, 370, 710, 1090, 1730, 2510]) * 1e-9,
            'powers': np.array([0.0, -1.5, -1.4, -3.6, -0.6, -9.1, -7.0, -12.0, -16.9])
        },
        'ETU': {  # Extended Typical Urban
            'delays': np.array([0, 50, 120, 200, 230, 500, 1600, 2300, 5000]) * 1e-9,
            'powers': np.array([-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, -3.0, -5.0, -7.0])
        }
    }
    
    def __init__(self, model_type: str, doppler_hz: float, 
                 carrier_freq: float, sampling_rate: float):
        """
        Initialize channel model.
        
        Args:
            model_type: 'EPA', 'EVA', or 'ETU'
            doppler_hz: Maximum Doppler frequency in Hz
            carrier_freq: Carrier frequency in Hz
            sampling_rate: System sampling rate in Hz
        """
        self.model_type = model_type.upper()
        self.doppler_hz = doppler_hz
        self.carrier_freq = carrier_freq
        self.sampling_rate = sampling_rate
        
        profile = self.CHANNEL_PROFILES[self.model_type]
        self.delays = profile['delays']
        self.powers_db = profile['powers']
        self.powers_linear = 10 ** (self.powers_db / 10)
        
        # Normalize power
        self.powers_linear = self.powers_linear / np.sum(self.powers_linear)
        
        # Convert delays to sample indices
        self.delay_samples = np.round(self.delays * sampling_rate).astype(int)
        self.num_paths = len(self.delays)
        
    def generate_time_varying_channel(self, num_samples: int, num_tx: int, num_rx: int) -> np.ndarray:
        """
        Generate time-varying multipath channel using Jakes model.
        
        Args:
            num_samples: Number of time samples
            num_tx: Number of transmit antennas
            num_rx: Number of receive antennas
            
        Returns:
            Channel matrix of shape (num_samples, num_rx, num_tx, max_delay + 1)
        """
        max_delay = np.max(self.delay_samples)
        channel = np.zeros((num_samples, num_rx, num_tx, max_delay + 1), dtype=complex)
        
        # Jakes model parameters
        num_oscillators = 20  # Number of sinusoids for Jakes model
        
        for path_idx in range(self.num_paths):
            delay_idx = self.delay_samples[path_idx]
            path_power = np.sqrt(self.powers_linear[path_idx])
            
            for tx in range(num_tx):
                for rx in range(num_rx):
                    # Generate complex fading using Jakes model
                    angles = 2 * np.pi * np.random.rand(num_oscillators)
                    phases = 2 * np.pi * np.random.rand(num_oscillators)
                    
                    time_vec = np.arange(num_samples) / self.sampling_rate
                    
                    # Real and imaginary parts
                    h_real = np.zeros(num_samples)
                    h_imag = np.zeros(num_samples)
                    
                    for n in range(num_oscillators):
                        doppler_shift = self.doppler_hz * np.cos(angles[n])
                        h_real += np.cos(2 * np.pi * doppler_shift * time_vec + phases[n])
                        h_imag += np.sin(2 * np.pi * doppler_shift * time_vec + phases[n])
                    
                    # Normalize and scale by path power
                    h_complex = (h_real + 1j * h_imag) / np.sqrt(2 * num_oscillators)
                    channel[:, rx, tx, delay_idx] = path_power * h_complex
        
        return channel


class OFDMSystem:
    """OFDM transmission system."""
    
    def __init__(self, config: OFDMConfig):
        """Initialize OFDM system."""
        self.config = config
        self.sampling_rate = config.fft_size * config.subcarrier_spacing
        
        # Define used subcarriers (center of spectrum)
        total_sc = config.fft_size
        used_sc = config.useful_subcarriers
        self.dc_idx = total_sc // 2
        self.used_indices = np.arange(
            self.dc_idx - used_sc // 2,
            self.dc_idx + used_sc // 2
        )
        
        # Remove DC subcarrier
        self.used_indices = self.used_indices[self.used_indices != self.dc_idx]
        
    def modulate(self, symbols: np.ndarray) -> np.ndarray:
        """
        OFDM modulation.
        
        Args:
            symbols: Frequency domain symbols, shape (num_symbols, useful_subcarriers)
            
        Returns:
            Time domain signal with CP, shape (num_symbols, fft_size + cp_length)
        """
        num_ofdm_symbols = symbols.shape[0]
        time_signal = np.zeros((num_ofdm_symbols, self.config.fft_size + self.config.cp_length), 
                              dtype=complex)
        
        for sym_idx in range(num_ofdm_symbols):
            # Map symbols to FFT bins
            freq_domain = np.zeros(self.config.fft_size, dtype=complex)
            freq_domain[self.used_indices] = symbols[sym_idx]
            
            # IFFT
            time_domain = np.fft.ifft(np.fft.ifftshift(freq_domain)) * np.sqrt(self.config.fft_size)
            
            # Add cyclic prefix
            time_signal[sym_idx] = np.concatenate([
                time_domain[-self.config.cp_length:],
                time_domain
            ])
        
        return time_signal
    
    def demodulate(self, received_signal: np.ndarray) -> np.ndarray:
        """
        OFDM demodulation.
        
        Args:
            received_signal: Time domain signal with CP, shape (num_symbols, fft_size + cp_length)
            
        Returns:
            Frequency domain symbols, shape (num_symbols, useful_subcarriers)
        """
        num_ofdm_symbols = received_signal.shape[0]
        symbols = np.zeros((num_ofdm_symbols, len(self.used_indices)), dtype=complex)
        
        for sym_idx in range(num_ofdm_symbols):
            # Remove cyclic prefix
            time_domain = received_signal[sym_idx, self.config.cp_length:]
            
            # FFT
            freq_domain = np.fft.fftshift(np.fft.fft(time_domain)) / np.sqrt(self.config.fft_size)
            
            # Extract used subcarriers
            symbols[sym_idx] = freq_domain[self.used_indices]
        
        return symbols


class PilotPattern:
    """Pilot pattern generation and management."""
    
    def __init__(self, num_subcarriers: int, num_symbols: int, pilot_density: float = 0.1):
        """
        Initialize pilot pattern.
        
        Args:
            num_subcarriers: Number of subcarriers
            num_symbols: Number of OFDM symbols
            pilot_density: Ratio of pilots to total resource elements
        """
        self.num_subcarriers = num_subcarriers
        self.num_symbols = num_symbols
        self.pilot_density = pilot_density
        
        # Generate comb-type pilot pattern
        total_res = num_subcarriers * num_symbols
        num_pilots = int(total_res * pilot_density)
        
        # Uniformly distributed pilots
        all_indices = np.arange(total_res)
        np.random.shuffle(all_indices)
        self.pilot_indices = np.sort(all_indices[:num_pilots])
        
        # Convert to 2D indices
        self.pilot_positions = np.unravel_index(self.pilot_indices, (num_symbols, num_subcarriers))
        
        # Create mask
        self.pilot_mask = np.zeros((num_symbols, num_subcarriers), dtype=bool)
        self.pilot_mask[self.pilot_positions] = True
        
    def insert_pilots(self, data_symbols: np.ndarray, pilot_symbols: np.ndarray) -> np.ndarray:
        """
        Insert pilot symbols into data.
        
        Args:
            data_symbols: Data symbols for non-pilot positions
            pilot_symbols: Pilot symbols
            
        Returns:
            Grid with both data and pilots
        """
        grid = np.zeros((self.num_symbols, self.num_subcarriers), dtype=complex)
        grid[self.pilot_mask] = pilot_symbols
        grid[~self.pilot_mask] = data_symbols
        return grid
    
    def extract_pilots(self, grid: np.ndarray) -> np.ndarray:
        """Extract pilot symbols from received grid."""
        return grid[self.pilot_mask]
    
    def get_pilot_positions(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get pilot positions as (symbol_indices, subcarrier_indices)."""
        return self.pilot_positions


class MIMOChannel:
    """MIMO channel simulation and application."""
    
    def __init__(self, ofdm_config: OFDMConfig, mimo_config: MIMOConfig, 
                 channel_model: ChannelModel):
        """Initialize MIMO channel."""
        self.ofdm_config = ofdm_config
        self.mimo_config = mimo_config
        self.channel_model = channel_model
        self.ofdm_system = OFDMSystem(ofdm_config)
        
    def generate_channel_frequency_response(self, num_symbols: int) -> np.ndarray:
        """
        Generate frequency-domain channel response for OFDM-MIMO.
        
        Args:
            num_symbols: Number of OFDM symbols
            
        Returns:
            Channel frequency response, shape (num_symbols, num_rx, num_tx, num_subcarriers)
        """
        # Total time samples for all OFDM symbols
        samples_per_symbol = self.ofdm_config.fft_size + self.ofdm_config.cp_length
        num_samples = num_symbols * samples_per_symbol
        
        # Generate time-domain channel
        h_time = self.channel_model.generate_time_varying_channel(
            num_samples, 
            self.mimo_config.num_tx, 
            self.mimo_config.num_rx
        )
        
        # Convert to frequency domain for each OFDM symbol
        num_sc = len(self.ofdm_system.used_indices)
        H_freq = np.zeros((num_symbols, self.mimo_config.num_rx, 
                          self.mimo_config.num_tx, num_sc), dtype=complex)
        
        for sym_idx in range(num_symbols):
            sample_idx = sym_idx * samples_per_symbol
            h_impulse = h_time[sample_idx]  # (num_rx, num_tx, max_delay + 1)
            
            # FFT to get frequency response
            for rx in range(self.mimo_config.num_rx):
                for tx in range(self.mimo_config.num_tx):
                    h_fft = np.fft.fft(h_impulse[rx, tx], n=self.ofdm_config.fft_size)
                    h_fft_shifted = np.fft.fftshift(h_fft)
                    H_freq[sym_idx, rx, tx] = h_fft_shifted[self.ofdm_system.used_indices]
        
        return H_freq
    
    def apply_channel(self, transmitted_symbols: np.ndarray, 
                     channel_response: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Apply MIMO channel to transmitted symbols.
        
        Args:
            transmitted_symbols: shape (num_symbols, num_tx, num_subcarriers)
            channel_response: shape (num_symbols, num_rx, num_tx, num_subcarriers)
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Received symbols, shape (num_symbols, num_rx, num_subcarriers)
        """
        num_symbols, num_rx, num_tx, num_sc = channel_response.shape
        received = np.zeros((num_symbols, num_rx, num_sc), dtype=complex)
        
        # Apply channel: y = Hx for each subcarrier
        for sym_idx in range(num_symbols):
            for sc_idx in range(num_sc):
                H = channel_response[sym_idx, :, :, sc_idx]  # (num_rx, num_tx)
                x = transmitted_symbols[sym_idx, :, sc_idx]  # (num_tx,)
                received[sym_idx, :, sc_idx] = H @ x
        
        # Add AWGN
        signal_power = np.mean(np.abs(received) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)
        
        noise = (np.random.randn(*received.shape) + 1j * np.random.randn(*received.shape)) * noise_std
        received = received + noise
        
        return received


def simulate_transmission(config: Dict, channel_type: str = 'EPA', 
                         doppler_hz: float = 50, snr_db: float = 10,
                         pilot_density: float = 0.1) -> Dict:
    """
    Complete MIMO-OFDM transmission simulation.
    
    Args:
        config: System configuration dictionary
        channel_type: Channel model type
        doppler_hz: Doppler frequency
        snr_db: SNR in dB
        pilot_density: Pilot density ratio
        
    Returns:
        Dictionary with transmitted symbols, received symbols, channel response, etc.
    """
    # Initialize configurations
    ofdm_cfg = OFDMConfig(
        fft_size=config['ofdm']['fft_size'],
        cp_length=config['ofdm']['cp_length'],
        num_symbols=config['ofdm']['num_symbols'],
        useful_subcarriers=config['ofdm']['useful_subcarriers'],
        subcarrier_spacing=config['ofdm']['subcarrier_spacing']
    )
    
    mimo_cfg = MIMOConfig(
        num_tx=config['mimo']['num_tx_antennas'],
        num_rx=config['mimo']['num_rx_antennas']
    )
    
    # Channel model
    sampling_rate = ofdm_cfg.fft_size * ofdm_cfg.subcarrier_spacing
    channel_model = ChannelModel(
        channel_type, doppler_hz, 
        config['channel']['carrier_freq'], 
        sampling_rate
    )
    
    # MIMO channel
    mimo_channel = MIMOChannel(ofdm_cfg, mimo_cfg, channel_model)
    
    # Pilot pattern
    num_sc = len(mimo_channel.ofdm_system.used_indices)
    pilot_pattern = PilotPattern(num_sc, ofdm_cfg.num_symbols, pilot_density)
    
    # Generate pilot symbols (known reference)
    num_pilots = np.sum(pilot_pattern.pilot_mask)
    pilot_symbols = np.exp(1j * np.random.uniform(0, 2*np.pi, num_pilots))
    
    # Generate data symbols
    num_data = ofdm_cfg.num_symbols * num_sc - num_pilots
    data_symbols = np.exp(1j * np.random.uniform(0, 2*np.pi, num_data))
    
    # Create transmission grid for each TX antenna
    tx_symbols = np.zeros((ofdm_cfg.num_symbols, mimo_cfg.num_tx, num_sc), dtype=complex)
    for tx in range(mimo_cfg.num_tx):
        tx_symbols[:, tx, :] = pilot_pattern.insert_pilots(data_symbols, pilot_symbols)
    
    # Generate channel
    H = mimo_channel.generate_channel_frequency_response(ofdm_cfg.num_symbols)
    
    # Apply channel
    rx_symbols = mimo_channel.apply_channel(tx_symbols, H, snr_db)
    
    return {
        'tx_symbols': tx_symbols,
        'rx_symbols': rx_symbols,
        'channel': H,
        'pilot_pattern': pilot_pattern,
        'pilot_symbols': pilot_symbols,
        'ofdm_config': ofdm_cfg,
        'mimo_config': mimo_cfg,
        'snr_db': snr_db
    }
