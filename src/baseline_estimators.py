"""
Baseline channel estimation methods: Least Squares (LS) and MMSE.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.interpolate import interp2d, griddata


class LSEstimator:
    """Least Squares channel estimator."""
    
    def __init__(self, interpolation_method: str = 'linear'):
        """
        Initialize LS estimator.
        
        Args:
            interpolation_method: Interpolation method for non-pilot positions
                                 ('linear', 'cubic', 'nearest')
        """
        self.interpolation_method = interpolation_method
    
    def estimate_at_pilots(self, rx_symbols: np.ndarray, tx_pilots: np.ndarray,
                          pilot_mask: np.ndarray) -> np.ndarray:
        """
        Estimate channel at pilot positions using LS.
        
        Args:
            rx_symbols: Received symbols, shape (num_symbols, num_rx, num_tx, num_subcarriers)
            tx_pilots: Transmitted pilot symbols
            pilot_mask: Boolean mask indicating pilot positions
            
        Returns:
            Channel estimates at pilot positions
        """
        # Extract pilots from received signal
        rx_pilots = rx_symbols[pilot_mask]
        
        # LS estimation: H = Y / X
        h_pilots = rx_pilots / (tx_pilots + 1e-12)
        
        return h_pilots
    
    def interpolate_channel(self, h_pilots: np.ndarray, pilot_positions: Tuple,
                           grid_shape: Tuple[int, int]) -> np.ndarray:
        """
        Interpolate channel estimates to full grid.
        
        Args:
            h_pilots: Channel estimates at pilot positions
            pilot_positions: (symbol_indices, subcarrier_indices)
            grid_shape: (num_symbols, num_subcarriers)
            
        Returns:
            Interpolated channel, shape (num_symbols, num_subcarriers)
        """
        num_symbols, num_subcarriers = grid_shape
        pilot_sym_idx, pilot_sc_idx = pilot_positions
        
        # Create full grid coordinates
        sym_grid = np.arange(num_symbols)
        sc_grid = np.arange(num_subcarriers)
        
        # Interpolate real and imaginary parts separately
        h_real_interp = griddata(
            np.column_stack([pilot_sym_idx, pilot_sc_idx]),
            h_pilots.real,
            np.array(np.meshgrid(sym_grid, sc_grid)).T.reshape(-1, 2),
            method=self.interpolation_method,
            fill_value=0.0
        ).reshape(num_symbols, num_subcarriers)
        
        h_imag_interp = griddata(
            np.column_stack([pilot_sym_idx, pilot_sc_idx]),
            h_pilots.imag,
            np.array(np.meshgrid(sym_grid, sc_grid)).T.reshape(-1, 2),
            method=self.interpolation_method,
            fill_value=0.0
        ).reshape(num_symbols, num_subcarriers)
        
        return h_real_interp + 1j * h_imag_interp
    
    def estimate(self, rx_symbols: np.ndarray, tx_pilots: np.ndarray,
                pilot_mask: np.ndarray, pilot_positions: Tuple) -> np.ndarray:
        """
        Complete LS channel estimation with interpolation.
        
        Args:
            rx_symbols: Received symbols
            tx_pilots: Transmitted pilot symbols
            pilot_mask: Pilot position mask
            pilot_positions: Pilot position indices
            
        Returns:
            Estimated channel, shape matching rx_symbols
        """
        original_shape = rx_symbols.shape
        num_symbols, num_rx, num_tx, num_subcarriers = original_shape
        
        # Estimate channel for each RX-TX pair
        H_est = np.zeros_like(rx_symbols, dtype=complex)
        
        for rx in range(num_rx):
            for tx in range(num_tx):
                # Get received signal for this RX-TX pair
                rx_grid = rx_symbols[:, rx, tx, :]  # (num_symbols, num_subcarriers)
                
                # Estimate at pilots
                rx_pilots = rx_grid[pilot_mask]
                h_pilots = rx_pilots / (tx_pilots + 1e-12)
                
                # Interpolate to full grid
                H_est[:, rx, tx, :] = self.interpolate_channel(
                    h_pilots, pilot_positions, (num_symbols, num_subcarriers)
                )
        
        return H_est


class MMSEEstimator:
    """Minimum Mean Square Error channel estimator."""
    
    def __init__(self, channel_covariance: Optional[np.ndarray] = None,
                 noise_variance: float = 0.01, estimate_statistics: bool = True):
        """
        Initialize MMSE estimator.
        
        Args:
            channel_covariance: Channel covariance matrix (if known)
            noise_variance: Noise variance
            estimate_statistics: Whether to estimate statistics from data
        """
        self.channel_covariance = channel_covariance
        self.noise_variance = noise_variance
        self.estimate_statistics = estimate_statistics
    
    def estimate_covariance(self, h_ls: np.ndarray) -> np.ndarray:
        """
        Estimate channel covariance from LS estimates.
        
        Args:
            h_ls: LS channel estimates
            
        Returns:
            Estimated covariance matrix
        """
        # Flatten spatial dimensions
        h_flat = h_ls.reshape(-1, h_ls.shape[-1])
        
        # Estimate covariance
        R_h = np.cov(h_flat, rowvar=False)
        
        return R_h
    
    def estimate_at_pilots(self, rx_symbols: np.ndarray, tx_pilots: np.ndarray,
                          pilot_mask: np.ndarray, snr_db: float = 10) -> np.ndarray:
        """
        MMSE estimation at pilot positions.
        
        Args:
            rx_symbols: Received symbols
            tx_pilots: Transmitted pilot symbols
            pilot_mask: Pilot position mask
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            MMSE channel estimates at pilot positions
        """
        # First get LS estimate
        rx_pilots = rx_symbols[pilot_mask]
        h_ls = rx_pilots / (tx_pilots + 1e-12)
        
        # Update noise variance based on SNR
        snr_linear = 10 ** (snr_db / 10)
        self.noise_variance = 1.0 / snr_linear
        
        if self.estimate_statistics or self.channel_covariance is None:
            # Use diagonal approximation of covariance
            signal_power = np.mean(np.abs(h_ls) ** 2)
            R_h = np.eye(len(h_ls)) * signal_power
        else:
            R_h = self.channel_covariance
        
        # MMSE filtering: H_mmse = R_h @ (R_h + sigma^2 * I)^-1 @ H_ls
        R_y = R_h + self.noise_variance * np.eye(len(h_ls))
        
        try:
            # Compute MMSE estimate
            mmse_matrix = R_h @ np.linalg.inv(R_y)
            h_mmse = mmse_matrix @ h_ls
        except np.linalg.LinAlgError:
            # Fallback to regularized inverse
            mmse_matrix = R_h @ np.linalg.inv(R_y + 1e-6 * np.eye(len(h_ls)))
            h_mmse = mmse_matrix @ h_ls
        
        return h_mmse
    
    def interpolate_channel(self, h_pilots: np.ndarray, pilot_positions: Tuple,
                           grid_shape: Tuple[int, int], 
                           interpolation_method: str = 'linear') -> np.ndarray:
        """
        Interpolate MMSE estimates to full grid.
        
        Uses same interpolation as LS estimator.
        """
        num_symbols, num_subcarriers = grid_shape
        pilot_sym_idx, pilot_sc_idx = pilot_positions
        
        # Create full grid coordinates
        sym_grid = np.arange(num_symbols)
        sc_grid = np.arange(num_subcarriers)
        
        # Interpolate real and imaginary parts
        h_real_interp = griddata(
            np.column_stack([pilot_sym_idx, pilot_sc_idx]),
            h_pilots.real,
            np.array(np.meshgrid(sym_grid, sc_grid)).T.reshape(-1, 2),
            method=interpolation_method,
            fill_value=0.0
        ).reshape(num_symbols, num_subcarriers)
        
        h_imag_interp = griddata(
            np.column_stack([pilot_sym_idx, pilot_sc_idx]),
            h_pilots.imag,
            np.array(np.meshgrid(sym_grid, sc_grid)).T.reshape(-1, 2),
            method=interpolation_method,
            fill_value=0.0
        ).reshape(num_symbols, num_subcarriers)
        
        return h_real_interp + 1j * h_imag_interp
    
    def estimate(self, rx_symbols: np.ndarray, tx_pilots: np.ndarray,
                pilot_mask: np.ndarray, pilot_positions: Tuple,
                snr_db: float = 10) -> np.ndarray:
        """
        Complete MMSE channel estimation.
        
        Args:
            rx_symbols: Received symbols
            tx_pilots: Transmitted pilot symbols
            pilot_mask: Pilot position mask
            pilot_positions: Pilot position indices
            snr_db: Signal-to-noise ratio
            
        Returns:
            Estimated channel
        """
        original_shape = rx_symbols.shape
        num_symbols, num_rx, num_tx, num_subcarriers = original_shape
        
        H_est = np.zeros_like(rx_symbols, dtype=complex)
        
        for rx in range(num_rx):
            for tx in range(num_tx):
                # Get received signal for this RX-TX pair
                rx_grid = rx_symbols[:, rx, tx, :]
                
                # MMSE estimate at pilots
                rx_pilots = rx_grid[pilot_mask]
                h_pilots = self.estimate_at_pilots(
                    rx_pilots.reshape(-1), tx_pilots, 
                    np.ones_like(tx_pilots, dtype=bool), snr_db
                )
                
                # Interpolate to full grid
                H_est[:, rx, tx, :] = self.interpolate_channel(
                    h_pilots, pilot_positions, (num_symbols, num_subcarriers)
                )
        
        return H_est


def equalize_channel(rx_symbols: np.ndarray, H_est: np.ndarray, 
                    method: str = 'zf') -> np.ndarray:
    """
    Equalize received symbols using estimated channel.
    
    Args:
        rx_symbols: Received symbols, shape (num_symbols, num_rx, num_subcarriers)
        H_est: Estimated channel, shape (num_symbols, num_rx, num_tx, num_subcarriers)
        method: Equalization method ('zf' for Zero-Forcing, 'mmse' for MMSE)
        
    Returns:
        Equalized symbols, shape (num_symbols, num_tx, num_subcarriers)
    """
    num_symbols, num_rx, num_tx, num_subcarriers = H_est.shape
    equalized = np.zeros((num_symbols, num_tx, num_subcarriers), dtype=complex)
    
    for sym_idx in range(num_symbols):
        for sc_idx in range(num_subcarriers):
            H = H_est[sym_idx, :, :, sc_idx]  # (num_rx, num_tx)
            y = rx_symbols[sym_idx, :, sc_idx]  # (num_rx,)
            
            if method == 'zf':
                # Zero-Forcing: x_hat = (H^H H)^-1 H^H y
                try:
                    W = np.linalg.inv(H.conj().T @ H + 1e-8 * np.eye(num_tx)) @ H.conj().T
                    equalized[sym_idx, :, sc_idx] = W @ y
                except np.linalg.LinAlgError:
                    # Fallback: pseudo-inverse
                    equalized[sym_idx, :, sc_idx] = np.linalg.pinv(H) @ y
            
            elif method == 'mmse':
                # MMSE equalization: x_hat = (H^H H + sigma^2 I)^-1 H^H y
                noise_var = 0.01  # Could be estimated
                W = np.linalg.inv(H.conj().T @ H + noise_var * np.eye(num_tx)) @ H.conj().T
                equalized[sym_idx, :, sc_idx] = W @ y
            
            else:
                raise ValueError(f"Unknown equalization method: {method}")
    
    return equalized


def evaluate_estimator(H_true: np.ndarray, H_est: np.ndarray) -> dict:
    """
    Evaluate channel estimation performance.
    
    Args:
        H_true: True channel
        H_est: Estimated channel
        
    Returns:
        Dictionary with MSE and NMSE metrics
    """
    mse = np.mean(np.abs(H_true - H_est) ** 2)
    
    # Normalized MSE
    channel_power = np.mean(np.abs(H_true) ** 2)
    nmse = mse / (channel_power + 1e-12)
    nmse_db = 10 * np.log10(nmse + 1e-12)
    
    return {
        'mse': mse,
        'nmse': nmse,
        'nmse_db': nmse_db
    }


if __name__ == '__main__':
    """Test baseline estimators."""
    import sys
    sys.path.append('.')
    from src.channel_simulator import simulate_transmission
    from src.utils import load_config
    
    # Load config
    config = load_config()
    
    # Simulate transmission
    print("Simulating MIMO-OFDM transmission...")
    sim_data = simulate_transmission(
        config, 
        channel_type='EVA',
        doppler_hz=50,
        snr_db=15,
        pilot_density=0.1
    )
    
    # Extract data
    rx_symbols = sim_data['rx_symbols']
    H_true = sim_data['channel']
    pilot_pattern = sim_data['pilot_pattern']
    tx_pilots = sim_data['pilot_symbols']
    
    # Reshape for estimation
    num_symbols = rx_symbols.shape[0]
    num_rx = rx_symbols.shape[1]
    num_tx = sim_data['tx_symbols'].shape[1]
    num_sc = rx_symbols.shape[2]
    
    # Create 4D arrays for estimators
    rx_4d = rx_symbols.reshape(num_symbols, num_rx, 1, num_sc)
    rx_4d = np.repeat(rx_4d, num_tx, axis=2)
    
    # LS Estimation
    print("\nLS Estimation...")
    ls_estimator = LSEstimator(interpolation_method='linear')
    H_ls = ls_estimator.estimate(
        rx_4d, tx_pilots, 
        pilot_pattern.pilot_mask, 
        pilot_pattern.pilot_positions
    )
    
    ls_metrics = evaluate_estimator(H_true, H_ls)
    print(f"LS NMSE: {ls_metrics['nmse_db']:.2f} dB")
    
    # MMSE Estimation
    print("\nMMSE Estimation...")
    mmse_estimator = MMSEEstimator(estimate_statistics=True)
    H_mmse = mmse_estimator.estimate(
        rx_4d, tx_pilots, 
        pilot_pattern.pilot_mask, 
        pilot_pattern.pilot_positions,
        snr_db=15
    )
    
    mmse_metrics = evaluate_estimator(H_true, H_mmse)
    print(f"MMSE NMSE: {mmse_metrics['nmse_db']:.2f} dB")
    
    print("\nBaseline estimators test completed successfully!")
