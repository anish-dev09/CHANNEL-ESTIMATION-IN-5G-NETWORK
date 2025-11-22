"""
Neural network models for AI-assisted channel estimation.

Implements:
- CNN-based estimator
- LSTM-based estimator
- Hybrid CNN-LSTM estimator
- Transformer-based estimator (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CNNChannelEstimator(nn.Module):
    """
    CNN-based channel estimator.
    
    Uses 2D convolutions on time-frequency grid to learn spatial patterns.
    """
    
    def __init__(self, input_channels: int = 5, hidden_channels: list = None,
                 kernel_size: int = 3, dropout: float = 0.1):
        """
        Initialize CNN estimator.
        
        Args:
            input_channels: Number of input channels (default: 5 = rx_real, rx_imag, 
                           H_ls_real, H_ls_imag, pilot_mask)
            hidden_channels: List of hidden channel dimensions
            kernel_size: Convolution kernel size
            dropout: Dropout probability
        """
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [64, 128, 256, 128, 64]
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        
        # Build convolutional layers
        layers = []
        in_ch = input_channels
        
        for out_ch in hidden_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ])
            in_ch = out_ch
        
        # Final layer to produce 2 channels (real and imaginary)
        layers.append(nn.Conv2d(in_ch, 2, kernel_size=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch, channels, num_symbols, num_subcarriers)
            
        Returns:
            Estimated channel, shape (batch, 2, num_symbols, num_subcarriers)
        """
        return self.network(x)


class LSTMChannelEstimator(nn.Module):
    """
    LSTM-based channel estimator.
    
    Processes temporal sequences to capture time-varying channel dynamics.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 256,
                 num_layers: int = 3, bidirectional: bool = True, dropout: float = 0.2):
        """
        Initialize LSTM estimator.
        
        Args:
            input_size: Size of input features per time step
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            bidirectional: Use bidirectional LSTM
            dropout: Dropout between LSTM layers
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, 2)  # 2 for real and imaginary
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch, num_symbols, features)
            
        Returns:
            Estimated channel, shape (batch, num_symbols, 2)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Map to channel estimates
        output = self.fc(lstm_out)
        
        return output


class HybridCNNLSTMEstimator(nn.Module):
    """
    Hybrid CNN-LSTM channel estimator.
    
    Uses CNN for spatial feature extraction and LSTM for temporal modeling.
    """
    
    def __init__(self, input_channels: int = 5, cnn_channels: list = None,
                 lstm_hidden: int = 256, lstm_layers: int = 2,
                 kernel_size: int = 3, dropout: float = 0.1):
        """
        Initialize hybrid estimator.
        
        Args:
            input_channels: Number of input channels
            cnn_channels: CNN hidden channel dimensions
            lstm_hidden: LSTM hidden size
            lstm_layers: Number of LSTM layers
            kernel_size: CNN kernel size
            dropout: Dropout probability
        """
        super().__init__()
        
        if cnn_channels is None:
            cnn_channels = [32, 64, 128]
        
        # CNN for spatial features
        cnn_layers = []
        in_ch = input_channels
        
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, 
                         padding=kernel_size//2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout)
            ])
            in_ch = out_ch
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM for temporal modeling
        # Input: (batch, num_symbols, cnn_channels[-1] * num_subcarriers)
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],  # Will be applied per subcarrier
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output layer
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch, channels, num_symbols, num_subcarriers)
            
        Returns:
            Estimated channel, shape (batch, 2, num_symbols, num_subcarriers)
        """
        batch_size, _, num_symbols, num_subcarriers = x.shape
        
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch, cnn_channels[-1], num_symbols, num_subcarriers)
        
        # Reshape for LSTM: process each subcarrier's time series
        # (batch, num_subcarriers, num_symbols, features)
        cnn_out = cnn_out.permute(0, 3, 2, 1)
        
        # Process each subcarrier independently
        outputs = []
        for sc in range(num_subcarriers):
            lstm_in = cnn_out[:, sc, :, :]  # (batch, num_symbols, features)
            lstm_out, _ = self.lstm(lstm_in)  # (batch, num_symbols, lstm_hidden*2)
            sc_out = self.fc(lstm_out)  # (batch, num_symbols, 2)
            outputs.append(sc_out)
        
        # Stack and reshape
        output = torch.stack(outputs, dim=3)  # (batch, num_symbols, 2, num_subcarriers)
        output = output.permute(0, 2, 1, 3)  # (batch, 2, num_symbols, num_subcarriers)
        
        return output


class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ResNetChannelEstimator(nn.Module):
    """
    ResNet-style channel estimator with residual connections.
    """
    
    def __init__(self, input_channels: int = 5, base_channels: int = 64,
                 num_blocks: int = 4, dropout: float = 0.1):
        """
        Initialize ResNet estimator.
        
        Args:
            input_channels: Number of input channels
            base_channels: Base number of channels
            num_blocks: Number of residual blocks
            dropout: Dropout probability
        """
        super().__init__()
        
        # Initial convolution
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(base_channels, dropout=dropout) 
            for _ in range(num_blocks)
        ])
        
        # Output convolution
        self.conv_out = nn.Conv2d(base_channels, 2, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch, channels, num_symbols, num_subcarriers)
            
        Returns:
            Estimated channel, shape (batch, 2, num_symbols, num_subcarriers)
        """
        x = self.conv_in(x)
        
        for block in self.res_blocks:
            x = block(x)
        
        x = self.conv_out(x)
        
        return x


class AttentionBlock(nn.Module):
    """Self-attention block for channel estimation."""
    
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor, shape (batch, seq_len, channels)
            
        Returns:
            Attention output
        """
        attn_out, _ = self.attention(x, x, x)
        return self.norm(x + attn_out)


def get_model(model_type: str, config: dict) -> nn.Module:
    """
    Factory function to get model by type.
    
    Args:
        model_type: 'cnn', 'lstm', 'hybrid', or 'resnet'
        config: Model configuration dictionary
        
    Returns:
        Initialized model
    """
    model_type = model_type.lower()
    
    if model_type == 'cnn':
        return CNNChannelEstimator(
            input_channels=config.get('input_channels', 5),
            hidden_channels=config.get('hidden_channels', [64, 128, 256, 128, 64]),
            kernel_size=config.get('kernel_size', 3),
            dropout=config.get('dropout', 0.1)
        )
    
    elif model_type == 'lstm':
        return LSTMChannelEstimator(
            input_size=config.get('input_size', 5),
            hidden_size=config.get('hidden_size', 256),
            num_layers=config.get('num_layers', 3),
            bidirectional=config.get('bidirectional', True),
            dropout=config.get('dropout', 0.2)
        )
    
    elif model_type == 'hybrid' or model_type == 'cnn_lstm':
        return HybridCNNLSTMEstimator(
            input_channels=config.get('input_channels', 5),
            cnn_channels=config.get('cnn_channels', [32, 64, 128]),
            lstm_hidden=config.get('lstm_hidden', 256),
            lstm_layers=config.get('lstm_layers', 2),
            dropout=config.get('dropout', 0.1)
        )
    
    elif model_type == 'resnet':
        return ResNetChannelEstimator(
            input_channels=config.get('input_channels', 5),
            base_channels=config.get('base_channels', 64),
            num_blocks=config.get('num_blocks', 4),
            dropout=config.get('dropout', 0.1)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


class ChannelEstimationLoss(nn.Module):
    """Custom loss function for channel estimation."""
    
    def __init__(self, loss_type: str = 'mse', 
                 channel_weight: float = 1.0,
                 pilot_weight: float = 0.0):
        """
        Initialize loss function.
        
        Args:
            loss_type: 'mse', 'mae', or 'huber'
            channel_weight: Weight for channel estimation loss
            pilot_weight: Additional weight for pilot positions
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.channel_weight = channel_weight
        self.pilot_weight = pilot_weight
        
        if loss_type == 'mse':
            self.base_loss = nn.MSELoss()
        elif loss_type == 'mae':
            self.base_loss = nn.L1Loss()
        elif loss_type == 'huber':
            self.base_loss = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                pilot_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute loss.
        
        Args:
            pred: Predicted channel
            target: True channel
            pilot_mask: Optional pilot position mask
            
        Returns:
            Loss value
        """
        # Base loss
        loss = self.base_loss(pred, target) * self.channel_weight
        
        # Additional loss at pilot positions
        if pilot_mask is not None and self.pilot_weight > 0:
            pilot_loss = self.base_loss(pred * pilot_mask, target * pilot_mask)
            loss = loss + pilot_loss * self.pilot_weight
        
        return loss


if __name__ == '__main__':
    """Test models."""
    print("Testing channel estimation models...")
    
    # Test parameters
    batch_size = 4
    input_channels = 5
    num_symbols = 14
    num_subcarriers = 600
    
    # Create dummy input
    x = torch.randn(batch_size, input_channels, num_symbols, num_subcarriers)
    
    # Test CNN model
    print("\n1. Testing CNN model...")
    cnn_model = CNNChannelEstimator(input_channels=input_channels)
    cnn_out = cnn_model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {cnn_out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in cnn_model.parameters()):,}")
    
    # Test ResNet model
    print("\n2. Testing ResNet model...")
    resnet_model = ResNetChannelEstimator(input_channels=input_channels)
    resnet_out = resnet_model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {resnet_out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in resnet_model.parameters()):,}")
    
    # Test Hybrid model
    print("\n3. Testing Hybrid CNN-LSTM model...")
    hybrid_model = HybridCNNLSTMEstimator(input_channels=input_channels)
    hybrid_out = hybrid_model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {hybrid_out.shape}")
    print(f"   Parameters: {sum(p.numel() for p in hybrid_model.parameters()):,}")
    
    # Test loss
    print("\n4. Testing loss function...")
    loss_fn = ChannelEstimationLoss(loss_type='mse')
    target = torch.randn_like(cnn_out)
    loss = loss_fn(cnn_out, target)
    print(f"   Loss value: {loss.item():.6f}")
    
    print("\nAll model tests passed!")
