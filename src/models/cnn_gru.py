"""
CNN-GRU Hybrid Model for Blood Glucose Prediction from PPG Signals

Combines CNN feature extraction with GRU temporal modeling for accurate
glucose level prediction from PPG waveforms.

Based on: "Non-Invasive Glucose Level Monitoring from PPG using a
Hybrid CNN-GRU Deep Learning Network" (2024)

Reported Performance:
- MAE: 2.96 mg/dL
- MAPE: 2.40%
- R²: 0.97
- RMSE: 3.94 mg/dL
"""

import torch
import torch.nn as nn
from .cnn_module import CNNFeatureExtractor
from .gru_module import GRUTemporalModel


class CNNGRU(nn.Module):
    """
    Complete CNN-GRU model for glucose prediction from PPG signals.

    Architecture Flow:
    PPG Signal → CNN Feature Extraction → GRU Temporal Modeling → Dense Layers → Glucose Prediction

    Args:
        input_length (int): Length of input PPG signal (e.g., 100 for 1-second at 100Hz)
        input_channels (int): Number of input channels (default: 1)
        dropout (float): Dropout rate for regularization (default: 0.3)
    """

    def __init__(self, input_length=100, input_channels=1, dropout=0.5):
        super(CNNGRU, self).__init__()

        self.input_length = input_length
        self.input_channels = input_channels

        # CNN module for spatial feature extraction
        self.cnn = CNNFeatureExtractor(input_channels=input_channels, dropout=dropout)

        # GRU module for temporal modeling with dropout
        # Input size = CNN output channels (256)
        self.gru = GRUTemporalModel(input_size=self.cnn.output_channels, dropout=dropout)

        # Dense layers for final prediction
        self.fc1 = nn.Linear(self.gru.output_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input PPG signal
               Shape: (batch_size, input_length) or (batch_size, 1, input_length)

        Returns:
            Glucose prediction (batch_size, 1)
        """
        # Ensure input is 3D: (batch_size, channels, length)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, length)

        # CNN feature extraction
        x = self.cnn(x)  # (batch, 256, length//8)

        # Prepare for GRU: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # (batch, length//8, 256)

        # GRU temporal modeling
        x = self.gru(x)  # (batch, 64)

        # Dense prediction layers
        x = self.fc1(x)    # (batch, 32)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)    # (batch, 1)

        return x

    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_cnn_gru(input_length=100, dropout=0.5):
    """
    Factory function to build CNN-GRU model with specified parameters.

    Args:
        input_length (int): Length of input PPG signal
        dropout (float): Dropout rate

    Returns:
        CNNGRU model instance
    """
    model = CNNGRU(input_length=input_length, input_channels=1, dropout=dropout)
    return model


if __name__ == "__main__":
    print("="*60)
    print("CNN-GRU Model for Glucose Prediction")
    print("="*60)

    # Build model
    model = build_cnn_gru(input_length=100, dropout=0.3)

    # Model info
    print(f"\nModel Parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 8
    input_length = 100

    # Test with 2D input (batch, length)
    dummy_input_2d = torch.randn(batch_size, input_length)
    output_2d = model(dummy_input_2d)
    print(f"\n2D Input shape: {dummy_input_2d.shape}")
    print(f"Output shape: {output_2d.shape}")

    # Test with 3D input (batch, channels, length)
    dummy_input_3d = torch.randn(batch_size, 1, input_length)
    output_3d = model(dummy_input_3d)
    print(f"\n3D Input shape: {dummy_input_3d.shape}")
    print(f"Output shape: {output_3d.shape}")

    # Show sample prediction
    print(f"\nSample glucose predictions (mg/dL):")
    print(output_3d.squeeze().detach().numpy()[:5])

    print("\nCNN-GRU model working correctly")
    print("="*60)
