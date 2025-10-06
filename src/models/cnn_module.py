"""
CNN Module for PPG Signal Feature Extraction

Implements convolutional layers for spatial pattern extraction from PPG signals.
Used as the first stage in the CNN-GRU hybrid architecture.

Based on: "Non-Invasive Glucose Level Monitoring from PPG using a
Hybrid CNN-GRU Deep Learning Network" (2024)
"""

import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    CNN module for extracting spatial features from PPG signals.

    Architecture:
    - Conv1D (64 filters) → BatchNorm → ReLU → MaxPool
    - Conv1D (128 filters) → BatchNorm → ReLU → MaxPool
    - Conv1D (256 filters) → BatchNorm → ReLU → MaxPool

    Args:
        input_channels (int): Number of input channels (default: 1 for single PPG signal)
    """

    def __init__(self, input_channels=1):
        super(CNNFeatureExtractor, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.output_channels = 256

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, input_channels, sequence_length)

        Returns:
            Feature tensor (batch_size, 256, sequence_length//8)
        """
        x = self.conv1(x)  # (batch, 64, length//2)
        x = self.conv2(x)  # (batch, 128, length//4)
        x = self.conv3(x)  # (batch, 256, length//8)
        return x

    def get_output_size(self, input_length):
        """Calculate output sequence length after pooling layers"""
        return input_length // 8  # 3 pooling layers with kernel=2


if __name__ == "__main__":
    print("Testing CNN Feature Extractor...")

    cnn = CNNFeatureExtractor(input_channels=1)
    print(f"Parameters: {sum(p.numel() for p in cnn.parameters()):,}")

    # Test with dummy input
    batch_size = 8
    input_length = 100
    dummy_input = torch.randn(batch_size, 1, input_length)
    output = cnn(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output channels: {cnn.output_channels}")
    print(f"Output length: {cnn.get_output_size(input_length)}")

    print("\nCNN module working correctly")
