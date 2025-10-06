"""
GRU Module for Temporal Dependency Modeling

Implements GRU layers for capturing temporal patterns in PPG feature sequences.
Used as the second stage in the CNN-GRU hybrid architecture.

Based on: "Non-Invasive Glucose Level Monitoring from PPG using a
Hybrid CNN-GRU Deep Learning Network" (2024)
"""

import torch
import torch.nn as nn


class GRUTemporalModel(nn.Module):
    """
    GRU module for modeling temporal dependencies in feature sequences.

    Architecture:
    - GRU Layer 1: input_size → 128 hidden units
    - GRU Layer 2: 128 → 64 hidden units
    - Returns final hidden state (not full sequence)

    Args:
        input_size (int): Size of input features (CNN output channels, typically 256)
        hidden_size_1 (int): Hidden units in first GRU layer (default: 128)
        hidden_size_2 (int): Hidden units in second GRU layer (default: 64)
    """

    def __init__(self, input_size=256, hidden_size_1=64, hidden_size_2=32, dropout=0.5):
        super(GRUTemporalModel, self).__init__()

        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2

        # First GRU layer with dropout
        self.gru1 = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size_1,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False
        )
        self.dropout1 = nn.Dropout(dropout)

        # Second GRU layer with dropout
        self.gru2 = nn.GRU(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=False
        )
        self.dropout2 = nn.Dropout(dropout)

        self.output_size = hidden_size_2

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor (batch_size, sequence_length, input_size)

        Returns:
            Last hidden state (batch_size, hidden_size_2)
        """
        # First GRU layer
        x, _ = self.gru1(x)  # (batch, seq, 64)
        x = self.dropout1(x)

        # Second GRU layer
        x, _ = self.gru2(x)  # (batch, seq, 32)
        x = self.dropout2(x)

        # Return last timestep
        x = x[:, -1, :]      # (batch, 32)

        return x


if __name__ == "__main__":
    print("Testing GRU Temporal Model...")

    # Simulate CNN output
    batch_size = 8
    seq_length = 12  # After CNN pooling: 100 // 8 = 12
    input_size = 256  # CNN output channels

    gru = GRUTemporalModel(input_size=input_size)
    print(f"Parameters: {sum(p.numel() for p in gru.parameters()):,}")

    dummy_input = torch.randn(batch_size, seq_length, input_size)
    output = gru(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output size: {gru.output_size}")

    print("\nGRU module working correctly")
