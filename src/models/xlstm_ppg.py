"""
xLSTM-based PPG Glucose Prediction Model

Adapted from: https://github.com/NX-AI/xlstm
For 1D PPG signal regression instead of CNN-GRU

Key differences from Vision-LSTM:
- Uses xLSTMBlockStack directly (not Vision-LSTM patches)
- Input: 1D sequence (100 samples) instead of 2D images
- Task: Regression (glucose prediction) instead of classification
"""

import torch
import torch.nn as nn

try:
    from xlstm import (
        xLSTMBlockStack,
        xLSTMBlockStackConfig,
        mLSTMBlockConfig,
        mLSTMLayerConfig,
        sLSTMBlockConfig,
        sLSTMLayerConfig,
        FeedForwardConfig,
    )
    XLSTM_AVAILABLE = True
except ImportError:
    XLSTM_AVAILABLE = False
    print("WARNING: xlstm not installed. Install with: pip install xlstm")


class xLSTMPPGRegressor(nn.Module):
    """
    xLSTM-based model for PPG glucose prediction.

    Architecture:
    1. Input projection: (batch, 100) -> (batch, 100, embedding_dim)
    2. xLSTM blocks: Process sequence with sLSTM + mLSTM
    3. Pooling: Aggregate sequence features
    4. Output head: Regression to glucose value

    Args:
        input_length: Length of PPG signal (default: 100 for 1 second at 100Hz)
        embedding_dim: Dimension of embedded features (default: 64)
        num_blocks: Number of xLSTM blocks (default: 4)
        dropout: Dropout rate (default: 0.3)
        slstm_at: Positions of sLSTM blocks, rest use mLSTM (default: [1])
    """

    def __init__(
        self,
        input_length: int = 100,
        embedding_dim: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.3,
        slstm_at: list = None,
    ):
        super().__init__()

        if not XLSTM_AVAILABLE:
            raise ImportError(
                "xlstm package not available. Install with:\n"
                "pip install xlstm"
            )

        self.input_length = input_length
        self.embedding_dim = embedding_dim
        self.num_blocks = num_blocks

        if slstm_at is None:
            slstm_at = [1]  # Use sLSTM in block 1, mLSTM in others

        # 1. Input projection: Map each time step to embedding dimension
        # (batch, 100) -> (batch, 100, embedding_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )

        # 2. xLSTM block stack configuration
        block_configs = []
        for block_idx in range(num_blocks):
            if block_idx in slstm_at:
                # sLSTM block for state tracking
                block_configs.append(
                    sLSTMBlockConfig(
                        slstm=sLSTMLayerConfig(
                            backend="vanilla",
                            num_heads=4,
                            conv1d_kernel_size=4,
                            bias_init="powerlaw_blockdependent",
                        ),
                        feedforward=FeedForwardConfig(
                            proj_factor=1.3,
                            act_fn="gelu",
                        ),
                    )
                )
            else:
                # mLSTM block for complex patterns
                block_configs.append(
                    mLSTMBlockConfig(
                        mlstm=mLSTMLayerConfig(
                            num_heads=4,
                        ),
                        feedforward=FeedForwardConfig(
                            proj_factor=1.3,
                            act_fn="gelu",
                        ),
                    )
                )

        xlstm_config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(num_heads=4),
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="vanilla",
                    num_heads=4,
                    conv1d_kernel_size=4,
                ),
            ),
            context_length=input_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=slstm_at,
        )

        # 3. xLSTM block stack
        self.xlstm_stack = xLSTMBlockStack(xlstm_config)

        # 4. Sequence pooling (aggregate temporal features)
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # 5. Output head for regression
        self.output_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, 1)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: PPG signal (batch_size, 100)

        Returns:
            glucose: Predicted glucose value (batch_size, 1)
        """
        batch_size = x.shape[0]

        # Reshape: (batch, 100) -> (batch, 100, 1)
        x = x.unsqueeze(-1)

        # Input projection: (batch, 100, 1) -> (batch, 100, embedding_dim)
        x = self.input_projection(x)

        # xLSTM processing: (batch, 100, embedding_dim) -> (batch, 100, embedding_dim)
        x = self.xlstm_stack(x)

        # Pooling: (batch, 100, embedding_dim) -> (batch, embedding_dim, 1)
        x = x.transpose(1, 2)  # (batch, embedding_dim, 100)
        x = self.pooling(x)  # (batch, embedding_dim, 1)
        x = x.squeeze(-1)  # (batch, embedding_dim)

        # Output: (batch, embedding_dim) -> (batch, 1)
        glucose = self.output_head(x)

        return glucose

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_xlstm_ppg(
    input_length: int = 100,
    embedding_dim: int = 64,
    num_blocks: int = 4,
    dropout: float = 0.3,
    slstm_at: list = None,
):
    """
    Build xLSTM model for PPG glucose prediction.

    Configuration options:
    - Tiny: embedding_dim=64, num_blocks=4 (~50K params)
    - Small: embedding_dim=128, num_blocks=6 (~200K params)
    - Base: embedding_dim=192, num_blocks=8 (~500K params)

    Args:
        input_length: Length of input PPG signal
        embedding_dim: Feature dimension
        num_blocks: Number of xLSTM blocks
        dropout: Dropout rate
        slstm_at: Block indices to use sLSTM (rest use mLSTM)

    Returns:
        model: xLSTM model for PPG regression
    """
    if slstm_at is None:
        slstm_at = [1]  # Default: sLSTM at block 1

    model = xLSTMPPGRegressor(
        input_length=input_length,
        embedding_dim=embedding_dim,
        num_blocks=num_blocks,
        dropout=dropout,
        slstm_at=slstm_at,
    )

    return model


if __name__ == "__main__":
    # Test the model
    print("Testing xLSTM PPG Model...")

    # Create model
    model = build_xlstm_ppg(
        input_length=100,
        embedding_dim=64,
        num_blocks=4,
        dropout=0.3,
    )

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, 100)
    y = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected output shape: ({batch_size}, 1)")

    assert y.shape == (batch_size, 1), "Output shape mismatch!"
    print("\n✅ Model test passed!")
