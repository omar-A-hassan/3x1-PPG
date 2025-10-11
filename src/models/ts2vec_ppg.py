"""
TS2Vec-based PPG Glucose Prediction Model

This module adapts TS2Vec (Time Series to Vector) for blood glucose prediction from PPG signals.

Two-Stage Training:
1. Stage 1 (Self-Supervised Pre-training): Train TS2Vec encoder on unlabeled PPG data using contrastive learning
2. Stage 2 (Supervised Fine-tuning): Freeze encoder, add regression head, fine-tune on labeled glucose data

Based on:
- TS2Vec: https://github.com/zhihanyue/ts2vec
- Paper: "Non-invasive blood glucose monitoring using PPG signals" (2024)

Architecture:
    Input PPG (batch, 100, 1)
         ↓
    TS2Vec Encoder (Dilated Conv blocks)
         ↓
    Temporal Pooling
         ↓
    Regression Head
         ↓
    Glucose Prediction (batch, 1)
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add ts2vec to path
TS2VEC_PATH = Path(__file__).parent.parent.parent / 'ts2vec'
sys.path.insert(0, str(TS2VEC_PATH))

try:
    from ts2vec import TS2Vec
    from models.encoder import TSEncoder
    TS2VEC_AVAILABLE = True
except ImportError as e:
    TS2VEC_AVAILABLE = False
    print(f"WARNING: TS2Vec not available: {e}")
    print("Please ensure ts2vec repository is cloned in the project root")


class TS2VecPPGRegressor(nn.Module):
    """
    TS2Vec-based model for PPG glucose prediction.

    This model uses a pre-trained TS2Vec encoder with a regression head
    for blood glucose level estimation from PPG signals.

    Args:
        input_dims: Number of input features (1 for PPG)
        output_dims: Representation dimension from TS2Vec encoder (default: 320)
        hidden_dims: Hidden dimension in encoder (default: 64)
        depth: Number of dilated conv blocks (default: 10)
        dropout: Dropout rate (default: 0.1)
        freeze_encoder: Whether to freeze encoder during fine-tuning (default: True)
    """

    def __init__(
        self,
        input_dims=1,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        dropout=0.1,
        freeze_encoder=True
    ):
        super().__init__()

        if not TS2VEC_AVAILABLE:
            raise ImportError("TS2Vec not available. Please clone ts2vec repository.")

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        self.freeze_encoder = freeze_encoder

        # TS2Vec encoder (dilated convolutions)
        self.encoder = TSEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            mask_mode='binomial'
        )

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Regression head for glucose prediction
        self.regression_head = nn.Sequential(
            nn.Linear(output_dims, output_dims // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dims // 2, output_dims // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dims // 4, 1)
        )

    def forward(self, x, mask=None):
        """
        Forward pass.

        Args:
            x: PPG signal (batch_size, seq_length) or (batch_size, seq_length, 1)
            mask: Optional mask for encoder

        Returns:
            glucose: Predicted glucose value (batch_size, 1)
        """
        batch_size = x.shape[0]

        # Ensure input is 3D: (batch, seq_length, input_dims)
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_length, 1)

        # TS2Vec encoding: (batch, seq_length, output_dims)
        x = self.encoder(x, mask=mask)

        # Global average pooling over time dimension
        # (batch, seq_length, output_dims) -> (batch, output_dims)
        x = torch.mean(x, dim=1)

        # Regression head: (batch, output_dims) -> (batch, 1)
        glucose = self.regression_head(x)

        return glucose

    def unfreeze_encoder(self):
        """Unfreeze encoder parameters for fine-tuning."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.freeze_encoder = False

    def freeze_encoder_params(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.freeze_encoder = True

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TS2VecPPGTrainer:
    """
    Two-stage trainer for TS2Vec PPG glucose prediction.

    Stage 1: Pre-train TS2Vec encoder with contrastive learning (self-supervised)
    Stage 2: Fine-tune with regression head on labeled glucose data (supervised)
    """

    def __init__(
        self,
        input_dims=1,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda'
    ):
        if not TS2VEC_AVAILABLE:
            raise ImportError("TS2Vec not available")

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        self.device = device

        # TS2Vec for pre-training
        self.ts2vec = None

        # Regression model for fine-tuning
        self.model = None

    def pretrain(
        self,
        train_data,
        n_epochs=None,
        n_iters=None,
        batch_size=16,
        lr=0.001,
        verbose=True
    ):
        """
        Stage 1: Pre-train TS2Vec encoder with contrastive learning.

        Args:
            train_data: Unlabeled PPG data (n_samples, seq_length, input_dims)
            n_epochs: Number of epochs
            n_iters: Number of iterations (alternative to n_epochs)
            batch_size: Batch size
            lr: Learning rate
            verbose: Print progress

        Returns:
            loss_log: Training loss history
        """
        print("="*60)
        print("Stage 1: Pre-training TS2Vec Encoder (Self-Supervised)")
        print("="*60)

        # Initialize TS2Vec
        self.ts2vec = TS2Vec(
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            device=self.device,
            lr=lr,
            batch_size=batch_size
        )

        # Pre-train with contrastive learning
        loss_log = self.ts2vec.fit(
            train_data,
            n_epochs=n_epochs,
            n_iters=n_iters,
            verbose=verbose
        )

        print(f"\nPre-training completed! Final loss: {loss_log[-1]:.4f}")
        return loss_log

    def build_regression_model(self, freeze_encoder=True, dropout=0.1):
        """
        Build regression model with pre-trained encoder.

        Args:
            freeze_encoder: Whether to freeze encoder weights
            dropout: Dropout rate in regression head
        """
        if self.ts2vec is None:
            raise ValueError("Must pre-train TS2Vec first!")

        # Create regression model
        self.model = TS2VecPPGRegressor(
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            dropout=dropout,
            freeze_encoder=freeze_encoder
        ).to(self.device)

        # Load pre-trained encoder weights
        self.model.encoder.load_state_dict(self.ts2vec._net.state_dict())

        print(f"Regression model built with {self.model.count_parameters():,} trainable parameters")
        print(f"Encoder frozen: {freeze_encoder}")

    def save_pretrained(self, path):
        """Save pre-trained TS2Vec encoder."""
        if self.ts2vec is None:
            raise ValueError("No pre-trained model to save")
        self.ts2vec.save(path)
        print(f"Pre-trained encoder saved to {path}")

    def load_pretrained(self, path):
        """Load pre-trained TS2Vec encoder."""
        self.ts2vec = TS2Vec(
            input_dims=self.input_dims,
            output_dims=self.output_dims,
            hidden_dims=self.hidden_dims,
            depth=self.depth,
            device=self.device
        )
        self.ts2vec.load(path)
        print(f"Pre-trained encoder loaded from {path}")


def build_ts2vec_ppg(
    input_dims=1,
    output_dims=320,
    hidden_dims=64,
    depth=10,
    dropout=0.1,
    freeze_encoder=True
):
    """
    Build TS2Vec PPG regression model.

    Configuration options:
    - Tiny: output_dims=128, hidden_dims=32, depth=6
    - Small: output_dims=256, hidden_dims=48, depth=8
    - Base: output_dims=320, hidden_dims=64, depth=10 (default)

    Args:
        input_dims: Number of input features (1 for PPG)
        output_dims: Representation dimension
        hidden_dims: Hidden dimension in encoder
        depth: Number of dilated conv blocks
        dropout: Dropout rate
        freeze_encoder: Whether encoder is frozen

    Returns:
        model: TS2Vec PPG regression model
    """
    model = TS2VecPPGRegressor(
        input_dims=input_dims,
        output_dims=output_dims,
        hidden_dims=hidden_dims,
        depth=depth,
        dropout=dropout,
        freeze_encoder=freeze_encoder
    )

    return model


if __name__ == "__main__":
    print("Testing TS2Vec PPG Model...")

    if not TS2VEC_AVAILABLE:
        print("ERROR: TS2Vec not available. Cannot run tests.")
        sys.exit(1)

    # Create model
    model = build_ts2vec_ppg(
        input_dims=1,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        dropout=0.1,
        freeze_encoder=False
    )

    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")

    # Test forward pass
    batch_size = 32
    seq_length = 100
    x = torch.randn(batch_size, seq_length)
    y = model(x)

    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Expected output shape: ({batch_size}, 1)")

    assert y.shape == (batch_size, 1), "Output shape mismatch!"
    print("\n✅ Model test passed!")
