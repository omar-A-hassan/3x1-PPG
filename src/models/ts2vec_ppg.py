"""
TSEncoder-based PPG Glucose Prediction Model

End-to-End Supervised Training:
- TSEncoder (Dilated Convolutions) + Regression Head
- Trained directly with MSE loss on glucose labels
- No pre-training stage required
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add ts2vec to path
TS2VEC_PATH = Path(__file__).parent.parent.parent / 'ts2vec'
sys.path.insert(0, str(TS2VEC_PATH))

try:
    from models.encoder import TSEncoder
    TS2VEC_AVAILABLE = True
except ImportError as e:
    TS2VEC_AVAILABLE = False
    print(f"WARNING: TSEncoder not available: {e}")


class TSEncoderPPGRegressor(nn.Module):
    """End-to-end TSEncoder model for PPG glucose prediction."""

    def __init__(self, input_dims=1, output_dims=320, hidden_dims=64, depth=10, dropout=0.1):
        super().__init__()
        
        if not TS2VEC_AVAILABLE:
            raise ImportError("TSEncoder not available. Please clone ts2vec repository.")
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        
        # TSEncoder (dilated convolutions)
        self.encoder = TSEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            hidden_dims=hidden_dims,
            depth=depth,
            mask_mode='binomial'
        )
        
        # Regression head
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
        """Forward pass."""
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # TSEncoder encoding
        x = self.encoder(x, mask=mask)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Regression
        glucose = self.regression_head(x)
        return glucose
    
    def count_parameters(self):
        """Count trainable parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total_params,
            'trainable': trainable_params,
            'encoder': sum(p.numel() for p in self.encoder.parameters()),
            'regression_head': sum(p.numel() for p in self.regression_head.parameters())
        }


def build_tsencoder_ppg(input_dims=1, output_dims=320, hidden_dims=64, depth=10, dropout=0.1):
    """Build TSEncoder PPG regression model for end-to-end training."""
    return TSEncoderPPGRegressor(input_dims, output_dims, hidden_dims, depth, dropout)


if __name__ == "__main__":
    print("Testing TSEncoder PPG Model...")
    if not TS2VEC_AVAILABLE:
        print("ERROR: TSEncoder not available.")
        sys.exit(1)
    
    model = build_tsencoder_ppg()
    print(f"Model: {model.count_parameters()}")
    
    x = torch.randn(32, 100)
    y = model(x)
    assert y.shape == (32, 1)
    print("✅ Test passed!")
