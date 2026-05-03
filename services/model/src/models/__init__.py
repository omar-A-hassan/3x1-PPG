"""
Models package for PPG-based glucose prediction.
"""

# TSEncoder end-to-end model
try:
    from .ts2vec_ppg import (
        TSEncoderPPGRegressor,
        build_tsencoder_ppg,
        TS2VEC_AVAILABLE
    )
except ImportError:
    TS2VEC_AVAILABLE = False
    TSEncoderPPGRegressor = None
    build_tsencoder_ppg = None

__all__ = [
    'TSEncoderPPGRegressor',
    'build_tsencoder_ppg',
    'TS2VEC_AVAILABLE',
]
