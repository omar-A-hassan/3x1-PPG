"""
Models package for PPG-based glucose prediction.
"""

# TS2Vec-based model (primary)
try:
    from .ts2vec_ppg import (
        TS2VecPPGRegressor,
        TS2VecPPGTrainer,
        build_ts2vec_ppg,
        TS2VEC_AVAILABLE
    )
except ImportError:
    TS2VEC_AVAILABLE = False
    TS2VecPPGRegressor = None
    TS2VecPPGTrainer = None
    build_ts2vec_ppg = None

__all__ = [
    'TS2VecPPGRegressor',
    'TS2VecPPGTrainer',
    'build_ts2vec_ppg',
    'TS2VEC_AVAILABLE',
]
