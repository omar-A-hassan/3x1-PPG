"""
Models package for PPG-based glucose prediction.
"""

from .cnn_module import CNNFeatureExtractor
from .gru_module import GRUTemporalModel
from .cnn_gru import CNNGRU, build_cnn_gru

# Optional: xLSTM-based model
try:
    from .xlstm_ppg import xLSTMPPGRegressor, build_xlstm_ppg
    XLSTM_AVAILABLE = True
except ImportError:
    XLSTM_AVAILABLE = False
    xLSTMPPGRegressor = None
    build_xlstm_ppg = None

__all__ = [
    'CNNFeatureExtractor',
    'GRUTemporalModel',
    'CNNGRU',
    'build_cnn_gru',
    'xLSTMPPGRegressor',
    'build_xlstm_ppg',
    'XLSTM_AVAILABLE',
]
