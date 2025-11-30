"""Módulo de Detectores de Anomalia"""

from .base import BaseAnomalyDetector, UnsupervisedDetector, SupervisedDetector
from .statistical import ZScoreDetector, IQRDetector, MovingAverageDetector, EWMADetector
from .tree_based import IsolationForestDetector, LOFDetector, DBSCANDetector
from .autoencoder import DenseAutoencoderDetector
from .lstm import LSTMAutoencoderDetector
from .ensemble import EnsembleDetector, VotingEnsemble, WeightedEnsemble, ConsensusEnsemble

__all__ = [
    'BaseAnomalyDetector',
    'UnsupervisedDetector',
    'SupervisedDetector',
    'ZScoreDetector',
    'IQRDetector',
    'MovingAverageDetector',
    'EWMADetector',
    'IsolationForestDetector',
    'LOFDetector',
    'DBSCANDetector',
    'DenseAutoencoderDetector',
    'LSTMAutoencoderDetector',
    'EnsembleDetector',
    'VotingEnsemble',
    'WeightedEnsemble',
    'ConsensusEnsemble'
]