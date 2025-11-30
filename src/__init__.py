"""Pacote de Detecção de Anomalias"""

__version__ = '1.0.0'
__author__ = 'Seu Nome'

from .models.anomaly_detector import (
    AnomalyDetector,
    ZScoreDetector,
    IQRDetector,
    IsolationForestDetector,
    LOFDetector,
    DBSCANDetector
)

from .evaluation.metrics import (
    MetricsCalculator,
    ThresholdAnalyzer,
    calculate_anomaly_percentage,
    calculate_ensemble_metrics
)

from .utils.config import (
    get_config,
    print_config,
    save_config,
    load_config,
    BASE_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    PLOTS_DIR
)

__all__ = [
    'AnomalyDetector',
    'ZScoreDetector',
    'IQRDetector',
    'IsolationForestDetector',
    'LOFDetector',
    'DBSCANDetector',
    'MetricsCalculator',
    'ThresholdAnalyzer',
    'calculate_anomaly_percentage',
    'calculate_ensemble_metrics',
    'get_config',
    'print_config',
    'save_config',
    'load_config',
    'BASE_DIR',
    'DATA_PROCESSED_DIR',
    'MODELS_DIR',
    'REPORTS_DIR',
    'PLOTS_DIR'
]