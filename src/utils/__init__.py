"""Módulo de Utilidades e Configuração"""

from .config import (
    get_config,
    print_config,
    save_config,
    load_config,
    BASE_DIR,
    DATA_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    PLOTS_DIR
)

from .logger import get_logger, log_section, log_performance

__all__ = [
    'get_config',
    'print_config',
    'save_config',
    'load_config',
    'get_logger',
    'log_section',
    'log_performance',
    'BASE_DIR',
    'DATA_DIR',
    'DATA_PROCESSED_DIR',
    'MODELS_DIR',
    'REPORTS_DIR',
    'PLOTS_DIR'
]