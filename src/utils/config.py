"""
Módulo de Configuração
Parâmetros centralizados do projeto
"""

from pathlib import Path
import json

# Caminhos
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
DATA_RAW_DIR = DATA_DIR / 'raw'
DATA_PROCESSED_DIR = DATA_DIR / 'processed'
OUTPUTS_DIR = BASE_DIR / 'outputs'
PLOTS_DIR = OUTPUTS_DIR / 'plots'
MODELS_DIR = OUTPUTS_DIR / 'models'
REPORTS_DIR = OUTPUTS_DIR / 'reports'

# Criar diretórios se não existirem
for dir_path in [DATA_RAW_DIR, DATA_PROCESSED_DIR, PLOTS_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONFIGURAÇÃO DE DADOS
# ============================================================================

DATA_CONFIG = {
    'train_file': str(DATA_PROCESSED_DIR / 'nyc_taxi_train.csv'),
    'test_file': str(DATA_PROCESSED_DIR / 'nyc_taxi_test.csv'),
    'train_size': 2000,  # Primeiros N pontos de treino
    'test_size': None,   # Usar todo o teste
    'date_column': 'timestamp',
    'value_column': 'value'
}


# ============================================================================
# CONFIGURAÇÃO DE DETECTORES
# ============================================================================

DETECTOR_CONFIG = {
    'zscore': {
        'enabled': True,
        'threshold': 3,
        'description': 'Z-Score (desvios padrão)'
    },
    'iqr': {
        'enabled': True,
        'multiplier': 1.5,
        'description': 'Interquartile Range'
    },
    'isolation_forest': {
        'enabled': True,
        'contamination': 0.05,
        'n_estimators': 100,
        'window_size': 24,
        'description': 'Isolation Forest (árvores aleatórias)'
    },
    'lof': {
        'enabled': True,
        'n_neighbors': 20,
        'contamination': 0.05,
        'window_size': 24,
        'description': 'Local Outlier Factor (densidade local)'
    },
    'dbscan': {
        'enabled': True,
        'eps': 1.5,
        'min_samples': 5,
        'window_size': 24,
        'description': 'DBSCAN (clustering por densidade)'
    }
}


# ============================================================================
# CONFIGURAÇÃO DE ENSEMBLE
# ============================================================================

ENSEMBLE_CONFIG = {
    'strategy': 'majority_vote',  # 'majority_vote', 'weighted', 'consensus'
    'majority_threshold': 3,  # >= N modelos concordam
    'weights': {
        'zscore': 1.0,
        'iqr': 1.0,
        'isolation_forest': 1.3,
        'lof': 1.3,
        'dbscan': 1.0
    },
    'description': 'Ensemble com votação ponderada'
}


# ============================================================================
# CONFIGURAÇÃO DE VISUALIZAÇÃO
# ============================================================================

PLOT_CONFIG = {
    'style': 'whitegrid',
    'figure_size': (16, 6),
    'dpi': 300,
    'font_size': 12,
    'colors': {
        'normal': 'steelblue',
        'anomaly': 'red',
        'threshold': 'orange',
        'background': 'white'
    }
}


# ============================================================================
# CONFIGURAÇÃO DE MÉTRICAS
# ============================================================================

METRICS_CONFIG = {
    'metrics': ['precision', 'recall', 'f1', 'specificity', 'roc_auc'],
    'threshold_analysis': True,
    'threshold_steps': 0.05,
    'save_confusion_matrix': True
}


# ============================================================================
# CONFIGURAÇÃO DE TREINAMENTO
# ============================================================================

TRAINING_CONFIG = {
    'random_seed': 42,
    'test_size': 0.2,
    'verbose': True,
    'save_models': True,
    'save_results': True
}


# ============================================================================
# CONFIGURAÇÃO DE LOGS
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': str(REPORTS_DIR / 'pipeline.log')
}


def get_config(section):
    """Retorna configuração de uma seção"""
    configs = {
        'data': DATA_CONFIG,
        'detector': DETECTOR_CONFIG,
        'ensemble': ENSEMBLE_CONFIG,
        'plot': PLOT_CONFIG,
        'metrics': METRICS_CONFIG,
        'training': TRAINING_CONFIG,
        'logging': LOGGING_CONFIG
    }
    return configs.get(section, {})


def print_config():
    """Mostra todas as configurações"""
    print("=" * 70)
    print("CONFIGURAÇÕES DO PROJETO")
    print("=" * 70)
    
    print("\n📁 CAMINHOS:")
    print(f"  Base: {BASE_DIR}")
    print(f"  Dados: {DATA_PROCESSED_DIR}")
    print(f"  Modelos: {MODELS_DIR}")
    print(f"  Relatórios: {REPORTS_DIR}")
    
    print("\n🔧 DETECTORES HABILITADOS:")
    for name, config in DETECTOR_CONFIG.items():
        if config.get('enabled'):
            print(f"  ✓ {name}: {config['description']}")
    
    print("\n🎯 ENSEMBLE:")
    print(f"  Estratégia: {ENSEMBLE_CONFIG['strategy']}")
    print(f"  Threshold: >= {ENSEMBLE_CONFIG['majority_threshold']} modelos")
    
    print("\n📊 MÉTRICAS:")
    print(f"  {', '.join(METRICS_CONFIG['metrics'])}")


def save_config(filepath):
    """Salva configuração em JSON"""
    config_dict = {
        'data': DATA_CONFIG,
        'detector': DETECTOR_CONFIG,
        'ensemble': ENSEMBLE_CONFIG,
        'plot': PLOT_CONFIG,
        'metrics': METRICS_CONFIG,
        'training': TRAINING_CONFIG
    }
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def load_config(filepath):
    """Carrega configuração de JSON"""
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == '__main__':
    print_config()