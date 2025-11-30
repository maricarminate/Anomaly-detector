#!/usr/bin/env python3
"""
Script: Treinar Pipeline Completo
Treina todos os detectores e cria ensemble

Execute: python scripts/train_pipeline.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

from src.data import DataLoader, DataPreprocessor
from src.models import (
    ZScoreDetector, IQRDetector, MovingAverageDetector, EWMADetector,
    IsolationForestDetector, LOFDetector, DBSCANDetector,
    EnsembleDetector
)
from src.evaluation import MetricsCalculator, calculate_ensemble_metrics
from src.utils import get_logger, print_config, MODELS_DIR, REPORTS_DIR


def main():
    """Pipeline completo de treinamento"""
    
    # Setup
    logger = get_logger(log_file=REPORTS_DIR / 'train_pipeline.log')
    logger.info("=" * 70)
    logger.info("PIPELINE DE TREINAMENTO - INÍCIO")
    logger.info("=" * 70)
    
    # Configuração
    print_config()
    
    # 1. CARREGAR DADOS
    logger.info("\n1. CARREGANDO DADOS")
    loader = DataLoader()
    
    try:
        data = loader.load_csv('data/processed/nyc_taxi_clean.csv', parse_dates=['timestamp'])
    except:
        logger.warning("Arquivo não encontrado. Carregando dados demo...")
        data = loader.load_nyc_taxi_demo()
    
    train, test = loader.split_train_test(train_size=0.8)
    train = train.head(2000)  # Limitar treino
    
    logger.info(f"Treino: {len(train)} pontos")
    logger.info(f"Teste: {len(test)} pontos")
    
    # 2. PRÉ-PROCESSAMENTO
    logger.info("\n2. PRÉ-PROCESSAMENTO")
    preprocessor = DataPreprocessor()
    train = preprocessor.handle_missing_values(train, method='forward_fill')
    test = preprocessor.handle_missing_values(test, method='forward_fill')
    logger.info("✓ Missing values tratados")
    
    # 3. CRIAR DETECTORES
    logger.info("\n3. CRIANDO DETECTORES")
    
    detectors = {
        'zscore': ZScoreDetector(threshold=3),
        'iqr': IQRDetector(multiplier=1.5),
        'moving_average': MovingAverageDetector(window=24, threshold=2),
        'ewma': EWMADetector(span=24, threshold=2),
        'isolation_forest': IsolationForestDetector(contamination=0.05, n_estimators=100),
        'lof': LOFDetector(n_neighbors=20, contamination=0.05),
        'dbscan': DBSCANDetector(eps=1.5, min_samples=5)
    }
    
    logger.info(f"Total de detectores: {len(detectors)}")
    
    # 4. TREINAR DETECTORES
    logger.info("\n4. TREINANDO DETECTORES")
    
    predictions = {}
    scores = {}
    
    for name, detector in detectors.items():
        logger.info(f"  Treinando {name}...")
        try:
            detector.fit(train)
            preds = detector.predict(test)
            predictions[name] = preds
            scores[name] = detector.get_scores(test)
            
            n_anomalies = np.sum(preds)
            pct = 100 * n_anomalies / len(preds)
            logger.info(f"    ✓ {n_anomalies} anomalias ({pct:.2f}%)")
        except Exception as e:
            logger.error(f"    ✗ Erro: {e}")
            continue
    
    # 5. CRIAR ENSEMBLE
    logger.info("\n5. CRIANDO ENSEMBLE")
    
    # Alinhar predições
    min_length = min(len(p) for p in predictions.values())
    aligned_predictions = {k: v[:min_length] for k, v in predictions.items()}
    
    # Criar matriz de predições
    pred_matrix = np.column_stack([aligned_predictions[k] for k in sorted(aligned_predictions.keys())])
    
    # Calcular ensemble
    ensemble_metrics = calculate_ensemble_metrics(pred_matrix)
    
    logger.info(f"Majority Vote: {ensemble_metrics['majority_vote_anomalies']} anomalias")
    logger.info(f"Consenso: {ensemble_metrics['consensus_anomalies']} anomalias")
    
    # Criar ensemble detector
    ensemble = EnsembleDetector(
        detectors=list(detectors.values()),
        strategy='majority_vote'
    )
    ensemble.is_fitted = True
    
    # 6. SALVAR MODELOS
    logger.info("\n6. SALVANDO MODELOS")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for name, detector in detectors.items():
        filepath = MODELS_DIR / f"{name}_{timestamp}.pkl"
        detector.save(filepath)
    
    ensemble_path = MODELS_DIR / f"ensemble_{timestamp}.pkl"
    ensemble.save(ensemble_path)
    logger.info(f"✓ Ensemble salvo: {ensemble_path}")
    
    # 7. SALVAR RESULTADOS
    logger.info("\n7. SALVANDO RESULTADOS")
    
    results_df = test.iloc[:min_length].copy()
    for name, preds in aligned_predictions.items():
        results_df[f'{name}_pred'] = preds
    
    results_df['majority_vote'] = (np.sum(pred_matrix, axis=1) >= len(detectors) / 2).astype(int)
    results_df['consensus'] = (np.sum(pred_matrix, axis=1) == len(detectors)).astype(int)
    
    results_path = REPORTS_DIR / f'pipeline_results_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"✓ Resultados salvos: {results_path}")
    
    # 8. SUMÁRIO
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE DE TREINAMENTO - CONCLUÍDO")
    logger.info("=" * 70)
    
    summary = {
        'timestamp': timestamp,
        'n_detectors': len(detectors),
        'n_train': len(train),
        'n_test': len(test),
        'ensemble_metrics': ensemble_metrics,
        'models_dir': str(MODELS_DIR),
        'results_file': str(results_path)
    }
    
    summary_path = REPORTS_DIR / f'pipeline_summary_{timestamp}.pkl'
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    logger.info(f"\n📊 RESUMO:")
    logger.info(f"  Detectores treinados: {len(detectors)}")
    logger.info(f"  Modelos salvos: {MODELS_DIR}")
    logger.info(f"  Resultados salvos: {results_path}")
    logger.info(f"  Sumário salvo: {summary_path}")
    
    print("\n✅ Pipeline concluído com sucesso!")
    print(f"📁 Modelos: {MODELS_DIR}")
    print(f"📊 Resultados: {results_path}")


if __name__ == '__main__':
    main()