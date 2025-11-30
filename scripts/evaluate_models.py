#!/usr/bin/env python3
"""
Script: Avaliar Modelos Treinados
Compara performance de todos os detectores

Execute: python scripts/evaluate_models.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob

from src.data import DataLoader
from src.models import BaseAnomalyDetector
from src.evaluation import MetricsCalculator, calculate_ensemble_metrics
from src.utils import get_logger, MODELS_DIR, REPORTS_DIR, PLOTS_DIR


def load_latest_models():
    """Carrega os modelos mais recentes"""
    model_files = sorted(glob(str(MODELS_DIR / "*.pkl")))
    
    if not model_files:
        raise FileNotFoundError("Nenhum modelo encontrado em outputs/models/")
    
    # Agrupar por timestamp (assumindo formato: detector_TIMESTAMP.pkl)
    timestamps = set('_'.join(f.split('_')[-2:]).replace('.pkl', '') 
                    for f in model_files)
    latest_timestamp = sorted(timestamps)[-1]
    
    # Carregar modelos do timestamp mais recente
    models = {}
    for f in model_files:
        if latest_timestamp in f:
            name = Path(f).stem.rsplit('_', 2)[0]  # Remove timestamp
            models[name] = BaseAnomalyDetector.load(f)
    
    return models, latest_timestamp


def main():
    """Avaliação completa dos modelos"""
    
    logger = get_logger(log_file=REPORTS_DIR / 'evaluate_models.log')
    logger.info("=" * 70)
    logger.info("AVALIAÇÃO DE MODELOS")
    logger.info("=" * 70)
    
    # 1. CARREGAR MODELOS
    logger.info("\n1. CARREGANDO MODELOS")
    try:
        models, timestamp = load_latest_models()
        logger.info(f"✓ {len(models)} modelos carregados (timestamp: {timestamp})")
        for name in models.keys():
            logger.info(f"  - {name}")
    except Exception as e:
        logger.error(f"✗ Erro ao carregar modelos: {e}")
        print(f"❌ Erro: {e}")
        print("Execute primeiro: python scripts/train_pipeline.py")
        return
    
    # 2. CARREGAR DADOS DE TESTE
    logger.info("\n2. CARREGANDO DADOS DE TESTE")
    loader = DataLoader()
    
    try:
        data = loader.load_csv('data/processed/nyc_taxi_clean.csv', parse_dates=['timestamp'])
    except:
        data = loader.load_nyc_taxi_demo()
    
    train, test = loader.split_train_test(train_size=0.8)
    logger.info(f"✓ Teste: {len(test)} pontos")
    
    # 3. FAZER PREDIÇÕES
    logger.info("\n3. FAZENDO PREDIÇÕES")
    predictions = {}
    scores = {}
    
    for name, model in models.items():
        if name == 'ensemble':
            continue
        
        logger.info(f"  Predizendo com {name}...")
        try:
            preds = model.predict(test)
            predictions[name] = preds
            scores[name] = model.get_scores(test)
            
            n_anomalies = np.sum(preds)
            pct = 100 * n_anomalies / len(preds)
            logger.info(f"    {n_anomalies} anomalias ({pct:.2f}%)")
        except Exception as e:
            logger.error(f"    ✗ Erro: {e}")
    
    # 4. CALCULAR MÉTRICAS
    logger.info("\n4. CALCULANDO MÉTRICAS")
    
    # Alinhar predições
    min_length = min(len(p) for p in predictions.values())
    aligned_preds = {k: v[:min_length] for k, v in predictions.items()}
    
    # Comparação
    comparison = pd.DataFrame({
        'Detector': list(aligned_preds.keys()),
        'Anomalias': [np.sum(p) for p in aligned_preds.values()],
        'Percentual': [f"{100*np.sum(p)/len(p):.2f}%" for p in aligned_preds.values()]
    })
    
    comparison = comparison.sort_values('Anomalias', ascending=False)
    
    logger.info("\n" + comparison.to_string(index=False))
    
    # 5. ENSEMBLE METRICS
    logger.info("\n5. MÉTRICAS DE ENSEMBLE")
    
    pred_matrix = np.column_stack([aligned_preds[k] for k in sorted(aligned_preds.keys())])
    ensemble_metrics = calculate_ensemble_metrics(pred_matrix)
    
    logger.info(f"Majority Vote: {ensemble_metrics['majority_vote_anomalies']} anomalias")
    logger.info(f"Consenso: {ensemble_metrics['consensus_anomalies']} anomalias")
    logger.info(f"Média de votos: {ensemble_metrics['average_votes']:.2f}")
    
    # 6. VISUALIZAÇÕES
    logger.info("\n6. GERANDO VISUALIZAÇÕES")
    
    # Plot 1: Comparação de detectores
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot
    axes[0].barh(comparison['Detector'], comparison['Anomalias'], 
                color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_title('Comparação de Detectores', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Número de Anomalias')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Heatmap de consenso
    sample_idx = np.arange(0, len(pred_matrix), max(1, len(pred_matrix)//100))
    heatmap_data = pred_matrix[sample_idx]
    
    sns.heatmap(heatmap_data.T, cmap='RdYlGn_r', cbar_kws={'label': 'Anomalia'}, ax=axes[1])
    axes[1].set_title('Consenso entre Detectores', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Índice de Tempo (amostra)')
    axes[1].set_ylabel('Detector')
    axes[1].set_yticklabels(sorted(aligned_preds.keys()), rotation=0)
    
    plt.tight_layout()
    plot_path = PLOTS_DIR / f'evaluation_{timestamp}.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Visualização salva: {plot_path}")
    plt.close()
    
    # 7. SALVAR RELATÓRIO
    logger.info("\n7. SALVANDO RELATÓRIO")
    
    report_path = REPORTS_DIR / f'evaluation_report_{timestamp}.csv'
    comparison.to_csv(report_path, index=False)
    logger.info(f"✓ Relatório salvo: {report_path}")
    
    # 8. SUMÁRIO
    logger.info("\n" + "=" * 70)
    logger.info("AVALIAÇÃO CONCLUÍDA")
    logger.info("=" * 70)
    
    print("\n✅ Avaliação concluída!")
    print(f"\n📊 RESUMO:")
    print(f"  Modelos avaliados: {len(models)}")
    print(f"  Melhor detector: {comparison.iloc[0]['Detector']} ({comparison.iloc[0]['Anomalias']} anomalias)")
    print(f"  Ensemble Majority Vote: {ensemble_metrics['majority_vote_anomalies']} anomalias")
    print(f"  Ensemble Consenso: {ensemble_metrics['consensus_anomalies']} anomalias")
    print(f"\n📁 Arquivos gerados:")
    print(f"  Relatório: {report_path}")
    print(f"  Visualização: {plot_path}")


if __name__ == '__main__':
    main()