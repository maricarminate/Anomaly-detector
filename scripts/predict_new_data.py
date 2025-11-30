#!/usr/bin/env python3
"""
Script: Predizer Anomalias em Novos Dados
Usa modelos treinados para detectar anomalias

Execute: python scripts/predict_new_data.py [--input data.csv] [--output results.csv]
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import argparse
from glob import glob
from datetime import datetime

from src.models import BaseAnomalyDetector
from src.utils import get_logger, MODELS_DIR, REPORTS_DIR


def load_ensemble_model():
    """Carrega o modelo de ensemble mais recente"""
    ensemble_files = sorted(glob(str(MODELS_DIR / "ensemble_*.pkl")))
    
    if not ensemble_files:
        raise FileNotFoundError("Nenhum ensemble encontrado. Execute train_pipeline.py primeiro.")
    
    latest = ensemble_files[-1]
    return BaseAnomalyDetector.load(latest), latest


def predict_on_data(data_path, output_path=None, use_ensemble=True):
    """
    Prediz anomalias em novos dados
    
    Args:
        data_path: Caminho para arquivo CSV
        output_path: Caminho para salvar resultados
        use_ensemble: Se True, usa ensemble; se False, usa todos os modelos
    """
    logger = get_logger()
    
    logger.info("=" * 70)
    logger.info("PREDIÇÃO EM NOVOS DADOS")
    logger.info("=" * 70)
    
    # 1. Carregar dados
    logger.info(f"\n1. Carregando dados: {data_path}")
    try:
        data = pd.read_csv(data_path, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(data_path, nrows=1).columns else None)
        logger.info(f"✓ {len(data)} pontos carregados")
    except Exception as e:
        logger.error(f"✗ Erro ao carregar dados: {e}")
        return None
    
    # 2. Carregar modelo
    logger.info(f"\n2. Carregando modelo (ensemble={use_ensemble})")
    
    if use_ensemble:
        try:
            model, model_path = load_ensemble_model()
            logger.info(f"✓ Ensemble carregado: {Path(model_path).name}")
        except Exception as e:
            logger.error(f"✗ Erro ao carregar ensemble: {e}")
            return None
        
        # Predizer
        logger.info("\n3. Fazendo predições...")
        predictions = model.predict(data)
        
        results = data.copy()
        results['anomaly'] = predictions
        results['detector'] = 'ensemble'
        
    else:
        # Carregar todos os modelos
        model_files = sorted(glob(str(MODELS_DIR / "*.pkl")))
        model_files = [f for f in model_files if 'ensemble' not in f]
        
        if not model_files:
            logger.error("✗ Nenhum modelo encontrado")
            return None
        
        logger.info(f"✓ {len(model_files)} modelos carregados")
        
        # Predizer com todos
        logger.info("\n3. Fazendo predições com múltiplos modelos...")
        results = data.copy()
        
        for model_file in model_files:
            name = Path(model_file).stem.rsplit('_', 2)[0]
            logger.info(f"  - {name}...")
            
            try:
                model = BaseAnomalyDetector.load(model_file)
                preds = model.predict(data)
                results[f'{name}_pred'] = preds
            except Exception as e:
                logger.warning(f"    Erro: {e}")
                continue
        
        # Majority vote
        pred_cols = [c for c in results.columns if c.endswith('_pred')]
        if pred_cols:
            results['anomaly'] = (results[pred_cols].sum(axis=1) >= len(pred_cols) / 2).astype(int)
    
    # 4. Estatísticas
    n_anomalies = results['anomaly'].sum()
    pct = 100 * n_anomalies / len(results)
    
    logger.info(f"\n4. RESULTADOS:")
    logger.info(f"  Total de pontos: {len(results)}")
    logger.info(f"  Anomalias detectadas: {n_anomalies} ({pct:.2f}%)")
    
    # 5. Salvar resultados
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f'predictions_{timestamp}.csv'
    
    results.to_csv(output_path, index=False)
    logger.info(f"\n✓ Resultados salvos: {output_path}")
    
    return results


def main():
    """Main function com argumentos de linha de comando"""
    
    parser = argparse.ArgumentParser(
        description='Prediz anomalias em novos dados usando modelos treinados'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/processed/nyc_taxi_test.csv',
        help='Caminho para arquivo CSV de entrada'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Caminho para salvar resultados (opcional)'
    )
    parser.add_argument(
        '--ensemble',
        action='store_true',
        default=True,
        help='Usar ensemble (padrão: True)'
    )
    parser.add_argument(
        '--all-models',
        action='store_true',
        help='Usar todos os modelos individualmente'
    )
    
    args = parser.parse_args()
    
    # Determinar se usa ensemble ou todos os modelos
    use_ensemble = not args.all_models
    
    # Executar predição
    results = predict_on_data(
        data_path=args.input,
        output_path=args.output,
        use_ensemble=use_ensemble
    )
    
    if results is not None:
        print("\n✅ Predição concluída com sucesso!")
        print(f"\n📊 Anomalias detectadas: {results['anomaly'].sum()} / {len(results)}")
        print(f"📁 Resultados salvos em: {args.output or 'outputs/reports/'}")
    else:
        print("\n❌ Erro na predição")
        sys.exit(1)


if __name__ == '__main__':
    main()