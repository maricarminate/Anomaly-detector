"""
Módulo de Métricas de Avaliação
Calcula métricas de performance dos detectores
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, auc,
    precision_recall_curve, roc_curve
)


class MetricsCalculator:
    """Calcula métricas de performance para detectores de anomalia"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate(self, y_true, y_pred, y_scores=None, name="Model"):
        """
        Calcula múltiplas métricas
        
        Args:
            y_true: Labels verdadeiros (0 ou 1)
            y_pred: Predições (0 ou 1)
            y_scores: Scores contínuos (opcional)
            name: Nome do modelo
        """
        
        # Remover NaN
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if y_scores is not None:
            y_scores = y_scores[mask]
        
        metrics = {
            'name': name,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'support': int(np.sum(y_true))
        }
        
        # Matriz de confusão
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            metrics['tn'] = int(cm[0, 0])
            metrics['fp'] = int(cm[0, 1])
            metrics['fn'] = int(cm[1, 0])
            metrics['tp'] = int(cm[1, 1])
            
            # Métricas adicionais
            metrics['specificity'] = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            metrics['sensitivity'] = metrics['recall']  # Mesmo que recall
        
        # ROC-AUC se houver scores
        if y_scores is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            except:
                metrics['roc_auc'] = None
        
        self.metrics[name] = metrics
        return metrics
    
    def compare(self):
        """Retorna DataFrame comparando todos os modelos"""
        import pandas as pd
        return pd.DataFrame(self.metrics).T
    
    def get_best(self, metric='f1'):
        """Retorna o melhor modelo segundo uma métrica"""
        best_name = max(self.metrics.keys(), 
                       key=lambda x: self.metrics[x].get(metric, 0))
        return best_name, self.metrics[best_name]
    
    def summary(self):
        """Mostra resumo de todas as métricas"""
        df = self.compare()
        return df[['precision', 'recall', 'f1', 'specificity', 'support']]


class ThresholdAnalyzer:
    """Analisa diferentes thresholds para otimizar performance"""
    
    def __init__(self):
        self.results = []
    
    def analyze(self, y_true, y_scores, thresholds=None):
        """
        Analisa performance em diferentes thresholds
        
        Args:
            y_true: Labels verdadeiros
            y_scores: Scores contínuos
            thresholds: Lista de thresholds a testar
        """
        if thresholds is None:
            thresholds = np.arange(0, 1.01, 0.05)
        
        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)
            
            if np.sum(y_pred) > 0:
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
            else:
                precision = recall = f1 = 0
            
            self.results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'anomalies': np.sum(y_pred)
            })
        
        return self.results
    
    def get_optimal_threshold(self, metric='f1'):
        """Retorna threshold ótimo para uma métrica"""
        if not self.results:
            return None
        
        best = max(self.results, key=lambda x: x.get(metric, 0))
        return best['threshold'], best
    
    def summary(self):
        """Retorna DataFrame com resultados"""
        import pandas as pd
        return pd.DataFrame(self.results)


def calculate_anomaly_percentage(predictions):
    """Calcula percentual de anomalias"""
    return 100 * np.sum(predictions) / len(predictions)


def calculate_ensemble_metrics(all_predictions, y_true=None):
    """
    Calcula métricas de ensemble
    
    Args:
        all_predictions: Array (n_samples, n_models)
        y_true: Labels verdadeiros (opcional)
    """
    
    # Votação por maioria
    n_models = all_predictions.shape[1]
    majority_threshold = n_models / 2
    majority_vote = (np.sum(all_predictions, axis=1) >= majority_threshold).astype(int)
    
    # Consenso
    consensus = (np.sum(all_predictions, axis=1) == n_models).astype(int)
    
    # Distribuição de votação
    vote_counts = np.sum(all_predictions, axis=1)
    
    metrics = {
        'majority_vote_anomalies': np.sum(majority_vote),
        'consensus_anomalies': np.sum(consensus),
        'majority_vote_pct': calculate_anomaly_percentage(majority_vote),
        'consensus_pct': calculate_anomaly_percentage(consensus),
        'average_votes': np.mean(vote_counts)
    }
    
    if y_true is not None:
        metrics['majority_vote_f1'] = f1_score(y_true, majority_vote, zero_division=0)
        metrics['consensus_f1'] = f1_score(y_true, consensus, zero_division=0)
    
    return metrics