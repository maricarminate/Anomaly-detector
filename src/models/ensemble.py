"""
Ensemble de Detectores de Anomalia
Combina múltiplos detectores para melhor performance
"""

import numpy as np
import pandas as pd
from .base import BaseAnomalyDetector


class EnsembleDetector(BaseAnomalyDetector):
    """Ensemble de múltiplos detectores"""
    
    def __init__(self, detectors=None, strategy='majority_vote', 
                 weights=None, threshold=None):
        """
        Args:
            detectors: Lista de detectores
            strategy: 'majority_vote', 'weighted', 'consensus'
            weights: Pesos para votação ponderada
            threshold: Threshold para majority vote
        """
        super().__init__(name="Ensemble")
        self.detectors = detectors or []
        self.strategy = strategy
        self.weights = weights
        self.vote_threshold = threshold
    
    def add_detector(self, detector):
        """Adiciona um detector ao ensemble"""
        self.detectors.append(detector)
    
    def fit(self, X_train):
        """Treina todos os detectores"""
        self._validate_input(X_train)
        
        print(f"Treinando {len(self.detectors)} detectores...")
        for detector in self.detectors:
            print(f"  - {detector.name}...")
            detector.fit(X_train)
        
        self.is_fitted = True
        print("✓ Todos os detectores treinados")
    
    def predict(self, X_test):
        """Prediz usando ensemble"""
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        self._validate_input(X_test)
        
        # Coletar predições
        all_predictions = []
        for detector in self.detectors:
            preds = detector.predict(X_test)
            all_predictions.append(preds)
        
        # Alinhar tamanhos
        min_length = min(len(p) for p in all_predictions)
        all_predictions = [p[:min_length] for p in all_predictions]
        all_predictions = np.column_stack(all_predictions)
        
        # Aplicar estratégia
        if self.strategy == 'majority_vote':
            self.predictions = self._majority_vote(all_predictions)
        elif self.strategy == 'weighted':
            self.predictions = self._weighted_vote(all_predictions)
        elif self.strategy == 'consensus':
            self.predictions = self._consensus(all_predictions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        return self.predictions
    
    def get_scores(self, X_test):
        """Retorna scores médios de todos os detectores"""
        self._validate_input(X_test)
        
        all_scores = []
        for detector in self.detectors:
            scores = detector.get_scores(X_test)
            all_scores.append(scores)
        
        # Alinhar e normalizar
        min_length = min(len(s) for s in all_scores)
        all_scores = [s[:min_length] for s in all_scores]
        
        # Média dos scores
        return np.mean(all_scores, axis=0)
    
    def _majority_vote(self, predictions):
        """Votação por maioria"""
        n_models = predictions.shape[1]
        
        if self.vote_threshold is None:
            self.vote_threshold = n_models / 2
        
        votes = np.sum(predictions, axis=1)
        return (votes >= self.vote_threshold).astype(int)
    
    def _weighted_vote(self, predictions):
        """Votação ponderada"""
        if self.weights is None:
            self.weights = np.ones(predictions.shape[1])
        
        weighted_sum = np.average(predictions, axis=1, weights=self.weights)
        return (weighted_sum >= 0.5).astype(int)
    
    def _consensus(self, predictions):
        """Todos os modelos devem concordar"""
        n_models = predictions.shape[1]
        votes = np.sum(predictions, axis=1)
        return (votes == n_models).astype(int)
    
    def summary(self):
        """Resumo do ensemble"""
        base_summary = super().summary()
        base_summary['n_detectors'] = len(self.detectors)
        base_summary['strategy'] = self.strategy
        base_summary['detectors'] = [d.name for d in self.detectors]
        return base_summary
    
    def detailed_summary(self):
        """Resumo detalhado de cada detector"""
        summaries = []
        for detector in self.detectors:
            summaries.append(detector.summary())
        return pd.DataFrame(summaries)


class VotingEnsemble(EnsembleDetector):
    """Ensemble com votação simples"""
    
    def __init__(self, detectors=None, threshold=None):
        super().__init__(
            detectors=detectors,
            strategy='majority_vote',
            threshold=threshold
        )
        self.name = "Voting Ensemble"


class WeightedEnsemble(EnsembleDetector):
    """Ensemble com votação ponderada"""
    
    def __init__(self, detectors=None, weights=None):
        super().__init__(
            detectors=detectors,
            strategy='weighted',
            weights=weights
        )
        self.name = "Weighted Ensemble"


class ConsensusEnsemble(EnsembleDetector):
    """Ensemble que requer consenso total"""
    
    def __init__(self, detectors=None):
        super().__init__(
            detectors=detectors,
            strategy='consensus'
        )
        self.name = "Consensus Ensemble"


def create_ensemble(train_data, test_data, detector_configs):
    """
    Cria e treina ensemble de detectores.
    
    Args:
        train_data: Dados de treino
        test_data: Dados de teste
        detector_configs: Lista de (detector_class, params)
    
    Returns:
        EnsembleDetector treinado
    """
    detectors = []
    
    for detector_class, params in detector_configs:
        detector = detector_class(**params)
        detector.fit(train_data)
        detectors.append(detector)
    
    ensemble = EnsembleDetector(detectors=detectors)
    ensemble.is_fitted = True
    
    return ensemble