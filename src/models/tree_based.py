"""
Métodos baseados em Árvores e Distância
Isolation Forest, LOF, DBSCAN
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest as SKIsolationForest
from sklearn.neighbors import LocalOutlierFactor as SKLocalOutlierFactor
from sklearn.cluster import DBSCAN as SKDBSCAN
from sklearn.preprocessing import StandardScaler
from .base import UnsupervisedDetector


class IsolationForestDetector(UnsupervisedDetector):
    """Detector baseado em Isolation Forest"""
    
    def __init__(self, contamination=0.05, n_estimators=100, window_size=24):
        super().__init__(name="Isolation Forest", contamination=contamination)
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = None
    
    def fit(self, X_train):
        """Treina Isolation Forest com features de janela"""
        self._validate_input(X_train)
        
        # Criar features
        features = self._create_features(X_train)
        X_scaled = self.scaler.fit_transform(features)
        
        # Treinar modelo
        self.model = SKIsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42
        )
        self.model.fit(X_scaled)
        self.is_fitted = True
    
    def predict(self, X_test):
        """Detecta anomalias usando Isolation Forest"""
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        self._validate_input(X_test)
        
        # Criar features
        features = self._create_features(X_test)
        X_scaled = self.scaler.transform(features)
        
        # Predizer
        preds = self.model.predict(X_scaled)
        self.predictions = (preds == -1).astype(int)
        
        # Pad com zeros no início
        self.predictions = np.concatenate([
            np.zeros(self.window_size),
            self.predictions
        ])[:len(self._extract_values(X_test))]
        
        self.scores = self.model.score_samples(X_scaled)
        
        return self.predictions
    
    def get_scores(self, X_test):
        """Retorna anomaly scores"""
        self._validate_input(X_test)
        
        features = self._create_features(X_test)
        X_scaled = self.scaler.transform(features)
        
        return self.model.score_samples(X_scaled)
    
    def _create_features(self, data):
        """Cria features com janela deslizante"""
        values = self._extract_values(data)
        
        features = []
        for i in range(self.window_size, len(values)):
            window = values[i - self.window_size:i]
            features.append({
                'mean': np.mean(window),
                'std': np.std(window),
                'min': np.min(window),
                'max': np.max(window),
                'range': np.max(window) - np.min(window),
                'current': values[i]
            })
        
        return pd.DataFrame(features)


class LOFDetector(UnsupervisedDetector):
    """Detector baseado em Local Outlier Factor"""
    
    def __init__(self, n_neighbors=20, contamination=0.05, window_size=24):
        super().__init__(name="Local Outlier Factor", contamination=contamination)
        self.n_neighbors = n_neighbors
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = None
    
    def fit(self, X_train):
        """Treina LOF"""
        self._validate_input(X_train)
        
        # Criar features
        features = self._create_features(X_train)
        X_scaled = self.scaler.fit_transform(features)
        
        # LOF não tem fit separado, mas inicializamos
        self.model = SKLocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination
        )
        self.is_fitted = True
    
    def predict(self, X_test):
        """Detecta anomalias usando LOF"""
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        self._validate_input(X_test)
        
        # Criar features
        features = self._create_features(X_test)
        X_scaled = self.scaler.transform(features)
        
        # Predizer
        preds = self.model.fit_predict(X_scaled)
        self.predictions = (preds == -1).astype(int)
        
        # Pad com zeros
        self.predictions = np.concatenate([
            np.zeros(self.window_size),
            self.predictions
        ])[:len(self._extract_values(X_test))]
        
        self.scores = -self.model.negative_outlier_factor_
        
        return self.predictions
    
    def get_scores(self, X_test):
        """Retorna LOF scores"""
        self._validate_input(X_test)
        
        features = self._create_features(X_test)
        X_scaled = self.scaler.transform(features)
        
        self.model.fit_predict(X_scaled)
        return -self.model.negative_outlier_factor_
    
    def _create_features(self, data):
        """Cria features com janela deslizante"""
        values = self._extract_values(data)
        
        features = []
        for i in range(self.window_size, len(values)):
            window = values[i - self.window_size:i]
            features.append({
                'mean': np.mean(window),
                'std': np.std(window),
                'min': np.min(window),
                'max': np.max(window),
                'range': np.max(window) - np.min(window)
            })
        
        return pd.DataFrame(features)


class DBSCANDetector(UnsupervisedDetector):
    """Detector baseado em DBSCAN"""
    
    def __init__(self, eps=1.5, min_samples=5, window_size=24, contamination=0.05):
        super().__init__(name="DBSCAN", contamination=contamination)
        self.eps = eps
        self.min_samples = min_samples
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.model = None
    
    def fit(self, X_train):
        """Treina DBSCAN"""
        self._validate_input(X_train)
        
        # Criar features
        features = self._create_features(X_train)
        X_scaled = self.scaler.fit_transform(features)
        
        # DBSCAN não tem fit, mas inicializamos
        self.model = SKDBSCAN(
            eps=self.eps,
            min_samples=self.min_samples
        )
        self.is_fitted = True
    
    def predict(self, X_test):
        """Detecta anomalias usando DBSCAN"""
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        self._validate_input(X_test)
        
        # Criar features
        features = self._create_features(X_test)
        X_scaled = self.scaler.transform(features)
        
        # Predizer
        labels = self.model.fit_predict(X_scaled)
        self.predictions = (labels == -1).astype(int)
        
        # Pad com zeros
        self.predictions = np.concatenate([
            np.zeros(self.window_size),
            self.predictions
        ])[:len(self._extract_values(X_test))]
        
        self.scores = self.predictions.astype(float)  # Binary scores
        
        return self.predictions
    
    def get_scores(self, X_test):
        """Retorna outlier scores (binário)"""
        self._validate_input(X_test)
        
        features = self._create_features(X_test)
        X_scaled = self.scaler.transform(features)
        
        labels = self.model.fit_predict(X_scaled)
        return (labels == -1).astype(float)
    
    def _create_features(self, data):
        """Cria features com janela deslizante"""
        values = self._extract_values(data)
        
        features = []
        for i in range(self.window_size, len(values)):
            window = values[i - self.window_size:i]
            features.append({
                'mean': np.mean(window),
                'std': np.std(window),
                'min': np.min(window),
                'max': np.max(window),
                'range': np.max(window) - np.min(window)
            })
        
        return pd.DataFrame(features)