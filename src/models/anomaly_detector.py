"""
Módulo de Detectores de Anomalia
Implementa todas as classes de detectores reutilizáveis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from abc import ABC, abstractmethod


class AnomalyDetector(ABC):
    """Classe base abstrata para detectores de anomalia"""
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.threshold = None
        self.predictions = None
        self.scores = None
    
    @abstractmethod
    def fit(self, X_train):
        """Treina o detector com dados de treino"""
        pass
    
    @abstractmethod
    def predict(self, X_test):
        """Prediz anomalias (0: normal, 1: anomalia)"""
        pass
    
    @abstractmethod
    def get_scores(self, X_test):
        """Retorna scores de anomalia"""
        pass
    
    def summary(self):
        """Mostra resumo do detector"""
        return {
            'name': self.name,
            'threshold': self.threshold,
            'anomalies': int(self.predictions.sum()) if self.predictions is not None else None
        }


class ZScoreDetector(AnomalyDetector):
    """Detector baseado em Z-Score"""
    
    def __init__(self, threshold=3):
        super().__init__("Z-Score")
        self.z_threshold = threshold
        self.mean = None
        self.std = None
    
    def fit(self, X_train):
        """Calcula média e desvio padrão do treino"""
        if isinstance(X_train, pd.DataFrame):
            self.mean = X_train['value'].mean()
            self.std = X_train['value'].std()
        else:
            self.mean = np.mean(X_train)
            self.std = np.std(X_train)
        self.threshold = self.z_threshold
    
    def predict(self, X_test):
        """Detecta anomalias usando Z-Score"""
        if isinstance(X_test, pd.DataFrame):
            values = X_test['value'].values
        else:
            values = X_test
        
        z_scores = np.abs((values - self.mean) / self.std)
        self.predictions = (z_scores > self.threshold).astype(int)
        self.scores = z_scores
        return self.predictions
    
    def get_scores(self, X_test):
        """Retorna Z-scores"""
        if isinstance(X_test, pd.DataFrame):
            values = X_test['value'].values
        else:
            values = X_test
        return np.abs((values - self.mean) / self.std)


class IQRDetector(AnomalyDetector):
    """Detector baseado em IQR (Interquartile Range)"""
    
    def __init__(self, multiplier=1.5):
        super().__init__("IQR")
        self.multiplier = multiplier
        self.Q1 = None
        self.Q3 = None
        self.lower_bound = None
        self.upper_bound = None
    
    def fit(self, X_train):
        """Calcula Q1, Q3 e bounds"""
        if isinstance(X_train, pd.DataFrame):
            self.Q1 = X_train['value'].quantile(0.25)
            self.Q3 = X_train['value'].quantile(0.75)
        else:
            self.Q1 = np.percentile(X_train, 25)
            self.Q3 = np.percentile(X_train, 75)
        
        IQR = self.Q3 - self.Q1
        self.lower_bound = self.Q1 - self.multiplier * IQR
        self.upper_bound = self.Q3 + self.multiplier * IQR
    
    def predict(self, X_test):
        """Detecta anomalias usando IQR"""
        if isinstance(X_test, pd.DataFrame):
            values = X_test['value'].values
        else:
            values = X_test
        
        self.predictions = ((values < self.lower_bound) | (values > self.upper_bound)).astype(int)
        self.scores = np.abs(values - self.Q1)
        return self.predictions
    
    def get_scores(self, X_test):
        """Retorna distância do Q1"""
        if isinstance(X_test, pd.DataFrame):
            values = X_test['value'].values
        else:
            values = X_test
        return np.abs(values - self.Q1)


class IsolationForestDetector(AnomalyDetector):
    """Detector baseado em Isolation Forest"""
    
    def __init__(self, contamination=0.05, n_estimators=100):
        super().__init__("Isolation Forest")
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = None
        self.scaler = StandardScaler()
        self.window_size = 24
    
    def fit(self, X_train, window_size=24):
        """Treina Isolation Forest com features de janela deslizante"""
        self.window_size = window_size
        features = self._create_features(X_train, window_size)
        X_scaled = self.scaler.fit_transform(features)
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42
        )
        self.model.fit(X_scaled)
    
    def predict(self, X_test, window_size=None):
        """Detecta anomalias usando Isolation Forest"""
        if window_size is None:
            window_size = self.window_size
        
        features = self._create_features(X_test, window_size)
        X_scaled = self.scaler.transform(features)
        
        self.predictions = (self.model.predict(X_scaled) == -1).astype(int)
        # Pad com zeros no início
        self.predictions = np.concatenate([np.zeros(window_size), self.predictions])[:len(X_test)]
        
        self.scores = self.model.score_samples(X_scaled)
        return self.predictions
    
    def get_scores(self, X_test, window_size=None):
        """Retorna anomaly scores"""
        if window_size is None:
            window_size = self.window_size
        
        features = self._create_features(X_test, window_size)
        X_scaled = self.scaler.transform(features)
        return self.model.score_samples(X_scaled)
    
    def _create_features(self, data, window_size):
        """Cria features com janela deslizante"""
        if isinstance(data, pd.DataFrame):
            data = data['value'].values
        
        features = []
        for i in range(window_size, len(data)):
            window = data[i-window_size:i]
            features.append({
                'mean': np.mean(window),
                'std': np.std(window),
                'min': np.min(window),
                'max': np.max(window),
                'range': np.max(window) - np.min(window),
                'current': data[i]
            })
        return pd.DataFrame(features)


class LOFDetector(AnomalyDetector):
    """Detector baseado em Local Outlier Factor"""
    
    def __init__(self, n_neighbors=20, contamination=0.05):
        super().__init__("Local Outlier Factor")
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.window_size = 24
    
    def fit(self, X_train, window_size=24):
        """Treina LOF"""
        self.window_size = window_size
        features = self._create_features(X_train, window_size)
        X_scaled = self.scaler.fit_transform(features)
        
        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination
        )
        self.model.fit(X_scaled)
    
    def predict(self, X_test, window_size=None):
        """Detecta anomalias usando LOF"""
        if window_size is None:
            window_size = self.window_size
        
        features = self._create_features(X_test, window_size)
        X_scaled = self.scaler.transform(features)
        
        self.predictions = (self.model.predict(X_scaled) == -1).astype(int)
        self.predictions = np.concatenate([np.zeros(window_size), self.predictions])[:len(X_test)]
        
        return self.predictions
    
    def get_scores(self, X_test, window_size=None):
        """Retorna LOF scores"""
        if window_size is None:
            window_size = self.window_size
        
        features = self._create_features(X_test, window_size)
        X_scaled = self.scaler.transform(features)
        return self.model.negative_outlier_factor_
    
    def _create_features(self, data, window_size):
        """Cria features com janela deslizante"""
        if isinstance(data, pd.DataFrame):
            data = data['value'].values
        
        features = []
        for i in range(window_size, len(data)):
            window = data[i-window_size:i]
            features.append({
                'mean': np.mean(window),
                'std': np.std(window),
                'min': np.min(window),
                'max': np.max(window),
                'range': np.max(window) - np.min(window)
            })
        return pd.DataFrame(features)


class DBSCANDetector(AnomalyDetector):
    """Detector baseado em DBSCAN"""
    
    def __init__(self, eps=1.5, min_samples=5):
        super().__init__("DBSCAN")
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.scaler = StandardScaler()
        self.window_size = 24
    
    def fit(self, X_train, window_size=24):
        """Treina DBSCAN"""
        self.window_size = window_size
        features = self._create_features(X_train, window_size)
        X_scaled = self.scaler.fit_transform(features)
        
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.model.fit(X_scaled)
    
    def predict(self, X_test, window_size=None):
        """Detecta anomalias usando DBSCAN"""
        if window_size is None:
            window_size = self.window_size
        
        features = self._create_features(X_test, window_size)
        X_scaled = self.scaler.transform(features)
        
        labels = self.model.fit_predict(X_scaled)
        self.predictions = (labels == -1).astype(int)
        self.predictions = np.concatenate([np.zeros(window_size), self.predictions])[:len(X_test)]
        
        return self.predictions
    
    def get_scores(self, X_test, window_size=None):
        """Retorna outlier scores"""
        if window_size is None:
            window_size = self.window_size
        
        features = self._create_features(X_test, window_size)
        X_scaled = self.scaler.transform(features)
        
        labels = self.model.fit_predict(X_scaled)
        # Score: 1 se outlier, 0 se não
        return (labels == -1).astype(float)
    
    def _create_features(self, data, window_size):
        """Cria features com janela deslizante"""
        if isinstance(data, pd.DataFrame):
            data = data['value'].values
        
        features = []
        for i in range(window_size, len(data)):
            window = data[i-window_size:i]
            features.append({
                'mean': np.mean(window),
                'std': np.std(window),
                'min': np.min(window),
                'max': np.max(window),
                'range': np.max(window) - np.min(window)
            })
        return pd.DataFrame(features)