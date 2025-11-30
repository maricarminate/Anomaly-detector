"""
Métodos Estatísticos para Detecção de Anomalias
Z-Score, IQR, Moving Average, EWMA
"""

import numpy as np
import pandas as pd
from .base import UnsupervisedDetector


class ZScoreDetector(UnsupervisedDetector):
    """Detector baseado em Z-Score (desvios padrão)"""
    
    def __init__(self, threshold=3, contamination=0.05):
        super().__init__(name="Z-Score", contamination=contamination)
        self.z_threshold = threshold
        self.mean = None
        self.std = None
    
    def fit(self, X_train):
        """Calcula média e desvio padrão"""
        self._validate_input(X_train)
        values = self._extract_values(X_train)
        
        self.mean = np.mean(values)
        self.std = np.std(values)
        self.threshold = self.z_threshold
        self.is_fitted = True
    
    def predict(self, X_test):
        """Detecta anomalias usando Z-Score"""
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        self._validate_input(X_test)
        values = self._extract_values(X_test)
        
        z_scores = np.abs((values - self.mean) / self.std)
        self.predictions = (z_scores > self.threshold).astype(int)
        self.scores = z_scores
        
        return self.predictions
    
    def get_scores(self, X_test):
        """Retorna Z-scores absolutos"""
        self._validate_input(X_test)
        values = self._extract_values(X_test)
        return np.abs((values - self.mean) / self.std)


class IQRDetector(UnsupervisedDetector):
    """Detector baseado em IQR (Interquartile Range)"""
    
    def __init__(self, multiplier=1.5, contamination=0.05):
        super().__init__(name="IQR", contamination=contamination)
        self.multiplier = multiplier
        self.Q1 = None
        self.Q3 = None
        self.IQR = None
        self.lower_bound = None
        self.upper_bound = None
    
    def fit(self, X_train):
        """Calcula quartis e bounds"""
        self._validate_input(X_train)
        values = self._extract_values(X_train)
        
        self.Q1 = np.percentile(values, 25)
        self.Q3 = np.percentile(values, 75)
        self.IQR = self.Q3 - self.Q1
        
        self.lower_bound = self.Q1 - self.multiplier * self.IQR
        self.upper_bound = self.Q3 + self.multiplier * self.IQR
        self.is_fitted = True
    
    def predict(self, X_test):
        """Detecta anomalias usando IQR"""
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        self._validate_input(X_test)
        values = self._extract_values(X_test)
        
        self.predictions = ((values < self.lower_bound) | (values > self.upper_bound)).astype(int)
        self.scores = np.maximum(self.lower_bound - values, values - self.upper_bound)
        
        return self.predictions
    
    def get_scores(self, X_test):
        """Retorna distância dos bounds"""
        self._validate_input(X_test)
        values = self._extract_values(X_test)
        return np.maximum(self.lower_bound - values, values - self.upper_bound)


class MovingAverageDetector(UnsupervisedDetector):
    """Detector baseado em Moving Average"""
    
    def __init__(self, window=24, threshold=2, contamination=0.05):
        super().__init__(name="Moving Average", contamination=contamination)
        self.window = window
        self.z_threshold = threshold
        self.error_std = None
    
    def fit(self, X_train):
        """Calcula desvio padrão do erro no treino"""
        self._validate_input(X_train)
        
        if isinstance(X_train, pd.DataFrame):
            values = X_train['value']
        else:
            values = pd.Series(X_train.flatten())
        
        ma = values.rolling(window=self.window, center=True).mean()
        error = values - ma
        self.error_std = error.std()
        self.threshold = self.z_threshold * self.error_std
        self.is_fitted = True
    
    def predict(self, X_test):
        """Detecta anomalias usando Moving Average"""
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        self._validate_input(X_test)
        
        if isinstance(X_test, pd.DataFrame):
            values = X_test['value']
        else:
            values = pd.Series(X_test.flatten())
        
        ma = values.rolling(window=self.window, center=True).mean()
        error = np.abs(values - ma)
        
        self.predictions = (error > self.threshold).astype(int)
        self.predictions = self.predictions.fillna(0).values
        self.scores = error.fillna(0).values
        
        return self.predictions
    
    def get_scores(self, X_test):
        """Retorna erro absoluto da média móvel"""
        self._validate_input(X_test)
        
        if isinstance(X_test, pd.DataFrame):
            values = X_test['value']
        else:
            values = pd.Series(X_test.flatten())
        
        ma = values.rolling(window=self.window, center=True).mean()
        return np.abs(values - ma).fillna(0).values


class EWMADetector(UnsupervisedDetector):
    """Detector baseado em Exponentially Weighted Moving Average"""
    
    def __init__(self, span=24, threshold=2, contamination=0.05):
        super().__init__(name="EWMA", contamination=contamination)
        self.span = span
        self.z_threshold = threshold
        self.error_std = None
    
    def fit(self, X_train):
        """Calcula desvio padrão do erro no treino"""
        self._validate_input(X_train)
        
        if isinstance(X_train, pd.DataFrame):
            values = X_train['value']
        else:
            values = pd.Series(X_train.flatten())
        
        ewma = values.ewm(span=self.span, adjust=False).mean()
        error = values - ewma
        self.error_std = error.std()
        self.threshold = self.z_threshold * self.error_std
        self.is_fitted = True
    
    def predict(self, X_test):
        """Detecta anomalias usando EWMA"""
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        self._validate_input(X_test)
        
        if isinstance(X_test, pd.DataFrame):
            values = X_test['value']
        else:
            values = pd.Series(X_test.flatten())
        
        ewma = values.ewm(span=self.span, adjust=False).mean()
        error = np.abs(values - ewma)
        
        self.predictions = (error > self.threshold).astype(int).values
        self.scores = error.values
        
        return self.predictions
    
    def get_scores(self, X_test):
        """Retorna erro absoluto do EWMA"""
        self._validate_input(X_test)
        
        if isinstance(X_test, pd.DataFrame):
            values = X_test['value']
        else:
            values = pd.Series(X_test.flatten())
        
        ewma = values.ewm(span=self.span, adjust=False).mean()
        return np.abs(values - ewma).values