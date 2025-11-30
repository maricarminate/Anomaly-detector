"""
Módulo Base para Detectores de Anomalia
Classe abstrata que define interface comum
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseAnomalyDetector(ABC):
    """
    Classe base abstrata para todos os detectores de anomalia.
    Define interface padrão que todos devem implementar.
    """
    
    def __init__(self, name="BaseDetector"):
        self.name = name
        self.model = None
        self.threshold = None
        self.predictions = None
        self.scores = None
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X_train):
        """
        Treina o detector com dados de treino.
        
        Args:
            X_train: Dados de treino (DataFrame ou array)
        """
        pass
    
    @abstractmethod
    def predict(self, X_test):
        """
        Prediz anomalias no conjunto de teste.
        
        Args:
            X_test: Dados de teste
            
        Returns:
            np.array: Predições binárias (0: normal, 1: anomalia)
        """
        pass
    
    @abstractmethod
    def get_scores(self, X_test):
        """
        Retorna scores de anomalia (valores contínuos).
        
        Args:
            X_test: Dados de teste
            
        Returns:
            np.array: Scores de anomalia
        """
        pass
    
    def fit_predict(self, X_train, X_test):
        """
        Treina e prediz em uma chamada.
        
        Args:
            X_train: Dados de treino
            X_test: Dados de teste
            
        Returns:
            np.array: Predições
        """
        self.fit(X_train)
        return self.predict(X_test)
    
    def summary(self):
        """
        Retorna resumo do detector.
        
        Returns:
            dict: Informações sobre o detector
        """
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'threshold': self.threshold,
            'n_anomalies': int(self.predictions.sum()) if self.predictions is not None else None,
            'n_samples': len(self.predictions) if self.predictions is not None else None,
            'anomaly_rate': f"{100 * self.predictions.sum() / len(self.predictions):.2f}%" if self.predictions is not None else None
        }
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __str__(self):
        summary = self.summary()
        return f"""
{self.name} Detector
{'='*50}
Fitted: {summary['is_fitted']}
Threshold: {summary['threshold']}
Anomalies: {summary['n_anomalies']} / {summary['n_samples']} ({summary['anomaly_rate']})
"""
    
    def _validate_input(self, X):
        """Valida formato de entrada"""
        if X is None:
            raise ValueError("Input data cannot be None")
        
        if isinstance(X, pd.DataFrame):
            if 'value' not in X.columns:
                raise ValueError("DataFrame must have 'value' column")
        elif not isinstance(X, np.ndarray):
            raise TypeError("Input must be DataFrame or numpy array")
        
        return True
    
    def _extract_values(self, X):
        """Extrai valores de DataFrame ou array"""
        if isinstance(X, pd.DataFrame):
            return X['value'].values
        else:
            return X.flatten() if X.ndim > 1 else X
    
    def save(self, filepath):
        """Salva o detector"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✓ Detector salvo: {filepath}")
    
    @staticmethod
    def load(filepath):
        """Carrega detector salvo"""
        import pickle
        with open(filepath, 'rb') as f:
            detector = pickle.load(f)
        print(f"✓ Detector carregado: {filepath}")
        return detector


class UnsupervisedDetector(BaseAnomalyDetector):
    """Classe base para detectores não supervisionados"""
    
    def __init__(self, name="UnsupervisedDetector", contamination=0.1):
        super().__init__(name)
        self.contamination = contamination
    
    def calculate_threshold(self, scores, method='percentile'):
        """
        Calcula threshold baseado nos scores.
        
        Args:
            scores: Array de scores
            method: 'percentile' ou 'std'
        """
        if method == 'percentile':
            percentile = (1 - self.contamination) * 100
            self.threshold = np.percentile(scores, percentile)
        elif method == 'std':
            self.threshold = np.mean(scores) + 3 * np.std(scores)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self.threshold


class SupervisedDetector(BaseAnomalyDetector):
    """Classe base para detectores supervisionados"""
    
    def __init__(self, name="SupervisedDetector"):
        super().__init__(name)
        self.labels_train = None
    
    def fit(self, X_train, y_train):
        """
        Treina com dados rotulados.
        
        Args:
            X_train: Features de treino
            y_train: Labels de treino (0: normal, 1: anomalia)
        """
        self.labels_train = y_train
        self._fit_supervised(X_train, y_train)
        self.is_fitted = True
    
    @abstractmethod
    def _fit_supervised(self, X_train, y_train):
        """Implementação específica do fit supervisionado"""
        pass
    
    def evaluate(self, X_test, y_test):
        """
        Avalia performance com labels verdadeiros.
        
        Args:
            X_test: Dados de teste
            y_test: Labels verdadeiros
            
        Returns:
            dict: Métricas de avaliação
        """
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        predictions = self.predict(X_test)
        
        return {
            'precision': precision_score(y_test, predictions, zero_division=0),
            'recall': recall_score(y_test, predictions, zero_division=0),
            'f1': f1_score(y_test, predictions, zero_division=0),
            'n_anomalies_true': int(y_test.sum()),
            'n_anomalies_pred': int(predictions.sum())
        }