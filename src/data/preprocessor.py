"""
Módulo de Pré-processamento
Normalização, scaling e feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPreprocessor:
    """Pré-processa dados para modelos"""
    
    def __init__(self):
        self.scaler = None
        self.scaler_type = None
    
    def normalize_minmax(self, data, feature_range=(0, 1)):
        """Normaliza dados com MinMaxScaler"""
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.scaler_type = 'minmax'
        
        if isinstance(data, pd.DataFrame):
            scaled = self.scaler.fit_transform(data[['value']])
            return pd.DataFrame(scaled, columns=['value'], index=data.index)
        else:
            return self.scaler.fit_transform(data.reshape(-1, 1))
    
    def normalize_standard(self, data):
        """Normaliza dados com StandardScaler"""
        self.scaler = StandardScaler()
        self.scaler_type = 'standard'
        
        if isinstance(data, pd.DataFrame):
            scaled = self.scaler.fit_transform(data[['value']])
            return pd.DataFrame(scaled, columns=['value'], index=data.index)
        else:
            return self.scaler.fit_transform(data.reshape(-1, 1))
    
    def transform(self, data):
        """Aplica transformação já treinada"""
        if self.scaler is None:
            raise ValueError("Scaler não foi treinado. Execute fit primeiro.")
        
        if isinstance(data, pd.DataFrame):
            scaled = self.scaler.transform(data[['value']])
            return pd.DataFrame(scaled, columns=['value'], index=data.index)
        else:
            return self.scaler.transform(data.reshape(-1, 1))
    
    def inverse_transform(self, data):
        """Inverte a transformação"""
        if self.scaler is None:
            raise ValueError("Scaler não foi treinado.")
        
        if isinstance(data, pd.DataFrame):
            original = self.scaler.inverse_transform(data[['value']])
            return pd.DataFrame(original, columns=['value'], index=data.index)
        else:
            return self.scaler.inverse_transform(data.reshape(-1, 1))
    
    def remove_outliers(self, data, method='iqr', threshold=1.5):
        """Remove outliers"""
        if method == 'iqr':
            Q1 = data['value'].quantile(0.25)
            Q3 = data['value'].quantile(0.75)
            IQR = Q3 - Q1
            
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            
            return data[(data['value'] >= lower) & (data['value'] <= upper)]
        
        elif method == 'zscore':
            z_scores = np.abs((data['value'] - data['value'].mean()) / data['value'].std())
            return data[z_scores < threshold]
    
    def handle_missing_values(self, data, method='forward_fill'):
        """Trata valores faltantes"""
        if method == 'forward_fill':
            return data.fillna(method='ffill').fillna(method='bfill')
        elif method == 'interpolate':
            return data.interpolate(method='linear')
        elif method == 'drop':
            return data.dropna()
    
    def create_sliding_windows(self, data, window_size, step=1):
        """Cria janelas deslizantes para dados temporais"""
        if isinstance(data, pd.DataFrame):
            values = data['value'].values
        else:
            values = data
        
        windows = []
        for i in range(0, len(values) - window_size + 1, step):
            windows.append(values[i:i + window_size])
        
        return np.array(windows)
    
    def create_features(self, data, window_size=24):
        """Cria features estatísticas de janelas"""
        if isinstance(data, pd.DataFrame):
            values = data['value'].values
        else:
            values = data
        
        features = []
        for i in range(window_size, len(values)):
            window = values[i - window_size:i]
            features.append({
                'mean': np.mean(window),
                'std': np.std(window),
                'min': np.min(window),
                'max': np.max(window),
                'range': np.max(window) - np.min(window),
                'median': np.median(window),
                'q25': np.percentile(window, 25),
                'q75': np.percentile(window, 75),
                'current': values[i]
            })
        
        return pd.DataFrame(features)
    
    def differencing(self, data, order=1):
        """Diferenciação de série temporal"""
        if isinstance(data, pd.DataFrame):
            result = data.copy()
            for _ in range(order):
                result['value'] = result['value'].diff()
            return result.dropna()
        else:
            result = data.copy()
            for _ in range(order):
                result = np.diff(result)
            return result


def preprocess_pipeline(data, normalize='minmax', remove_outliers=False, 
                       handle_missing='forward_fill'):
    """Pipeline completo de pré-processamento"""
    
    preprocessor = DataPreprocessor()
    
    # Tratar missing values
    if handle_missing:
        data = preprocessor.handle_missing_values(data, method=handle_missing)
    
    # Remover outliers
    if remove_outliers:
        data = preprocessor.remove_outliers(data)
    
    # Normalizar
    if normalize == 'minmax':
        data = preprocessor.normalize_minmax(data)
    elif normalize == 'standard':
        data = preprocessor.normalize_standard(data)
    
    return data, preprocessor


if __name__ == '__main__':
    # Exemplo
    data = pd.DataFrame({
        'value': np.random.randn(100).cumsum() + 10
    })
    
    processed, preprocessor = preprocess_pipeline(data)
    print("Original shape:", data.shape)
    print("Processed shape:", processed.shape)