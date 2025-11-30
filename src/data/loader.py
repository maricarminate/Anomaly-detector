"""
Módulo de Carregamento de Dados
Funções para carregar e validar dados
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
from io import StringIO


class DataLoader:
    """Carrega dados de diferentes fontes"""
    
    def __init__(self):
        self.data = None
        self.metadata = {}
    
    def load_csv(self, filepath, parse_dates=None):
        """Carrega dados de CSV"""
        self.data = pd.read_csv(filepath, parse_dates=parse_dates)
        self.metadata['source'] = str(filepath)
        self.metadata['shape'] = self.data.shape
        return self.data
    
    def load_from_url(self, url, parse_dates=None):
        """Carrega dados de URL"""
        response = requests.get(url)
        self.data = pd.read_csv(StringIO(response.text), parse_dates=parse_dates)
        self.metadata['source'] = url
        self.metadata['shape'] = self.data.shape
        return self.data
    
    def load_nyc_taxi_demo(self):
        """Carrega dados de demo do NYC Taxi"""
        url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause/NYC_taxi.csv"
        
        try:
            self.data = pd.read_csv(url, parse_dates=['timestamp'])
            self.data.columns = ['timestamp', 'value']
            self.metadata['source'] = 'NYC Taxi (NAB)'
        except:
            print("⚠️  Erro ao baixar. Criando dados simulados...")
            self._create_synthetic_data()
        
        self.metadata['shape'] = self.data.shape
        return self.data
    
    def _create_synthetic_data(self):
        """Cria dados sintéticos para testes"""
        dates = pd.date_range(start='2014-01-01', periods=10000, freq='H')
        values = np.sin(np.arange(len(dates)) / 24 * 2 * np.pi) * 5000 + 10000
        values += np.random.normal(0, 500, len(dates))
        
        self.data = pd.DataFrame({'timestamp': dates, 'value': values})
        self.metadata['source'] = 'Synthetic Data'
    
    def split_train_test(self, train_size=0.8):
        """Separa em treino e teste"""
        split_idx = int(len(self.data) * train_size)
        
        train = self.data.iloc[:split_idx].copy()
        test = self.data.iloc[split_idx:].copy()
        
        self.metadata['train_size'] = len(train)
        self.metadata['test_size'] = len(test)
        
        return train, test
    
    def head(self, n=5):
        """Mostra primeiras linhas"""
        return self.data.head(n)
    
    def info(self):
        """Mostra informações sobre os dados"""
        return {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing': self.data.isnull().sum().to_dict(),
            'metadata': self.metadata
        }
    
    def save(self, filepath):
        """Salva dados em CSV"""
        self.data.to_csv(filepath, index=False)
        print(f"✓ Dados salvos: {filepath}")


def load_data(filepath, parse_dates=None):
    """Função auxiliar para carregar dados"""
    loader = DataLoader()
    return loader.load_csv(filepath, parse_dates=parse_dates)


def load_demo_data():
    """Carrega dados de demonstração"""
    loader = DataLoader()
    return loader.load_nyc_taxi_demo()


if __name__ == '__main__':
    loader = DataLoader()
    data = loader.load_nyc_taxi_demo()
    print(loader.info())