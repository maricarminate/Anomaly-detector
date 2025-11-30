"""
Testes Unitários para Módulos de Dados
Loader, Preprocessor
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import numpy as np
import pandas as pd
import tempfile
import os

from src.data import DataLoader, DataPreprocessor


class TestDataLoader(unittest.TestCase):
    """Testes para DataLoader"""
    
    def setUp(self):
        """Setup"""
        self.loader = DataLoader()
        
        # Criar arquivo temporário
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        self.temp_file.write('timestamp,value\n')
        self.temp_file.write('2014-01-01,100\n')
        self.temp_file.write('2014-01-02,150\n')
        self.temp_file.write('2014-01-03,120\n')
        self.temp_file.close()
    
    def tearDown(self):
        """Cleanup"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_load_csv(self):
        """Testa carregamento de CSV"""
        data = self.loader.load_csv(self.temp_file.name, parse_dates=['timestamp'])
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertEqual(len(data), 3)
        self.assertIn('timestamp', data.columns)
        self.assertIn('value', data.columns)
    
    def test_load_demo_data(self):
        """Testa carregamento de dados demo"""
        data = self.loader.load_nyc_taxi_demo()
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)
        self.assertIn('timestamp', data.columns)
        self.assertIn('value', data.columns)
    
    def test_split_train_test(self):
        """Testa split treino/teste"""
        self.loader.load_csv(self.temp_file.name)
        train, test = self.loader.split_train_test(train_size=0.6)
        
        self.assertEqual(len(train), 2)  # 60% de 3 = 1.8 → 2
        self.assertEqual(len(test), 1)
    
    def test_metadata(self):
        """Testa metadata"""
        self.loader.load_csv(self.temp_file.name)
        info = self.loader.info()
        
        self.assertIn('shape', info)
        self.assertIn('columns', info)
        self.assertIn('metadata', info)
    
    def test_save_data(self):
        """Testa salvar dados"""
        self.loader.load_csv(self.temp_file.name)
        
        temp_output = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_output.close()
        
        try:
            self.loader.save(temp_output.name)
            self.assertTrue(os.path.exists(temp_output.name))
        finally:
            if os.path.exists(temp_output.name):
                os.unlink(temp_output.name)


class TestDataPreprocessor(unittest.TestCase):
    """Testes para DataPreprocessor"""
    
    def setUp(self):
        """Setup"""
        self.preprocessor = DataPreprocessor()
        
        np.random.seed(42)
        self.data = pd.DataFrame({
            'value': np.random.normal(100, 10, 100)
        })
        
        # Dados com missing values
        self.data_with_nan = self.data.copy()
        self.data_with_nan.loc[10:15, 'value'] = np.nan
    
    def test_normalize_minmax(self):
        """Testa normalização MinMax"""
        normalized = self.preprocessor.normalize_minmax(self.data)
        
        self.assertEqual(len(normalized), len(self.data))
        self.assertGreaterEqual(normalized['value'].min(), 0)
        self.assertLessEqual(normalized['value'].max(), 1)
    
    def test_normalize_standard(self):
        """Testa normalização Standard"""
        normalized = self.preprocessor.normalize_standard(self.data)
        
        self.assertEqual(len(normalized), len(self.data))
        # Mean deve ser próx