"""
Testes Unitários para Detectores de Anomalia
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
import numpy as np
import pandas as pd

from src.models import (
    ZScoreDetector, IQRDetector, MovingAverageDetector, EWMADetector,
    IsolationForestDetector, LOFDetector, DBSCANDetector,
    EnsembleDetector
)


class TestStatisticalDetectors(unittest.TestCase):
    """Testes para detectores estatísticos"""
    
    def setUp(self):
        """Setup executado antes de cada teste"""
        np.random.seed(42)
        
        # Dados sintéticos
        self.train_data = pd.DataFrame({
            'value': np.random.normal(100, 10, 1000)
        })
        
        # Teste com algumas anomalias
        self.test_data = pd.DataFrame({
            'value': np.concatenate([
                np.random.normal(100, 10, 90),  # Normal
                [200, 250, 300],  # Anomalias
                np.random.normal(100, 10, 7)   # Normal
            ])
        })
    
    def test_zscore_detector(self):
        """Testa Z-Score Detector"""
        detector = ZScoreDetector(threshold=3)
        
        # Fit
        detector.fit(self.train_data)
        self.assertTrue(detector.is_fitted)
        self.assertIsNotNone(detector.mean)
        self.assertIsNotNone(detector.std)
        
        # Predict
        predictions = detector.predict(self.test_data)
        self.assertEqual(len(predictions), len(self.test_data))
        self.assertTrue(np.sum(predictions) > 0)  # Deve detectar algumas anomalias
        
        # Scores
        scores = detector.get_scores(self.test_data)
        self.assertEqual(len(scores), len(self.test_data))
    
    def test_iqr_detector(self):
        """Testa IQR Detector"""
        detector = IQRDetector(multiplier=1.5)
        
        detector.fit(self.train_data)
        self.assertTrue(detector.is_fitted)
        self.assertIsNotNone(detector.Q1)
        self.assertIsNotNone(detector.Q3)
        
        predictions = detector.predict(self.test_data)
        self.assertEqual(len(predictions), len(self.test_data))
        self.assertTrue(np.sum(predictions) > 0)
    
    def test_moving_average_detector(self):
        """Testa Moving Average Detector"""
        detector = MovingAverageDetector(window=10, threshold=2)
        
        detector.fit(self.train_data)
        self.assertTrue(detector.is_fitted)
        
        predictions = detector.predict(self.test_data)
        self.assertEqual(len(predictions), len(self.test_data))
    
    def test_ewma_detector(self):
        """Testa EWMA Detector"""
        detector = EWMADetector(span=10, threshold=2)
        
        detector.fit(self.train_data)
        self.assertTrue(detector.is_fitted)
        
        predictions = detector.predict(self.test_data)
        self.assertEqual(len(predictions), len(self.test_data))


class TestTreeBasedDetectors(unittest.TestCase):
    """Testes para detectores baseados em árvores"""
    
    def setUp(self):
        """Setup"""
        np.random.seed(42)
        
        # Mais dados para tree-based methods
        self.train_data = pd.DataFrame({
            'value': np.random.normal(100, 10, 500)
        })
        
        self.test_data = pd.DataFrame({
            'value': np.concatenate([
                np.random.normal(100, 10, 45),
                [200, 250],
                np.random.normal(100, 10, 3)
            ])
        })
    
    def test_isolation_forest(self):
        """Testa Isolation Forest"""
        detector = IsolationForestDetector(contamination=0.05, window_size=10)
        
        detector.fit(self.train_data)
        self.assertTrue(detector.is_fitted)
        self.assertIsNotNone(detector.model)
        
        predictions = detector.predict(self.test_data)
        self.assertTrue(len(predictions) <= len(self.test_data))
    
    def test_lof_detector(self):
        """Testa LOF"""
        detector = LOFDetector(n_neighbors=10, contamination=0.05, window_size=10)
        
        detector.fit(self.train_data)
        self.assertTrue(detector.is_fitted)
        
        predictions = detector.predict(self.test_data)
        self.assertTrue(len(predictions) <= len(self.test_data))
    
    def test_dbscan_detector(self):
        """Testa DBSCAN"""
        detector = DBSCANDetector(eps=1.5, min_samples=3, window_size=10)
        
        detector.fit(self.train_data)
        self.assertTrue(detector.is_fitted)
        
        predictions = detector.predict(self.test_data)
        self.assertTrue(len(predictions) <= len(self.test_data))


class TestEnsemble(unittest.TestCase):
    """Testes para Ensemble"""
    
    def setUp(self):
        """Setup"""
        np.random.seed(42)
        
        self.train_data = pd.DataFrame({
            'value': np.random.normal(100, 10, 500)
        })
        
        self.test_data = pd.DataFrame({
            'value': np.concatenate([
                np.random.normal(100, 10, 45),
                [200, 250],
                np.random.normal(100, 10, 3)
            ])
        })
    
    def test_ensemble_creation(self):
        """Testa criação de ensemble"""
        detectors = [
            ZScoreDetector(threshold=3),
            IQRDetector(multiplier=1.5)
        ]
        
        ensemble = EnsembleDetector(detectors=detectors, strategy='majority_vote')
        self.assertEqual(len(ensemble.detectors), 2)
    
    def test_ensemble_fit_predict(self):
        """Testa fit e predict do ensemble"""
        detectors = [
            ZScoreDetector(threshold=3),
            IQRDetector(multiplier=1.5)
        ]
        
        ensemble = EnsembleDetector(detectors=detectors, strategy='majority_vote')
        ensemble.fit(self.train_data)
        
        self.assertTrue(ensemble.is_fitted)
        
        predictions = ensemble.predict(self.test_data)
        self.assertEqual(len(predictions), len(self.test_data))
    
    def test_ensemble_strategies(self):
        """Testa diferentes estratégias de ensemble"""
        detectors = [
            ZScoreDetector(threshold=3),
            IQRDetector(multiplier=1.5)
        ]
        
        # Majority vote
        ensemble_majority = EnsembleDetector(detectors=detectors, strategy='majority_vote')
        ensemble_majority.fit(self.train_data)
        pred_majority = ensemble_majority.predict(self.test_data)
        
        # Consensus
        ensemble_consensus = EnsembleDetector(detectors=detectors, strategy='consensus')
        ensemble_consensus.fit(self.train_data)
        pred_consensus = ensemble_consensus.predict(self.test_data)
        
        # Consensus deve detectar menos anomalias
        self.assertLessEqual(np.sum(pred_consensus), np.sum(pred_majority))


class TestDetectorSaveLoad(unittest.TestCase):
    """Testes para salvar e carregar detectores"""
    
    def setUp(self):
        """Setup"""
        np.random.seed(42)
        self.train_data = pd.DataFrame({
            'value': np.random.normal(100, 10, 100)
        })
        self.test_file = Path('test_model.pkl')
    
    def tearDown(self):
        """Cleanup"""
        if self.test_file.exists():
            self.test_file.unlink()
    
    def test_save_load_detector(self):
        """Testa salvar e carregar detector"""
        detector = ZScoreDetector(threshold=3)
        detector.fit(self.train_data)
        
        # Salvar
        detector.save(self.test_file)
        self.assertTrue(self.test_file.exists())
        
        # Carregar
        loaded_detector = ZScoreDetector.load(self.test_file)
        self.assertTrue(loaded_detector.is_fitted)
        self.assertEqual(detector.mean, loaded_detector.mean)
        self.assertEqual(detector.std, loaded_detector.std)


class TestInputValidation(unittest.TestCase):
    """Testes para validação de entrada"""
    
    def test_invalid_input_types(self):
        """Testa tipos de entrada inválidos"""
        detector = ZScoreDetector()
        
        # None
        with self.assertRaises(ValueError):
            detector.fit(None)
        
        # String
        with self.assertRaises(TypeError):
            detector.fit("invalid")
    
    def test_predict_before_fit(self):
        """Testa predict sem fit"""
        detector = ZScoreDetector()
        
        test_data = pd.DataFrame({'value': [1, 2, 3]})
        
        with self.assertRaises(RuntimeError):
            detector.predict(test_data)


def run_tests():
    """Executa todos os testes"""
    
    print("=" * 70)
    print("EXECUTANDO TESTES UNITÁRIOS")
    print("=" * 70)
    
    # Criar suite de testes
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Adicionar testes
    suite.addTests(loader.loadTestsFromTestCase(TestStatisticalDetectors))
    suite.addTests(loader.loadTestsFromTestCase(TestTreeBasedDetectors))
    suite.addTests(loader.loadTestsFromTestCase(TestEnsemble))
    suite.addTests(loader.loadTestsFromTestCase(TestDetectorSaveLoad))
    suite.addTests(loader.loadTestsFromTestCase(TestInputValidation))
    
    # Executar
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Sumário
    print("\n" + "=" * 70)
    print("RESUMO DOS TESTES")
    print("=" * 70)
    print(f"Testes executados: {result.testsRun}")
    print(f"Sucessos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Falhas: {len(result.failures)}")
    print(f"Erros: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)