"""
LSTM Autoencoder para Detecção de Anomalias em Séries Temporais
"""

import numpy as np
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Força CPU

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from .base import UnsupervisedDetector


class LSTMAutoencoderDetector(UnsupervisedDetector):
    """Detector baseado em LSTM Autoencoder"""
    
    def __init__(self, seq_length=20, latent_dim=16, epochs=10, 
                 batch_size=32, contamination=0.05):
        super().__init__(name="LSTM Autoencoder", contamination=contamination)
        self.seq_length = seq_length
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler()
        self.model = None
    
    def fit(self, X_train):
        """Treina LSTM Autoencoder"""
        self._validate_input(X_train)
        
        # Normalizar
        values = self._extract_values(X_train).reshape(-1, 1)
        train_scaled = self.scaler.fit_transform(values)
        
        # Criar sequências
        X_sequences = self._create_sequences(train_scaled)
        
        # Build model
        self.model = self._build_model()
        
        # Treinar
        self.model.fit(
            X_sequences, X_sequences,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0
        )
        
        # Calcular threshold
        train_pred = self.model.predict(X_sequences, verbose=0)
        train_mae = np.mean(np.abs(train_pred - X_sequences), axis=(1, 2))
        self.threshold = np.percentile(train_mae, 95)
        
        self.is_fitted = True
    
    def predict(self, X_test):
        """Detecta anomalias usando reconstruction error"""
        if not self.is_fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")
        
        self._validate_input(X_test)
        
        # Normalizar
        values = self._extract_values(X_test).reshape(-1, 1)
        test_scaled = self.scaler.transform(values)
        
        # Criar sequências
        X_sequences = self._create_sequences(test_scaled)
        
        # Predizer
        test_pred = self.model.predict(X_sequences, verbose=0)
        test_mae = np.mean(np.abs(test_pred - X_sequences), axis=(1, 2))
        
        # Classificar
        preds = (test_mae > self.threshold).astype(int)
        
        # Pad
        self.predictions = np.concatenate([
            np.zeros(self.seq_length),
            preds
        ])[:len(values)]
        
        self.scores = test_mae
        
        return self.predictions
    
    def get_scores(self, X_test):
        """Retorna reconstruction error"""
        self._validate_input(X_test)
        
        values = self._extract_values(X_test).reshape(-1, 1)
        test_scaled = self.scaler.transform(values)
        
        X_sequences = self._create_sequences(test_scaled)
        test_pred = self.model.predict(X_sequences, verbose=0)
        
        return np.mean(np.abs(test_pred - X_sequences), axis=(1, 2))
    
    def _build_model(self):
        """Constrói LSTM Autoencoder"""
        input_layer = layers.Input(shape=(self.seq_length, 1))
        
        # Encoder LSTM
        encoder = layers.LSTM(32, activation='relu', return_sequences=True)(input_layer)
        encoder = layers.LSTM(self.latent_dim, activation='relu', 
                             return_sequences=False, name='latent')(encoder)
        
        # Decoder LSTM
        decoder = layers.RepeatVector(self.seq_length)(encoder)
        decoder = layers.LSTM(self.latent_dim, activation='relu', 
                             return_sequences=True)(decoder)
        decoder = layers.LSTM(32, activation='relu', return_sequences=True)(decoder)
        
        # Output
        output = layers.TimeDistributed(layers.Dense(1))(decoder)
        
        # Model
        model = Model(input_layer, output)
        model.compile(optimizer=Adam(0.001), loss='mse')
        
        return model
    
    def _create_sequences(self, data):
        """Cria sequências para o LSTM"""
        sequences = []
        for i in range(len(data) - self.seq_length):
            sequences.append(data[i:i + self.seq_length])
        return np.array(sequences)
    
    def save_model(self, filepath):
        """Salva modelo Keras"""
        self.model.save(filepath)
        print(f"✓ Modelo LSTM salvo: {filepath}")
    
    def load_model(self, filepath):
        """Carrega modelo Keras"""
        self.model = tf.keras.models.load_model(filepath)
        self.is_fitted = True
        print(f"✓ Modelo LSTM carregado: {filepath}")