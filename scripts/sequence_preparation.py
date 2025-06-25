"""
Sequence preparation utilities for transformer models.

This module handles the conversion of tabular time series data into
sequences suitable for transformer models.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset


class SequencePreparator:
    """
    Prepares sequences from time series data for transformer models.
    
    This class handles:
    - Feature scaling
    - Sequence creation with sliding windows
    - Train/test splitting while maintaining temporal order
    """
    
    def __init__(self, seq_length=30, prediction_length=1, 
                 feature_columns=None, target_column='close'):
        """
        Initialize the sequence preparator.
        
        Parameters:
        -----------
        seq_length : int
            Length of input sequences
        prediction_length : int
            Number of steps to predict
        feature_columns : list or None
            List of feature column names. If None, uses default features
        target_column : str
            Name of the target column
        """
        self.seq_length = seq_length
        self.prediction_length = prediction_length
        self.feature_columns = feature_columns or self._get_default_features()
        self.target_column = target_column
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def _get_default_features(self):
        """Get default feature columns for transformer."""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'bb_high', 'bb_low', 'ma_20', 'ma_20_slope'
        ]
        
    def fit(self, data):
        """
        Fit the scaler on training data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Training data with feature columns
        """
        # Select and validate features
        available_features = []
        for col in self.feature_columns:
            if col in data.columns:
                available_features.append(col)
                
        if not available_features:
            raise ValueError("No valid feature columns found in data")
            
        self.feature_columns = available_features
        
        # Fit scaler
        feature_data = data[self.feature_columns].values
        self.scaler.fit(feature_data)
        self.is_fitted = True
        
    def transform(self, data, include_targets=True):
        """
        Transform data into scaled sequences.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to transform
        include_targets : bool
            Whether to include target values
            
        Returns:
        --------
        X : np.ndarray
            Input sequences of shape (n_samples, seq_length, n_features)
        y : np.ndarray or None
            Target values of shape (n_samples, prediction_length)
        """
        if not self.is_fitted:
            raise ValueError("SequencePreparator must be fitted before transform")
            
        # Scale features
        feature_data = data[self.feature_columns].values
        scaled_data = self.scaler.transform(feature_data)
        
        # Find target column index
        if include_targets and self.target_column in self.feature_columns:
            target_idx = self.feature_columns.index(self.target_column)
        else:
            target_idx = None
            
        # Create sequences
        X, y = self._create_sequences(scaled_data, target_idx)
        
        if include_targets and target_idx is not None:
            return X, y
        else:
            return X, None
            
    def fit_transform(self, data, include_targets=True):
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data, include_targets)
        
    def _create_sequences(self, data, target_idx):
        """
        Create sequences using sliding windows.
        
        Parameters:
        -----------
        data : np.ndarray
            Scaled data array
        target_idx : int or None
            Index of target column in features
            
        Returns:
        --------
        X : np.ndarray
            Input sequences
        y : np.ndarray or None
            Target sequences
        """
        n_samples = len(data) - self.seq_length - self.prediction_length + 1
        
        if n_samples <= 0:
            raise ValueError("Not enough data to create sequences")
            
        X = np.zeros((n_samples, self.seq_length, data.shape[1]))
        
        if target_idx is not None:
            y = np.zeros((n_samples, self.prediction_length))
        else:
            y = None
            
        for i in range(n_samples):
            # Input sequence
            X[i] = data[i:i + self.seq_length]
            
            # Target values
            if target_idx is not None:
                target_start = i + self.seq_length
                target_end = target_start + self.prediction_length
                y[i] = data[target_start:target_end, target_idx]
                
        return X, y
        
    def inverse_transform_predictions(self, predictions):
        """
        Convert scaled predictions back to original scale.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Scaled predictions
            
        Returns:
        --------
        np.ndarray
            Predictions in original scale
        """
        if not self.is_fitted:
            raise ValueError("SequencePreparator must be fitted first")
            
        # Find target column index
        if self.target_column not in self.feature_columns:
            raise ValueError(f"Target column {self.target_column} not in features")
            
        target_idx = self.feature_columns.index(self.target_column)
        
        # Create dummy array for inverse transform
        n_samples = len(predictions)
        dummy = np.zeros((n_samples, len(self.feature_columns)))
        dummy[:, target_idx] = predictions.flatten()
        
        # Inverse transform
        inversed = self.scaler.inverse_transform(dummy)
        
        return inversed[:, target_idx]


class StockSequenceDataset(Dataset):
    """
    PyTorch Dataset for stock price sequences.
    
    This dataset handles the loading and batching of sequence data
    for training transformer models.
    """
    
    def __init__(self, X, y=None):
        """
        Initialize the dataset.
        
        Parameters:
        -----------
        X : np.ndarray
            Input sequences of shape (n_samples, seq_length, n_features)
        y : np.ndarray or None
            Target values of shape (n_samples, prediction_length)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]
            

def prepare_data_for_transformer(data, seq_length=30, prediction_length=1,
                                train_end_date=None, feature_columns=None):
    """
    Convenience function to prepare data for transformer training.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw data with datetime index
    seq_length : int
        Length of input sequences
    prediction_length : int
        Number of steps to predict
    train_end_date : str or None
        End date for training data. If None, uses 80% for training
    feature_columns : list or None
        Feature columns to use
        
    Returns:
    --------
    dict
        Dictionary containing:
        - X_train, y_train: Training sequences and targets
        - X_test, y_test: Test sequences and targets
        - preparator: Fitted SequencePreparator instance
        - train_data, test_data: Original split data
    """
    # Sort by date
    data = data.sort_index()
    
    # Split data
    if train_end_date:
        train_data = data[data.index <= train_end_date]
        test_data = data[data.index > train_end_date]
    else:
        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
    # Create preparator
    preparator = SequencePreparator(
        seq_length=seq_length,
        prediction_length=prediction_length,
        feature_columns=feature_columns
    )
    
    # Prepare sequences
    X_train, y_train = preparator.fit_transform(train_data)
    X_test, y_test = preparator.transform(test_data)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'preparator': preparator,
        'train_data': train_data,
        'test_data': test_data
    }
