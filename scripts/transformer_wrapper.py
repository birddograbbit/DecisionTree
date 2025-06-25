"""
Wrapper to integrate TimeSeriesTransformer with DecisionTree system.

This module provides a BaseModel-compatible wrapper for the transformer model,
allowing seamless integration with the existing trading system.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# These will be imported from the main system when integrated
# from src.models.base_model import BaseModel
# For now, we'll create a minimal interface


class TransformerModelWrapper:
    """
    Wrapper class that makes TimeSeriesTransformer compatible with BaseModel interface.
    
    This wrapper handles:
    - Data preparation and sequence creation
    - Training with PyTorch optimization
    - Prediction with proper output format
    - Model persistence
    """
    
    def __init__(self, seq_length=30, prediction_length=1, 
                 n_features=9, d_model=64, n_heads=8, n_layers=2,
                 dropout=0.1, learning_rate=0.001, batch_size=32,
                 epochs=20, device=None, **kwargs):
        """
        Initialize the transformer wrapper.
        
        Parameters:
        -----------
        seq_length : int
            Length of input sequences
        prediction_length : int
            Number of steps to predict
        n_features : int
            Number of input features
        d_model : int
            Model dimension
        n_heads : int
            Number of attention heads
        n_layers : int
            Number of transformer layers
        dropout : float
            Dropout rate
        learning_rate : float
            Learning rate for optimization
        batch_size : int
            Batch size for training
        epochs : int
            Number of training epochs
        device : str or None
            Device to use ('cuda' or 'cpu')
        **kwargs : dict
            Additional arguments
        """
        # Model configuration
        self.seq_length = seq_length
        self.prediction_length = prediction_length
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        
        # Training configuration
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        # Device configuration
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        # Initialize model and components
        self.model = None
        self.preparator = None
        self.feature_columns = None
        self.is_fitted = False
        
    def _create_model(self):
        """Create the transformer model."""
        from scripts.transformer_model import TimeSeriesTransformer
        
        self.model = TimeSeriesTransformer(
            feature_size=self.n_features,
            num_layers=self.n_layers,
            d_model=self.d_model,
            nhead=self.n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=self.dropout,
            seq_length=self.seq_length,
            prediction_length=self.prediction_length
        ).to(self.device)
        
    def train(self, X, y):
        """
        Train the transformer model.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Feature matrix with shape (n_samples, n_features)
        y : pd.Series or np.ndarray
            Target values (1 for up, 0 for down)
            
        Returns:
        --------
        self
            For method chaining
        """
        from scripts.sequence_preparation import SequencePreparator, StockSequenceDataset
        
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, (pd.Series, np.ndarray)):
            y = np.array(y)
            
        # Store feature columns
        self.feature_columns = list(X.columns)
        self.n_features = len(self.feature_columns)
        
        # Create sequence preparator
        self.preparator = SequencePreparator(
            seq_length=self.seq_length,
            prediction_length=self.prediction_length,
            feature_columns=self.feature_columns,
            target_column=self.feature_columns[-1]  # Assume last column is close price
        )
        
        # Prepare sequences
        # Combine X and y for sequence creation
        data = X.copy()
        data['target'] = y
        
        # Fit preparator and create sequences
        X_seq, y_seq = self.preparator.fit_transform(data, include_targets=False)
        
        # Create target sequences from original targets
        # We need to align targets with sequences
        y_seq = self._create_target_sequences(y)
        
        # Create dataset and dataloader
        dataset = StockSequenceDataset(X_seq, y_seq)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=True
        )
        
        # Create model
        self._create_model()
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.BCEWithLogitsLoss()  # Binary classification
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y.float())
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            # Print progress
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}")
                
        self.is_fitted = True
        return self
        
    def predict(self, X):
        """
        Generate predictions for given features.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
            
        Returns:
        --------
        np.ndarray
            Predicted probabilities for positive class
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before prediction")
            
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_columns)
            
        # Prepare sequences
        X_seq, _ = self.preparator.transform(X, include_targets=False)
        
        # Handle case where we don't have enough data for sequences
        if len(X_seq) == 0:
            # Return neutral predictions for insufficient data
            return np.full(len(X), 0.5)
            
        # Create dataset
        dataset = StockSequenceDataset(X_seq)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        # Generate predictions
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch_X in dataloader:
                batch_X = batch_X.to(self.device)
                outputs = self.model(batch_X)
                probs = torch.sigmoid(outputs).cpu().numpy()
                predictions.extend(probs.flatten())
                
        # Convert to numpy array
        predictions = np.array(predictions)
        
        # Pad predictions to match input length
        # (sequences produce fewer predictions than input rows)
        if len(predictions) < len(X):
            # Pad with neutral predictions at the beginning
            padding = np.full(len(X) - len(predictions), 0.5)
            predictions = np.concatenate([padding, predictions])
            
        return predictions
        
    def _create_target_sequences(self, y):
        """
        Create target sequences from raw targets.
        
        Parameters:
        -----------
        y : np.ndarray
            Raw target values
            
        Returns:
        --------
        np.ndarray
            Target values aligned with sequences
        """
        # For sequence prediction, we need targets starting from seq_length
        n_sequences = len(y) - self.seq_length - self.prediction_length + 1
        
        if n_sequences <= 0:
            return np.array([])
            
        # Extract targets for each sequence
        targets = np.zeros(n_sequences)
        for i in range(n_sequences):
            # Target is at position seq_length + i
            target_idx = self.seq_length + i
            if target_idx < len(y):
                targets[i] = y[target_idx]
                
        return targets
        
    def get_feature_importance(self):
        """
        Return feature importance scores.
        
        Note: Transformers don't have traditional feature importance.
        We'll return attention weights or uniform importance.
        
        Returns:
        --------
        dict
            Feature importance scores
        """
        if not self.is_fitted:
            return {}
            
        # For now, return uniform importance
        # In future, could analyze attention weights
        importance = {col: 1.0 / len(self.feature_columns) 
                     for col in self.feature_columns}
        return importance
        
    def save(self, path):
        """
        Save model to disk.
        
        Parameters:
        -----------
        path : str
            Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save untrained model")
            
        # Create directory if needed
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        save_dict = {
            'model_state': self.model.state_dict(),
            'model_config': {
                'seq_length': self.seq_length,
                'prediction_length': self.prediction_length,
                'n_features': self.n_features,
                'd_model': self.d_model,
                'n_heads': self.n_heads,
                'n_layers': self.n_layers,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'epochs': self.epochs
            },
            'preparator': self.preparator,
            'feature_columns': self.feature_columns,
            'is_fitted': self.is_fitted
        }
        
        torch.save(save_dict, path)
        
    @classmethod
    def load(cls, path):
        """
        Load model from disk.
        
        Parameters:
        -----------
        path : str
            Path to saved model
            
        Returns:
        --------
        TransformerModelWrapper
            Loaded model instance
        """
        # Load saved state
        save_dict = torch.load(path, map_location='cpu')
        
        # Create instance
        instance = cls(**save_dict['model_config'])
        
        # Restore state
        instance._create_model()
        instance.model.load_state_dict(save_dict['model_state'])
        instance.preparator = save_dict['preparator']
        instance.feature_columns = save_dict['feature_columns']
        instance.is_fitted = save_dict['is_fitted']
        
        return instance
