"""
TimeSeriesTransformer model for stock price prediction.

This module implements a transformer-based neural network designed for
time series forecasting of stock prices.
"""

import torch
import torch.nn as nn
import numpy as np


class TimeSeriesTransformer(nn.Module):
    """
    Transformer model for time series prediction.
    
    Architecture:
    - Input embedding layer to project features to d_model dimensions
    - Positional encoding for sequence position information
    - Transformer encoder layers with multi-head attention
    - Output projection to prediction dimension
    
    Parameters:
    -----------
    feature_size : int
        Number of input features
    num_layers : int
        Number of transformer encoder layers
    d_model : int
        Dimension of the model (embedding size)
    nhead : int
        Number of attention heads
    dim_feedforward : int
        Dimension of feedforward network
    dropout : float
        Dropout rate
    seq_length : int
        Input sequence length
    prediction_length : int
        Number of steps to predict
    """
    
    def __init__(
        self,
        feature_size=9,
        num_layers=2,
        d_model=64,
        nhead=8,
        dim_feedforward=256,
        dropout=0.1,
        seq_length=30,
        prediction_length=1
    ):
        super(TimeSeriesTransformer, self).__init__()
        
        self.feature_size = feature_size
        self.d_model = d_model
        self.seq_length = seq_length
        self.prediction_length = prediction_length
        
        # Input projection layer
        self.input_fc = nn.Linear(feature_size, d_model)
        
        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_length, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True  # Use batch_first for easier handling
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Output projection
        self.fc_out = nn.Linear(d_model, prediction_length)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize input/output projections
        nn.init.xavier_uniform_(self.input_fc.weight)
        nn.init.zeros_(self.input_fc.bias)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.zeros_(self.fc_out.bias)
        
        # Initialize positional embeddings
        nn.init.normal_(self.pos_embedding, mean=0, std=0.02)
        
    def forward(self, src):
        """
        Forward pass through the transformer.
        
        Parameters:
        -----------
        src : torch.Tensor
            Input tensor of shape [batch_size, seq_length, feature_size]
            
        Returns:
        --------
        torch.Tensor
            Output predictions of shape [batch_size, prediction_length]
        """
        batch_size, seq_len, _ = src.shape
        
        # Project input features to d_model dimensions
        src = self.input_fc(src)  # [batch_size, seq_length, d_model]
        
        # Add positional encoding
        src = src + self.pos_embedding[:, :seq_len, :]
        
        # Pass through transformer encoder
        # Note: nn.TransformerEncoder expects [seq_length, batch_size, d_model]
        # But we use batch_first=True, so it expects [batch_size, seq_length, d_model]
        encoded = self.transformer_encoder(src)  # [batch_size, seq_length, d_model]
        
        # Use the last time step for prediction
        last_step = encoded[:, -1, :]  # [batch_size, d_model]
        
        # Project to prediction dimension
        out = self.fc_out(last_step)  # [batch_size, prediction_length]
        
        return out
    
    def predict_proba(self, X):
        """
        Generate probability predictions for compatibility with BaseModel interface.
        
        This method converts continuous price predictions to probabilities
        indicating upward movement.
        
        Parameters:
        -----------
        X : torch.Tensor
            Input tensor
            
        Returns:
        --------
        np.ndarray
            Probability of positive price movement
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(X)
            # Convert to probabilities using sigmoid
            # Assumes predictions > 0 indicate upward movement
            probabilities = torch.sigmoid(predictions).cpu().numpy()
        return probabilities.flatten()
    
    def save_checkpoint(self, path):
        """
        Save model checkpoint.
        
        Parameters:
        -----------
        path : str
            Path to save the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'feature_size': self.feature_size,
                'num_layers': self.transformer_encoder.num_layers,
                'd_model': self.d_model,
                'nhead': self.transformer_encoder.layers[0].self_attn.num_heads,
                'dim_feedforward': self.transformer_encoder.layers[0].linear1.out_features,
                'dropout': self.transformer_encoder.layers[0].dropout.p,
                'seq_length': self.seq_length,
                'prediction_length': self.prediction_length
            }
        }
        torch.save(checkpoint, path)
        
    @classmethod
    def load_checkpoint(cls, path, device='cpu'):
        """
        Load model from checkpoint.
        
        Parameters:
        -----------
        path : str
            Path to the checkpoint
        device : str or torch.device
            Device to load the model on
            
        Returns:
        --------
        TimeSeriesTransformer
            Loaded model instance
        """
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model
