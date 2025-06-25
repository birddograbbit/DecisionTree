"""
Transformer module for stock price prediction.

This module provides transformer-based models for integration with
the DecisionTree trading system.
"""

# Core components
from .transformer_model import TimeSeriesTransformer
from .transformer_wrapper import TransformerModelWrapper
from .sequence_preparation import (
    SequencePreparator,
    StockSequenceDataset,
    prepare_data_for_transformer
)

# Technical indicators
from .technical_indicators_transformer import (
    add_technical_indicators,
    prepare_features_for_transformer,
    create_target_variable
)

# Strategy components
from .hybrid_strategy import (
    HybridTransformerStrategy,
    AdaptiveHybridStrategy,
    create_ensemble_predictions
)

__all__ = [
    # Models
    'TimeSeriesTransformer',
    'TransformerModelWrapper',
    
    # Data preparation
    'SequencePreparator',
    'StockSequenceDataset',
    'prepare_data_for_transformer',
    
    # Indicators
    'add_technical_indicators',
    'prepare_features_for_transformer',
    'create_target_variable',
    
    # Strategies
    'HybridTransformerStrategy',
    'AdaptiveHybridStrategy',
    'create_ensemble_predictions',
]

# Version info
__version__ = '1.0.0'
__author__ = 'DecisionTree Team'
