# src/models/__init__.py
"""
Models package for the trading system.

This package contains model classes that implement the BaseModel interface,
allowing for consistent use of different machine learning models.
"""

from .base_model import BaseModel
from .decision_tree_model import DecisionTreeModel
from .random_forest_model import RandomForestModel

try:
    from .xgboost_model import XGBoostModel
    __all__ = ['BaseModel', 'DecisionTreeModel', 'RandomForestModel', 'XGBoostModel']
except ImportError:
    # XGBoost not available
    __all__ = ['BaseModel', 'DecisionTreeModel', 'RandomForestModel']

# Import model factory last to ensure all models are registered
from .model_factory import ModelFactory
__all__.append('ModelFactory')
