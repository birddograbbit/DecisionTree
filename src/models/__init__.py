"""
Models package for decision tree-based trading strategies.

This package contains various machine learning models that can be used
for predicting market movements and generating trading signals.
"""

from .base_model import BaseModel
from .decision_tree_model import DecisionTreeModel
from .random_forest_model import RandomForestModel
from .stacking_model import StackingModel

try:
    from .xgboost_model import XGBoostModel
    __all__ = ['BaseModel', 'DecisionTreeModel', 'RandomForestModel', 'XGBoostModel', 'StackingModel']
except ImportError:
    __all__ = ['BaseModel', 'DecisionTreeModel', 'RandomForestModel', 'StackingModel']