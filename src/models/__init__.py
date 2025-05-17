"""
Decision Tree Trading Strategy - Models module

This module contains various model implementations for trading strategies.
"""

from .base_model import BaseModel
from .decision_tree_model import DecisionTreeModel
from .random_forest_model import RandomForestModel
from .stacking_model import StackingModel

try:
    from .xgboost_model import XGBoostModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

__all__ = ["BaseModel", "DecisionTreeModel", "RandomForestModel", "StackingModel"]

if XGBOOST_AVAILABLE:
    __all__.append("XGBoostModel")