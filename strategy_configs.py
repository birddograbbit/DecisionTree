"""
Strategy configurations for different model types.

This module contains pre-defined configurations for various trading strategies
and models. These configurations can be used with the strategy_runner.py script.
"""
import config

# Decision Tree strategy configurations
DECISION_TREE_CONFIG = {
    'name': 'Decision Tree',
    'model_type': 'decision_tree',
    'model_params': {
        'max_depth': 5
        # Removed class_weight parameter as DecisionTreeModel doesn't support it
    },
    'position_sizing': 'confidence',
    'consecutive_buys': False,
    'min_holding_days': 1,
    'use_adaptive_thresholds': 'never'  # Use standard thresholds
}

# Decision Tree (Calibrated) strategy configuration
DECISION_TREE_CALIBRATED_CONFIG = {
    'name': 'Decision Tree (Calibrated)',
    'model_type': 'decision_tree',
    'model_params': {
        'max_depth': 5
        # Removed class_weight parameter as DecisionTreeModel doesn't support it
    },
    'use_calibration': True,
    'position_sizing': 'confidence',
    'consecutive_buys': False,
    'min_holding_days': 1,
    'use_adaptive_thresholds': 'always',  # Always use adaptive thresholds
    'buy_percentile': 80,                 # Use 80th percentile as buy threshold
    'sell_percentile': 20                 # Use 20th percentile as sell threshold
}

# Random Forest strategy configuration
RANDOM_FOREST_CONFIG = {
    'name': 'Random Forest',
    'model_type': 'random_forest',
    'model_params': {
        'n_estimators': 100,
        'max_depth': 5,
        'class_weight': 'balanced'
    },
    'position_sizing': 'confidence',
    'consecutive_buys': False,
    'min_holding_days': 1,
    'use_adaptive_thresholds': 'auto'  # Use adaptive thresholds if needed
}

# Random Forest (Calibrated) strategy configuration
RANDOM_FOREST_CALIBRATED_CONFIG = {
    'name': 'Random Forest (Calibrated)',
    'model_type': 'random_forest',
    'model_params': {
        'n_estimators': 100,
        'max_depth': 5,
        'class_weight': 'balanced'
    },
    'use_calibration': True,
    'position_sizing': 'confidence',
    'consecutive_buys': False,
    'min_holding_days': 1,
    'use_adaptive_thresholds': 'always',  # Always use adaptive thresholds
    'buy_percentile': 80,                 # Use 80th percentile as buy threshold
    'sell_percentile': 20                 # Use 20th percentile as sell threshold
}

# XGBoost with fixed position sizing
XGBOOST_FIXED_CONFIG = {
    'name': 'XGBoost (Fixed Position)',
    'model_type': 'xgboost',
    'model_params': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    'position_sizing': 'fixed',
    'consecutive_buys': False,
    'min_holding_days': 1,
    'use_adaptive_thresholds': 'auto'  # Use adaptive thresholds if needed
}

# XGBoost with confidence-based position sizing
XGBOOST_CONFIDENCE_CONFIG = {
    'name': 'XGBoost (Confidence-Scaled)',
    'model_type': 'xgboost',
    'model_params': {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    },
    'position_sizing': 'confidence',
    'consecutive_buys': False,
    'min_holding_days': 1,
    'use_adaptive_thresholds': 'auto'  # Use adaptive thresholds if needed
}

# Stacking Ensemble
STACKING_CONFIG = {
    'name': 'Stacking Ensemble',
    'model_type': 'stacking',
    'model_params': {
        'base_models': [
            {
                'model_type': 'decision_tree',
                'model_params': {'max_depth': 5}  # Removed class_weight
            },
            {
                'model_type': 'random_forest',
                'model_params': {'n_estimators': 100, 'max_depth': 5, 'class_weight': 'balanced'}
            },
            {
                'model_type': 'xgboost',
                'model_params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
            }
        ],
        'meta_model_type': 'logistic_regression',
        'meta_model_params': {'C': 1.0, 'max_iter': 1000}
    },
    'position_sizing': 'confidence',
    'consecutive_buys': False,
    'min_holding_days': 1,
    'use_adaptive_thresholds': 'auto'  # Use adaptive thresholds if needed
}

# Regime-Adaptive Random Forest
REGIME_ADAPTIVE_RF_CONFIG = {
    'name': 'Regime Adaptive RF',
    'model_type': 'random_forest',
    'model_params': {
        'n_estimators': 100,
        'max_depth': 5,
        'class_weight': 'balanced'
    },
    'use_calibration': True,
    'position_sizing': 'confidence',
    'consecutive_buys': False,
    'min_holding_days': 1,
    'use_adaptive_thresholds': 'auto',  # always,never,auto use adaptive thresholds, 
    'buy_percentile': 80,                 # Use 80th percentile as buy threshold 
    'sell_percentile': 20                 # Use 20th percentile as sell threshold
}

# All strategy configurations
STRATEGY_CONFIGS = {
    'decision_tree': DECISION_TREE_CONFIG,
    'decision_tree_calibrated': DECISION_TREE_CALIBRATED_CONFIG,
    'random_forest': RANDOM_FOREST_CONFIG,
    'random_forest_calibrated': RANDOM_FOREST_CALIBRATED_CONFIG,
    'xgboost_fixed': XGBOOST_FIXED_CONFIG,
    'xgboost_confidence': XGBOOST_CONFIDENCE_CONFIG,
    'stacking': STACKING_CONFIG,
    'regime_adaptive_rf': REGIME_ADAPTIVE_RF_CONFIG,
    'transformer': {
        'name': 'Transformer',
        'model_type': 'transformer',
        'model_params': config.TRANSFORMER_CONFIG['default'],
        'position_sizing': 'confidence',
        'use_adaptive_thresholds': 'auto'
    },
    'hybrid': {
        'name': 'Hybrid',
        'model_type': 'hybrid',
        'model_params': {
            'dt_params': {},
            'tf_params': config.TRANSFORMER_CONFIG['default']
        },
        'position_sizing': 'confidence',
        'use_adaptive_thresholds': 'auto'
    }
}