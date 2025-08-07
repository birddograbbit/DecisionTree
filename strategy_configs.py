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

# 5-Minute ML Strategy Configurations
DECISION_TREE_5MIN_CONFIG = {
    'name': 'Decision Tree (5min)',
    'model_type': 'decision_tree',
    'model_params': {
        'max_depth': 3,
        'calibrate': False
    },
    'position_sizing': 'confidence',
    'consecutive_buys': False,
    'min_holding_days': 1,
    'use_adaptive_thresholds': 'never'
}

RANDOM_FOREST_5MIN_CONFIG = {
    'name': 'Random Forest (5min)',
    'model_type': 'random_forest',
    'model_params': {
        'n_estimators': 50,
        'max_depth': 3,
        'class_weight': 'balanced',
        'calibrate': False
    },
    'position_sizing': 'confidence',
    'consecutive_buys': False,
    'min_holding_days': 1,
    'use_adaptive_thresholds': 'auto'
}

XGBOOST_5MIN_CONFIG = {
    'name': 'XGBoost (5min)',
    'model_type': 'xgboost',
    'model_params': {
        'n_estimators': 200,
        'max_depth': 3,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.7
    },
    'position_sizing': 'confidence',
    'consecutive_buys': False,
    'min_holding_days': 1,
    'use_adaptive_thresholds': 'auto'
}

TRANSFORMER_5MIN_CONFIG = {
    'name': 'Transformer (5min)',
    'model_type': 'transformer',
    'model_params': config.TRANSFORMER_CONFIG['5min'],
    'position_sizing': 'confidence',
    'use_adaptive_thresholds': 'auto'
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

# Momentum Strategy Configurations

# BB-RSI-ADX Momentum Continuation Strategy
BB_RSI_ADX_CONFIG = {
    'name': 'BB-RSI-ADX Momentum',
    'model_type': 'bb_rsi_adx',
    'symbol': 'SPY',  # Will be overridden by command line
    'position_size': 0.1,
    'primary_timeframe': '1h',
    # Strategy parameters
    'bb_period': 20,
    'bb_std': 2,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'supertrend_period': 10,
    'supertrend_multiplier': 3,
    'adx_primary_threshold': 20,
    'adx_secondary_threshold': 40,
    'atr_stop_multiplier': 6,
    'bars_to_enter': 1,
    'allow_same_bar_exit': False
}

# TEMA Trend Following Strategy
TEMA_CONFIG = {
    'name': 'TEMA Trend Following',
    'model_type': 'tema',
    'symbol': 'SPY',  # Will be overridden by command line
    'position_size': 0.1,
    'primary_timeframe': '1h',
    # Strategy parameters
    'tema_primary_fast': 10,
    'tema_primary_slow': 80,
    'tema_secondary_fast': 20,
    'tema_secondary_slow': 70,
    'adx_threshold': 40,
    'cmo_long_threshold': 40,
    'cmo_short_threshold': -40,
    'atr_entry_offset': 1,
    'atr_stop_loss': 3,
    'atr_take_profit': 3,
    'bars_to_enter': 6,
    'use_dual_timeframe': True,
    'allow_same_bar_exit': False
}

# Quod Stochastic Strategy
QUOD_CONFIG = {
    'name': 'Quod Stochastic',
    'model_type': 'quod',
    'symbol': 'SPY',  # Will be overridden by command line
    'position_size': 0.1,
    'primary_timeframe': '5T',  # 5-minute bars
    # Strategy modes
    'use_stoch_reversal': True,
    'use_stoch_pullback': True,
    'use_d60_trend_entry': False,
    'use_d60_trend_exit': True,
    'use_trailing_stop': False,
    'use_force_eod': False,
    # Stochastic parameters
    'stoch_k_period': 14,
    'stoch_d_period': 3,
    'stoch_overbought': 80,
    'stoch_oversold': 20,
    # D60 trend parameters
    'd60_lookback': 60,
    'trend_threshold': 0.02,
    # Position management
    'long_tp_perc': 1.01,
    'long_sl_perc': 0.99,
    'long_trail_activation_perc': 1.005,
    'long_trail_offset_ticks': 100,
    # Exit thresholds
    'rev_long_exit_count': 3,
    'rev_long_exit_threshold': 80.0,
    'rev_short_exit_count': 3,
    'rev_short_exit_threshold': 20.0,
    'pullback_long_exit_count': 2,
    'pullback_long_exit_threshold': 80.0,
    'pullback_short_exit_count': 2,
    'pullback_short_exit_threshold': 20.0,
    # End of day exit
    'end_of_day_hour': 16,
    'end_of_day_minute': 0,
    'allow_same_bar_exit': True
}

# 5-Minute Strategy Configurations

# BB-RSI-ADX 5-Minute Configuration
BB_RSI_ADX_5MIN_CONFIG = {
    'name': 'BB-RSI-ADX Momentum (5min)',
    'model_type': 'bb_rsi_adx',
    'symbol': 'SPY',  # Will be overridden by command line
    'position_size': 0.05,  # Smaller position size for higher frequency
    'primary_timeframe': '5T',
    # Strategy parameters optimized for 5-minute bars
    'bb_period': 20,
    'bb_std': 2,
    'rsi_period': 9,  # Faster RSI for 5-min bars
    'rsi_overbought': 75,  # Slightly higher threshold
    'rsi_oversold': 25,  # Slightly lower threshold
    'supertrend_period': 10,
    'supertrend_multiplier': 2.5,  # Tighter multiplier
    'adx_primary_threshold': 15,  # Lower threshold for 5-min
    'adx_secondary_threshold': 30,  # Lower secondary threshold
    'atr_stop_multiplier': 4,  # Tighter stops for 5-min
    'bars_to_enter': 1,
    'allow_same_bar_exit': True  # Allow quick exits
}

# TEMA 5-Minute Configuration
TEMA_5MIN_CONFIG = {
    'name': 'TEMA Trend Following (5min)',
    'model_type': 'tema',
    'symbol': 'SPY',  # Will be overridden by command line
    'position_size': 0.05,
    'primary_timeframe': '5T',
    # Strategy parameters optimized for 5-minute bars
    'tema_primary_fast': 8,  # Faster periods for 5-min
    'tema_primary_slow': 26,
    'tema_secondary_fast': 12,
    'tema_secondary_slow': 35,
    'adx_threshold': 25,  # Lower threshold
    'cmo_long_threshold': 30,  # Less extreme thresholds
    'cmo_short_threshold': -30,
    'atr_entry_offset': 0.5,  # Tighter entry
    'atr_stop_loss': 2,  # Tighter stops
    'atr_take_profit': 2,  # Smaller targets
    'bars_to_enter': 3,  # Shorter persistence
    'use_dual_timeframe': False,  # Single timeframe for 5-min
    'allow_same_bar_exit': True
}

HYBRID_XGB_TEMA_5MIN_CONFIG = {
    'name': 'Hybrid XGBoost+TEMA (5min)',
    'model_type': 'hybrid_momentum',
    'ml_model_type': 'xgboost',
    'ml_model_params': XGBOOST_5MIN_CONFIG['model_params'],
    'momentum_strategy': 'tema',
    'agree_only': True,
    'weights': (0.3, 0.7),
    'timeframe': '5min'
}

# All strategy configurations
STRATEGY_CONFIGS = {
    'decision_tree': DECISION_TREE_CONFIG,
    'decision_tree_calibrated': DECISION_TREE_CALIBRATED_CONFIG,
    'random_forest': RANDOM_FOREST_CONFIG,
    'random_forest_calibrated': RANDOM_FOREST_CALIBRATED_CONFIG,
    'xgboost_fixed': XGBOOST_FIXED_CONFIG,
    'xgboost_confidence': XGBOOST_CONFIDENCE_CONFIG,
    'decision_tree_5min': DECISION_TREE_5MIN_CONFIG,
    'random_forest_5min': RANDOM_FOREST_5MIN_CONFIG,
    'xgboost_5min': XGBOOST_5MIN_CONFIG,
    'transformer_5min': TRANSFORMER_5MIN_CONFIG,
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
    },
    # Momentum strategies
    'bb_rsi_adx': BB_RSI_ADX_CONFIG,
    'tema': TEMA_CONFIG,
    'quod': QUOD_CONFIG,
    # 5-minute momentum strategies
    'bb_rsi_adx_5min': BB_RSI_ADX_5MIN_CONFIG,
    'tema_5min': TEMA_5MIN_CONFIG,
    'hybrid_xgb_tema_5min': HYBRID_XGB_TEMA_5MIN_CONFIG,
    'multi_tf_tema': {
        'name': 'Multi-Timeframe TEMA',
        'model_type': 'multi_timeframe',
        'base_strategy': 'tema',
        'timeframes': ['5min', '15min', '1h', '1D'],
        'combine_method': 'average'
    },
    # Meta-strategy configuration
    'meta_strategy': {
        'name': 'meta_strategy',
        'model_type': None,  # Meta-strategy doesn't use ML models directly
        'model_params': {},
        'lookback': 20,
        'use_momentum_features': True,
        'use_multi_timeframe': True,
        'timeframes': ['5min'],
        'selection_method': 'performance',  # 'performance' or 'regime'
        'performance_window': 390,  # 5 days for 5-minute data (78 bars/day)
        'switch_cooldown': 78,  # 1 day cooldown for 5-minute data
        'strategies': ['quod', 'tema', 'bb_rsi_adx'],
        'regime_method': 'trend_volatility',
        'regime_map': {
            'strong_uptrend': 'tema',
            'uptrend': 'tema',
            'weak_uptrend': 'bb_rsi_adx',
            'volatile_neutral': 'bb_rsi_adx',
            'neutral': 'quod',
            'low_vol_neutral': 'quod',
            'weak_downtrend': 'bb_rsi_adx',
            'downtrend': 'tema',
            'strong_downtrend': 'tema'
        },
        'regime_override': True
    }
}