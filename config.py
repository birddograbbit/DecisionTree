# config.py
"""
Configuration settings for the decision tree trading project.
"""

# IBKR connection settings
IBKR_HOST = '127.0.0.1'
IBKR_PORT = 7497  # Use 7496 for TWS, 7497 for IB Gateway paper trading
IBKR_CLIENT_ID = 1

# Data settings
DATA_START_DATE = '20000101'  # Training data start date
DATA_TRAIN_END_DATE = '20091231'  # Training data end date
DATA_TEST_END_DATE = '20240510'  # Testing data end date

# Model settings
LOOKBACK_PERIOD = 10  # Number of days to look back for feature creation
LOOKBACK_PERIOD_5MIN = 78  # Number of 5-minute bars to look back (1 trading day)
# Note: FEATURE_COUNT removed as it was unused. Feature selection is now handled
# through the feature auditing and pruning system in feature_engineering.py

# Backtesting settings
INITIAL_CAPITAL = 100000  # Initial capital for backtesting
COMMISSION_RATE = 0.0005  # Commission rate per trade (0.05%)
SLIPPAGE_RATE = 0.0001  # Slippage rate per trade (0.01%)

# Risk management settings
MAX_POSITION_SIZE = 0.05  # Maximum position size as fraction of total capital
MAX_PORTFOLIO_ALLOCATION = 0.5  # Maximum portfolio allocation
STOP_LOSS_PCT = 0.05  # Stop loss percentage
MAX_DRAWDOWN = 0.15  # Maximum drawdown threshold
MAX_HOLDING_DAYS = 10  # Maximum holding period in days

# Hyperparameter optimization settings
OPTUNA_TRIALS = 100  # Number of trials for hyperparameter optimization
TIMESERIES_CV_SPLITS = 5  # Number of splits for time series cross-validation
RANDOM_STATE = 42  # Random seed for reproducibility


# Directory to store optimized hyperparameters
HYPERPARAMS_DIR = 'data/hyperparameters'
HYPERPARAMS_VERSIONED_DIR = f"{HYPERPARAMS_DIR}/versioned"
HYPERPARAMS_REGIME_DIR = f"{HYPERPARAMS_DIR}/regimes"

# Hyperparameter search spaces
DECISION_TREE_PARAMS = {
    'max_depth': (2, 20),  # Min and max values for max_depth
    'min_samples_split': (2, 20),  # Min and max values for min_samples_split
    'min_samples_leaf': (1, 20),  # Min and max values for min_samples_leaf
    'max_features': ['sqrt', 'log2', None],  # Options for max_features
    'criterion': ['gini', 'entropy'],  # Options for criterion
    'class_weight': ['balanced', None]  # Options for class_weight
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': (50, 500),  # Min and max values for n_estimators
    'max_depth': (2, 30),  # Min and max values for max_depth
    'min_samples_split': (2, 20),  # Min and max values for min_samples_split
    'min_samples_leaf': (1, 10),  # Min and max values for min_samples_leaf
    'max_features': ['sqrt', 'log2', None],  # Options for max_features
    'bootstrap': [True, False],  # Options for bootstrap
    'class_weight': ['balanced', 'balanced_subsample', None]  # Options for class_weight
}

XGBOOST_PARAMS = {
    'n_estimators': (50, 500),  # Min and max values for n_estimators
    'max_depth': (3, 12),  # Min and max values for max_depth
    'learning_rate': (0.01, 0.3),  # Min and max values for learning_rate (log scale)
    'subsample': (0.6, 1.0),  # Min and max values for subsample
    'colsample_bytree': (0.6, 1.0),  # Min and max values for colsample_bytree
    'gamma': (0, 5),  # Min and max values for gamma
    'min_child_weight': (1, 10),  # Min and max values for min_child_weight
    'reg_alpha': (0, 5),  # Min and max values for reg_alpha
    'reg_lambda': (0, 5),  # Min and max values for reg_lambda
    'scale_pos_weight': (1, 10),  # Min and max values for scale_pos_weight
    'use_focal_loss': [True, False],  # Whether to use focal loss
    'focal_gamma': (0.5, 5.0),  # Min and max values for focal loss gamma
    'focal_alpha': (0.1, 0.9),  # Min and max values for focal loss alpha
    'class_weight': ['balanced', None]  # Options for class_weight
}


# Transformer and hybrid model configurations
TRANSFORMER_CONFIG = {
    'default': {
        'seq_length': 30,
        'prediction_length': 1,
        'n_features': 9,
        'd_model': 64,
        'n_heads': 8,
        'n_layers': 2,
        'dropout': 0.1,
        'epochs': 20,
    },
    '5min': {
        'seq_length': 20,
        'prediction_length': 1,
        'n_features': 16,
        'd_model': 32,
        'n_heads': 4,
        'n_layers': 2,
        'dropout': 0.1,
        'epochs': 20,
    }
}

HYBRID_CONFIG = {
    'balanced': {
        'dt_weight': 0.5,
        'tf_weight': 0.5,
        'regime_adaptive': True
    }
}
# Feature audit settings
FEATURE_AUDIT_N_REPEATS = 10  # Number of times to permute each feature
TOP_N_FEATURES = 10  # Number of top features to keep after pruning
COLLINEARITY_THRESHOLD = 0.8  # Threshold for detecting collinear features

# Transaction costs
TRANSACTION_COST = 0.001  # 0.1% per trade (daily trading)
TRANSACTION_COST_5MIN = 0.0005  # 0.05% per trade (5-minute trading)
SLIPPAGE_5MIN = 0.0001  # 0.01% slippage for 5-minute trading
