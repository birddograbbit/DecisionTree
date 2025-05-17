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
FEATURE_COUNT = 2  # Number of features to use (can be adjusted)

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