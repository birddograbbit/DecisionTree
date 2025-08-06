# Ensemble-Based Trading Strategy
## Advanced Trading System for S&P 500 with IBKR

## Table of Contents
1. [Project Overview](#project-overview)
2. [Current System Status](#current-system-status)
3. [System Architecture](#system-architecture)
4. [Project File Structure](#project-file-structure)
5. [User Guide](#user-guide)
6. [Feature Details](#feature-details)
7. [Model Development](#model-development)
8. [Backtesting Framework](#backtesting-framework)
9. [Risk Management](#risk-management)
10. [Performance Evaluation](#performance-evaluation)
11. [Development Roadmap](#development-roadmap)
12. [Appendices](#appendices)

## Project Overview

### Objectives
- Implement an ensemble-based machine learning system for trading S&P 500 stocks
- Develop a modular pipeline from data acquisition to live trading
- Achieve a superior CAGR to maximum drawdown ratio compared to S&P 500 buy-and-hold
- Create a framework that allows for continuous improvement and model updating
- Support multiple model types through a unified interface
- Adapt to market regimes for optimized trading performance

### Success Metrics
- CAGR/Max Drawdown ratio > 0.40 (compared to S&P 500's ~0.18)
- Accuracy of directional prediction > 60% 
- Strategy Sharpe ratio > 1.0
- Maximum drawdown < 25%
- Ensemble model outperforms single decision tree classifier

## Current System Status

The trading system has undergone significant development and currently includes the following completed components:

### Completed Features
1. **Core Engine-Based Architecture**
   - Modular engine design with clear separation of responsibilities
   - Consistent interfaces between components
   - Plugin architecture for easy extension with new models and strategies

2. **Multiple Model Support**
   - BaseModel interface with concrete implementations
   - Support for Decision Tree, Random Forest, and XGBoost models
   - Model Factory pattern for easy model creation

3. **Strategy Framework**
   - BaseStrategy interface for consistent strategy implementation
   - TrendFollowingStrategy implementation
   - RegimeAdaptiveStrategy for adapting to market conditions

4. **Market Regime Detection**
   - Versatile RegimeDetector for identifying market states
   - Multiple detection methods (trend-volatility, MA crossover, etc.)
   - Statistics generation for regime-specific analysis

5. **Performance Analysis**
   - Comprehensive backtesting framework
   - Detailed performance metrics
   - Visualization tools for analysis

### Recent Enhancements
1. **Market Regime Detection and Adaptation**
   - Added RegimeDetector class to identify different market states
   - Implemented RegimeAdaptiveStrategy to adjust trading parameters based on regime
   - Created visualization tools for regime analysis

2. **Model Stacking and Ensemble Methods**
   - Implemented StackingModel for combining multiple base models
   - Added model performance comparison functionality
   - Created tools for analyzing ensemble performance

### Test Results (2025-07-27)
Comprehensive testing reveals the system is functional but underperforming v0.2 targets:

1. **Model Performance vs Targets**
   - All models operational but below performance thresholds
   - Best Sharpe Ratio: 0.368 (Decision Tree) vs 0.7 target
   - Best CAGR/DD Ratio: 0.299 (Decision Tree) vs 0.35 target
   - Best CV Accuracy: 53.4% (Stacking) vs 57% target
   - Test coverage: 14.61% vs 25% threshold

2. **Individual Model Results**
   - Decision Tree: Sharpe 0.368, CAGR/DD 0.299, 5 trades
   - Random Forest: Sharpe 0.009, CAGR/DD 0.129, 6 trades
   - XGBoost: Sharpe 0.062, CAGR/DD 0.095, 63 trades
   - Stacking Ensemble: Sharpe -0.070, CAGR/DD 0.131, 5 trades

3. **Transformer Performance**
   - Latency: 11.24ms (well below 100ms target)
   - Integration with hybrid strategy confirmed working

### Known Bugs and Issues
1. **RegimeAdaptiveStrategy Date Ambiguity Error**
   - Error: "'date' is both an index level and a column label, which is ambiguous"
   - Occurs during regime detection initialization
   - Needs fix in data preprocessing logic

2. **XGBoost Focal Loss Warnings**
   - Parameters "focal_alpha", "focal_gamma", "use_focal_loss" not recognized
   - Expected behavior but should be cleaned up

3. **Low Test Coverage**
   - Current: 14.61% vs 25% minimum requirement
   - Missing comprehensive test coverage for all modules

4. **Performance Issues**
   - All models significantly underperforming v0.2 targets
   - Need hyperparameter optimization and feature engineering improvements

### Current Limitations
1. Limited real-world testing of regime-adaptive strategies
2. Need for further optimization of regime-specific parameters
3. Interactive dashboard not yet implemented
4. Limited hyperparameter optimization framework
5. Models not meeting v0.2 performance targets

## System Architecture

### High-Level Components (Engine-Based Design)
1. **Data Engine**
   - Historical data retrieval and storage
   - Live market data streaming
   - Data preprocessing and cleaning

2. **Feature Engine**
   - Technical indicator calculation
   - Feature extraction and engineering
   - Feature selection and transformation
   - Market regime detection

3. **Model Engine**
   - Model interface for multiple model types
   - Model factory pattern for unified creation
   - Model training, validation, and testing
   - Ensemble model support

4. **Signal Engine**
   - Convert predictions to trading signals
   - Signal filtering and validation
   - Position sizing and timing rules
   - Regime-adaptive signal generation

5. **Execution Engine**
   - Backtesting engine
   - Order execution
   - Position and risk management

6. **Strategy Engine**
   - Coordinate workflow between engines
   - Strategy interface for multiple strategies
   - Regime-adaptive strategies
   - Performance tracking and optimization

### Technology Stack
- **Programming Language**: Python 3.9+
- **Trading Connection**: ib_insync library for IBKR API
- **Data Analysis**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost
- **Data Visualization**: matplotlib, seaborn
- **Data Storage**: SQLite for development, PostgreSQL for production
- **Version Control**: Git
- **Environment Management**: Conda or venv

## Project File Structure

The project has the following file structure. Core files are marked with (C) and temporary/test files are marked with (T).

```
decision_tree_trading/
│
├── data/                               # Data directory
│   ├── raw/                            # (C) Raw data files
│   │   ├── historical_data_STOCK_SPY_1_day2000-2009.csv
│   │   └── historical_data_STOCK_SPY_1_day2010-2025.csv
│   ├── processed/                      # (C) Processed data
│   └── models/                         # (C) Trained model storage
│
├── src/                                # (C) Source code
│   ├── data/                           # (C) Data handling
│   │   ├── __init__.py
│   │   ├── data_acquisition.py
│   │   └── preprocessing.py
│   │
│   ├── features/                       # (C) Feature engineering
│   │   ├── __init__.py
│   │   ├── indicators.py
│   │   ├── feature_engineering.py
│   │   └── regime_detection.py         # (NEW) Market regime detection
│   │
│   ├── models/                         # (C) Model implementations
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── decision_tree_model.py
│   │   ├── random_forest_model.py
│   │   ├── xgboost_model.py
│   │   ├── stacking_model.py           # (NEW) Model stacking implementation
│   │   ├── stacked_model.py            # (NEW) Alternative stacking implementation
│   │   └── model_factory.py
│   │
│   ├── engines/                        # (C) Engine components
│   │   ├── __init__.py
│   │   ├── model_engine.py
│   │   └── signal_engine.py
│   │
│   ├── strategies/                     # (C) Strategy implementations
│   │   ├── __init__.py
│   │   ├── base_strategy.py
│   │   ├── trend_following.py
│   │   └── regime_adaptive_strategy.py # (NEW) Regime-adaptive strategy
│   │
│   ├── backtesting/                    # (C) Backtesting framework
│   │   ├── __init__.py
│   │   ├── engine.py
│   │   └── performance.py
│   │
│   └── trading/                        # (C) Trading implementations
│       ├── __init__.py
│       ├── live_trading.py
│       └── risk_management.py
│
├── results/                            # (C) Backtest results
├── comparison_results/                 # (C) Strategy comparison results
├── regime_detection_results/           # (C) Regime detection results
├── regime_strategy_results/            # (C) Regime strategy results
├── stacking_strategy_test/             # (T) Stacking test results
│
├── config.py                           # (C) Configuration settings
├── strategy_runner.py                  # (C) Main strategy execution script
├── test_regime_detection.py            # (C) Regime detection test script
├── test_regime_adaptive_strategy.py    # (C) Regime strategy test script
└── Decision_Tree_Classifier_Strategy.md # (C) Project documentation
│
│ # Temporary/Test files (can be archived):
├── enhanced_stacking_model_test.py     # (T) Testing script
├── fix_and_backtest.py                 # (T) Fix script
├── fix_walkforward.py                  # (T) Fix script
├── simple_backtest.py                  # (T) Simple test
├── simplified_performance.py           # (T) Simple performance test
├── test_simple_stacking.py             # (T) Stacking test
├── test_stacking_model.py              # (T) Model test
├── test_stacking_only.py               # (T) Stacking test
├── test_stacking_strategy.py           # (T) Strategy test
├── test_upgrade.py                     # (T) Upgrade test
├── verify_fixes.py                     # (T) Verification script
└── visualize_results.py                # (T) Visualization script
```

### Core Files vs. Temporary Files

The project contains two types of files:

1. **Core Files (C)**: Essential components of the trading system architecture. These files should be maintained, updated, and documented.

2. **Temporary/Test Files (T)**: Files used for testing specific functionality or fixing issues. These can be archived or removed to clean up the project directory.

## User Guide

This section provides instructions for using the trading system.

### Installation

1. Clone the repository and set up the environment:
```bash
git clone <repository-url>
cd decision_tree_trading
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure Interactive Brokers (IBKR):
   - Install TWS or IB Gateway
   - Configure API settings (see Appendix C)
   - Ensure the platform is running when executing live trading scripts

### Basic Usage

#### Running Strategy Comparison

The simplest way to use the system is through the `strategy_runner.py` script, which allows comparing different strategies:

```bash
# Run in comparison mode
python strategy_runner.py --data data/raw --mode compare --output comparison_results

# Run a single strategy
python strategy_runner.py --data data/raw --mode single --model random_forest --output results
```

Arguments:
- `--data`: Path to data directory or file
- `--mode`: 'single' for one strategy, 'compare' for multiple
- `--model`: Model type ('decision_tree', 'random_forest', 'xgboost', 'stacking')
- `--strategy`: Strategy type ('trend_following', 'regime_adaptive')
- `--output`: Directory to save results
- `--train-end`: End date for training data (format: YYYY-MM-DD)
- `--symbol`: Trading symbol (default: 'SPY')

#### Testing Regime Detection

To test and visualize market regime detection:

```bash
python test_regime_detection.py --data data/raw/historical_data_STOCK_SPY_1_day2010-2025.csv --output regime_detection_results
```

This script will:
1. Load and process the price data
2. Apply different regime detection methods
3. Generate visualizations and statistics
4. Compare regime detection methods

#### Testing Regime-Adaptive Strategy

To test the regime-adaptive strategy against the standard trend-following strategy:

```bash
python test_regime_adaptive_strategy.py --data data/raw/historical_data_STOCK_SPY_1_day2010-2025.csv --output regime_strategy_results
```

This script will:
1. Load and process the price data
2. Run both strategies with the same model configuration
3. Compare performance metrics
4. Generate visualizations of results

### Advanced Usage

#### Using Different Models

The system supports multiple model types:

```bash
# Use Decision Tree
python strategy_runner.py --data data/raw --model decision_tree

# Use Random Forest
python strategy_runner.py --data data/raw --model random_forest

# Use XGBoost
python strategy_runner.py --data data/raw --model xgboost

# Use Stacking Ensemble
python strategy_runner.py --data data/raw --model stacking
```

#### Position Sizing Options

`TrendFollowingStrategy` supports two position sizing modes via the
`position_sizing` configuration key:

- **fixed** – every trade uses the same size.
- **confidence** – position size scales with prediction confidence (default).

Example configuration snippet:

```python
{
    'name': 'XGBoost',
    'model_type': 'xgboost',
    'model_params': {...},
    'position_sizing': 'fixed'
}
```

#### Creating Custom Strategies

To create a custom strategy:

1. Create a new file in `src/strategies/` (e.g., `my_strategy.py`)
2. Implement the `BaseStrategy` interface
3. Add your strategy to the `__init__.py` file in the strategies directory
4. Use your strategy in the runner script

Example custom strategy skeleton:

```python
from .base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def initialize(self, config):
        # Initialize strategy parameters
        pass
        
    def generate_features(self, data):
        # Generate features for the strategy
        pass
        
    def generate_signals(self, features, predictions, dates):
        # Generate trading signals
        pass
        
    def backtest(self, data, train_data=None, test_data=None):
        # Run backtest for the strategy
        pass
```

## Feature Details

### Market Regime Detection

The system includes a versatile regime detection module (`src/features/regime_detection.py`) that can identify different market states. This is useful for adapting trading strategies to changing market conditions.

#### Available Methods

1. **Trend-Volatility Regime**
   - Combines trend direction and volatility level
   - Categorizes markets into 9 different regimes (e.g., strong_uptrend, weak_downtrend, etc.)
   - Parameters: fast_window, slow_window, vol_window, vol_threshold

2. **Moving Average Crossover Regime**
   - Uses multiple moving average relationships to identify trends
   - Categorizes markets into 5 regimes (e.g., strong_uptrend, neutral, strong_downtrend, etc.)
   - Parameters: short_window, medium_window, long_window

3. **Volatility Regime**
   - Focuses only on volatility levels
   - Categorizes markets into 3 regimes (high_volatility, normal_volatility, low_volatility)
   - Parameters: vol_window, high_vol_threshold, low_vol_threshold

4. **Statistical Regime**
   - Uses statistical measures like z-score and moving average slope
   - Categorizes markets into 9 regimes based on trend and mean-reversion characteristics
   - Parameters: ma_window, std_window, z_threshold, slope_threshold

#### Usage Example

```python
from src.features.regime_detection import RegimeDetector

# Create regime detector with trend-volatility method
detector = RegimeDetector(
    method='trend_volatility',
    fast_window=20,
    slow_window=50,
    vol_window=20,
    vol_threshold=0.75
)

# Detect regimes
regime_data = detector.detect_regime(price_data)

# Get current regime
current_regime = detector.get_current_regime()

# Get regime statistics
regime_stats = detector.get_regime_stats()

# Visualize regimes
fig = detector.plot_regimes()
```

### Model Stacking

The system includes model stacking functionality (`src/models/stacking_model.py`) for combining multiple models to improve performance.

#### Stacking Architecture

1. **Base Models**: Multiple models (Decision Tree, Random Forest, XGBoost) are trained on the same data.
2. **Meta-Features**: Predictions from base models are used as features for the meta-model.
3. **Meta-Model**: A model that learns how to combine base model predictions.

#### Usage Example

```python
from src.models.model_factory import ModelFactory

# Create stacking model
stacking_model = ModelFactory.create_model(
    'stacking',
    base_models=[
        {'model_type': 'decision_tree', 'model_params': {'max_depth': 5}},
        {'model_type': 'random_forest', 'model_params': {'n_estimators': 100, 'max_depth': 5}},
        {'model_type': 'xgboost', 'model_params': {'n_estimators': 100, 'max_depth': 5}}
    ],
    meta_model_type='logistic_regression',
    meta_model_params={'C': 1.0},
    cv=5,
    use_features=False
)

# Train the model
stacking_model.train(X_train, y_train)

# Make predictions
predictions = stacking_model.predict(X_test)
```

## Model Development

The system supports multiple model types through a unified `BaseModel` interface. This allows easy extension with new model types while maintaining a consistent API.

### Model Interface

All models implement the following interface:

```python
class BaseModel(ABC):
    @abstractmethod
    def train(self, X, y):
        """Train the model on given data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Generate predictions for given features."""
        pass

    @abstractmethod
    def get_feature_importance(self):
        """Return feature importance scores."""
        pass

    @abstractmethod
    def save(self, path):
        """Save model to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        """Load model from disk."""
        pass
```

### Model Factory

Models are created through the `ModelFactory` class:

```python
from src.models.model_factory import ModelFactory

# Create Decision Tree model
dt_model = ModelFactory.create_model('decision_tree', max_depth=5)

# Create Random Forest model
rf_model = ModelFactory.create_model('random_forest', n_estimators=100, max_depth=5)

# Create XGBoost model
xgb_model = ModelFactory.create_model('xgboost', n_estimators=100, max_depth=5, learning_rate=0.1)
```

### Model Engine

The `ModelEngine` class provides a high-level API for model training, evaluation, and prediction:

```python
from src.engines.model_engine import ModelEngine

# Create model engine
engine = ModelEngine(model_type='random_forest', model_params={'n_estimators': 100})

# Train model
engine.train(X_train, y_train, cross_validation=True, cv=5)

# Make predictions
predictions = engine.predict(X_test)

# Get feature importance
importances = engine.get_feature_importance(top_n=10)

# Save model
engine.save('models/random_forest_model.pkl')
```

## Backtesting Framework

The system includes a comprehensive backtesting framework for evaluating strategy performance.

### Backtesting Process

1. **Data Preparation**: Historical data is loaded and preprocessed.
2. **Feature Engineering**: Features are generated from price data.
3. **Model Training**: Models are trained on historical data.
4. **Signal Generation**: Trading signals are generated from model predictions.
5. **Backtest Execution**: Signals are applied to historical data to simulate trading.
6. **Performance Evaluation**: Results are analyzed and visualized.

### Performance Metrics

The backtesting framework calculates various performance metrics:

- **Return Metrics**: Total return, CAGR, annualized return
- **Risk Metrics**: Maximum drawdown, volatility, downside deviation
- **Risk-adjusted Metrics**: Sharpe ratio, Sortino ratio, CAGR/Max DD ratio
- **Trade Metrics**: Win rate, average win, average loss, profit factor
- **Model Metrics**: Accuracy, precision, recall, F1 score, AUC

## Risk Management

The system includes a comprehensive risk management framework to control trading risk.

### Risk Management Rules

1. **Position Sizing**: Limit individual position sizes to 5% of total capital
2. **Portfolio Allocation**: Maximum 50% of portfolio invested at any time
3. **Stop Loss**: Implement 5% stop loss on individual positions
4. **Maximum Drawdown Threshold**: Pause trading if drawdown exceeds 15%
5. **Correlation Control**: Ensure positions are not highly correlated
6. **Volatility Adjustment**: Reduce position sizes during high market volatility
7. **Time-based Exit**: Maximum holding period of 10 trading days

### Regime-Adaptive Risk Management

The regime-adaptive strategy adjusts risk parameters based on the detected market regime:

1. **Strong Uptrend**: More aggressive position sizing, lower entry threshold
2. **Weak Uptrend**: Moderate position sizing, standard entry threshold
3. **Neutral**: Reduced position sizing, higher entry threshold
4. **Weak Downtrend**: Minimal position sizing, very high entry threshold
5. **Strong Downtrend**: Extremely conservative or avoid trading
6. **Custom Thresholds**: Specify `buy_threshold` and `sell_threshold` in
   `regime_params` to override the default 0.65/0.35 levels. Setting
   `use_low_thresholds: true` applies 0.55 and 0.45 by default if specific
   values are not provided.

## Performance Evaluation

The system includes tools for evaluating and comparing strategy performance.

### Visualization

Performance visualization tools include:

1. **Equity Curve**: Plot of portfolio value over time
2. **Drawdown Chart**: Visualization of portfolio drawdowns
3. **Trade Returns**: Bar chart of individual trade returns
4. **ROC Curves**: Comparison of model prediction accuracy
5. **Regime Performance**: Analysis of performance in different market regimes

### Benchmark Comparison

Strategies are compared against a buy-and-hold benchmark to evaluate their added value.

## Development Roadmap

Future development will focus on enhancing the system's capabilities and performance:

### Phase 1: Optimization and Robustness (1-2 Months)
1. **Hyperparameter Optimization Framework**
   - Implement automated grid search and Bayesian optimization
   - Create cross-validation framework for time series data
   - Develop parameter sensitivity analysis tools

2. **Enhanced Regime Detection**
   - Implement additional regime detection methods (e.g., Hidden Markov Models)
   - Optimize regime-specific parameters
   - Create regime transition probability analysis

3. **Performance Improvements**
   - Profile and optimize CPU/memory usage
   - Implement parallel processing for model training and backtesting
   - Optimize data storage and retrieval

### Phase 2: Advanced Features (2-3 Months)
1. **Interactive Dashboard**
   - Create web-based dashboard for monitoring
   - Implement real-time performance visualization
   - Develop portfolio management interface

2. **Reinforcement Learning Integration**
   - Implement RL-based strategy optimizer
   - Create deep Q-network for signal generation
   - Develop policy gradient methods for position sizing

3. **Alternative Data Sources**
   - Add support for sentiment analysis
   - Implement news event impact analysis
   - Create earnings surprise indicators

### Phase 3: Production and Scaling (2-3 Months)
1. **Multi-Asset Support**
   - Extend to multi-asset class trading
   - Implement portfolio optimization
   - Create cross-asset signals

2. **Cloud Deployment**
   - Set up cloud infrastructure for 24/7 operation
   - Implement monitoring and alerting
   - Create disaster recovery protocols

3. **Performance Tracking**
   - Implement automated performance reporting
   - Create attribution analysis tools
   - Develop strategy drift detection

## Appendices

### Appendix A: Required Python Packages

```
# Core packages
ib_insync>=0.9.70      # Interactive Brokers API
pandas>=1.3.5          # Data manipulation
numpy>=1.21.5          # Numerical operations
scikit-learn>=1.0.2    # Machine learning (Decision Tree, Random Forest)
matplotlib>=3.5.1      # Visualization
seaborn>=0.11.2        # Enhanced visualization

# Optional packages
xgboost>=1.6.1         # XGBoost models
sqlalchemy>=1.4.36     # Database connection
tqdm>=4.64.0           # Progress bars
joblib>=1.1.0          # Parallel processing
```

### Appendix B: IBKR API Setup

1. **Download and Install TWS or IB Gateway**:
   - TWS: https://www.interactivebrokers.com/en/index.php?f=16040
   - IB Gateway: https://www.interactivebrokers.com/en/index.php?f=16457

2. **Configure API Settings**:
   - Open TWS or IB Gateway
   - Go to File > Global Configuration > API > Settings
   - Enable "Enable ActiveX and Socket Clients"
   - Set Socket port (default: 7496 for TWS, 4001 for IB Gateway)
   - Check "Allow connections from localhost only" for security

3. **API Connection Testing**:
```python
from ib_insync import *

# Connect to TWS or IB Gateway
ib = IB()
ib.connect('127.0.0.1', 7496, clientId=1)  # Use 7497 for paper trading

# Check if connected
print(f"Connected: {ib.isConnected()}")

# Get account information
account_summary = ib.accountSummary()
print(account_summary)

# Disconnect
ib.disconnect()
```

### Appendix C: Common Issues and Troubleshooting

1. **IBKR Connection Issues**:
   - Ensure TWS or IB Gateway is running
   - Check that API settings are correctly configured
   - Verify the port number (7496 for TWS live, 7497 for TWS paper, 4001 for Gateway live, 4002 for Gateway paper)
   - Check for other clients using the same clientId

2. **Data Retrieval Issues**:
   - IBKR has limitations on historical data requests
   - Use the proper contract specifications
   - Handle request pacing requirements
   - Consider using local data cache

3. **Model Performance Issues**:
   - Ensure feature engineering is appropriate
   - Check for data leakage in cross-validation
   - Consider market regime changes
   - Use appropriate model hyperparameters

4. **Backtesting Issues**:
   - Account for survivorship bias
   - Include transaction costs and slippage
   - Be cautious of overfitting
   - Validate results across different time periods
