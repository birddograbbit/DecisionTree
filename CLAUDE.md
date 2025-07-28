# DecisionTree Project Context for Claude

## Project Overview
The DecisionTree project is a hybrid ML trading system that combines traditional machine learning models (Decision Trees, Random Forest, XGBoost) with transformer-based deep learning for stock price prediction and trading signal generation.

**Primary Goal**: Achieve v0.2 performance targets:
- Annual Return: 20%
- Sharpe Ratio: 0.75
- Max Drawdown: < 20%

**Current Status**: 
- Best Sharpe: 0.37 (Decision Tree)
- Key Issue: Low trading frequency (5-6 trades) due to conservative thresholds

## Project Structure
```
DecisionTree/
├── src/
│   ├── models/
│   │   ├── base_model.py              # Abstract base class for all models
│   │   ├── decision_tree_model.py     # Decision tree implementation
│   │   ├── random_forest_model.py     # Random forest implementation
│   │   ├── xgboost_model.py          # XGBoost with focal loss support
│   │   ├── model_factory.py          # Factory pattern for model creation
│   │   ├── hyperparameter_optimization.py  # Optuna-based optimization
│   │   ├── hyperparameter_manager.py # Manages optimized parameters
│   │   ├── transformer/              # Transformer model components
│   │   │   ├── transformer_model.py  # Core transformer architecture
│   │   │   ├── transformer_wrapper.py # Wrapper for BaseModel interface
│   │   │   └── sequence_preparation.py # Data preparation for transformer
│   │   └── ensemble/
│   │       ├── stacking_model.py     # Stacking ensemble
│   │       └── hybrid_strategy.py    # Hybrid model combining approaches
│   ├── strategies/
│   │   ├── trend_following.py       # Main trading strategy
│   │   └── regime_adaptive_strategy.py # Regime-based adaptive strategy
│   ├── features/
│   │   ├── feature_engineering.py   # Feature creation and scaling
│   │   └── indicators.py            # Technical indicators
│   ├── data/
│   │   └── preprocessing.py         # Data loading and preprocessing
│   └── utils/
│       └── metrics.py               # Performance metrics
├── data/
│   └── raw/
│       ├── historical_data_STOCK_SPY_1_day2000-2009.csv
│       └── historical_data_STOCK_SPY_1_day2010-2025.csv
├── docs/
│   ├── fixes/
│   │   ├── focal_loss_implementation.md  # XGBoost focal loss fix
│   │   └── regime_adaptive_date_fix.md   # Date ambiguity fix
│   ├── next_steps_recommendations.md      # Comprehensive improvement plan
│   └── immediate_action_plan.md          # Quick wins implementation
├── config.py                        # System configuration
├── strategy_configs.py              # Strategy configurations
├── strategy_runner.py               # Main execution script
└── requirements.txt                 # Python dependencies
```

## Recent Fixes Implemented

### 1. XGBoost Focal Loss Implementation
- **Problem**: XGBoost parameter warnings for focal loss
- **Solution**: Implemented focal loss as custom objective function
- **Location**: `src/models/xgboost_model.py`
- **Impact**: Better handling of class imbalance (57% up vs 43% down days)

### 2. RegimeAdaptiveStrategy Date Fix
- **Problem**: Pandas ambiguity error with 'date' as both index and column
- **Solution**: Proper date handling in generate_signals method
- **Location**: `src/strategies/regime_adaptive_strategy.py`
- **Impact**: Enables regime-based trading

## Key Technical Details

### Models Available
1. **Decision Tree**: Simple, interpretable, best current Sharpe (0.37)
2. **Random Forest**: Ensemble of decision trees
3. **XGBoost**: Gradient boosting with focal loss support
4. **Transformer**: Deep learning for sequence prediction
5. **Stacking Ensemble**: Combines multiple models
6. **Hybrid**: Combines decision tree with transformer

### Strategy Types
1. **TrendFollowingStrategy**: Base strategy using probability thresholds
2. **RegimeAdaptiveStrategy**: Adapts parameters based on market regimes

### Configuration Parameters
```python
# From config.py
CONFIDENCE_THRESHOLDS = {
    'BUY': 0.65,   # Too conservative - recommended: 0.55
    'SELL': 0.35   # Too conservative - recommended: 0.45
}
LOOKBACK_PERIOD = 10
TRANSACTION_COST = 0.001  # 0.1% per trade
```

## Performance Analysis Results
From `optimized_comparison/strategy_comparison.csv`:
- Decision Tree: 42% return, 0.37 Sharpe, 5 trades
- Random Forest: 16% return, 0.01 Sharpe, 6 trades
- XGBoost: 24% return, 0.06 Sharpe, 63 trades
- Regime Adaptive: Working but limited by few trades

## Main Issues to Address
1. **Wrong Optimization Metric**: Using accuracy instead of Sharpe ratio
2. **Conservative Thresholds**: 0.65/0.35 causing very few trades
3. **Limited Features**: Basic indicators, no momentum-specific features
4. **No Multi-Timeframe Analysis**: Single timeframe limiting signal quality

## Recommended Next Steps (Priority Order)

### 1. Implement Strategy Adapter Pattern (HIGH)
Create flexible architecture to test sophisticated momentum strategies:
- BB-RSI-ADX: Bollinger Bands + RSI extremes + ADX trend strength
- TEMA: Triple Exponential Moving Average with trend filters
- Quod: Stochastic reversal/pullback with position management

### 2. Trading-Focused Optimization (HIGH)
Replace accuracy with Sharpe ratio in hyperparameter optimization

### 3. Adaptive Thresholds (HIGH)
Dynamic thresholds based on recent probability distribution

### 4. Enhanced Transformer Features (HIGH)
Add momentum indicators and multi-timeframe analysis

### 5. Walk-Forward Optimization (MEDIUM)
More realistic backtesting with rolling windows

## Command Examples

### Basic Usage
```bash
# Run single strategy
python strategy_runner.py --data data/raw --model random_forest --mode single --output rf_test

# Compare strategies
python strategy_runner.py --data data/raw --mode compare --output comparison_results

# Use optimized parameters
python strategy_runner.py --data data/raw --use-optimized --mode compare

# Test with focal loss
python strategy_runner.py --data data/raw --model xgboost \
    --model-params "use_focal_loss=True,focal_gamma=2.0" --mode single
```

### Feature Analysis
```bash
# Run feature importance audit
python strategy_runner.py --data data/raw --mode feature_audit --output feature_analysis
```

## External Momentum Strategies
Located in experimental directories:
- `/Users/jt/Coding/experimental/trading_strategies/bbrsiadx/`
- `/Users/jt/Coding/experimental/trading_strategies/tema_trendfollowing/`
- `/Users/jt/Coding/TWS/quod_rotation/TV_aligned/`

These contain sophisticated momentum strategies that could significantly enhance the transformer module through the strategy adapter pattern.

## Environment Details
- Platform: macOS Darwin 24.5.0
- Python: 3.13 (virtual environment at `/Users/jt/Coding/TWS/.venv`)
- Key Dependencies: pandas, scikit-learn, xgboost, torch, optuna

## Important Notes for Future Sessions
1. When running commands, always use the virtual environment
2. The system uses daily SPY data from 1999-2025
3. Transaction costs are 0.1% per trade
4. Look-ahead bias has been carefully avoided in all strategies
5. The transformer model supports both CPU and GPU (with macOS patches)

## Quick Reference for Common Tasks

### Modify Thresholds
Edit `config.py`:
```python
CONFIDENCE_THRESHOLDS = {'BUY': 0.55, 'SELL': 0.45}
```

### Add New Strategy
1. Create adapter in `src/strategies/`
2. Register in `strategy_configs.py`
3. Add to `StrategyRegistry` (when implemented)

### Run Hyperparameter Optimization
```bash
python -m src.models.hyperparameter_optimization --model random_forest --n_trials 100
```

### Test Specific Features
```python
# In strategy_runner.py or test script
model_params = {
    'use_focal_loss': True,
    'focal_gamma': 2.0,
    'use_adaptive_thresholds': 'always',
    'buy_percentile': 70,
    'sell_percentile': 30
}
```

## Contact Points
- Documentation: `/docs/` directory
- Performance results: `/optimized_comparison/` directory
- Configuration: `config.py` and `strategy_configs.py`
- Main execution: `strategy_runner.py`

This context should be provided at the start of any new conversation about the DecisionTree project to maintain continuity.