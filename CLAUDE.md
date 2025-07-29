# DecisionTree Project Context for Claude

## Project Overview
The DecisionTree project is a hybrid ML trading system that combines traditional machine learning models (Decision Trees, Random Forest, XGBoost) with transformer-based deep learning for stock price prediction and trading signal generation. The system now includes sophisticated momentum strategies and a meta-strategy framework for intelligent strategy orchestration.

**Primary Goal**: Achieve v0.2 performance targets:
- Annual Return: 20%
- Sharpe Ratio: 0.75
- Max Drawdown: < 20%

**Current Status**: 
- Best Sharpe: 1.53 (Quod strategy on 5-minute data) ✅
- Meta-strategy framework implemented with performance tracking ✅
- ML strategies need optimization for intraday data
- Momentum strategies excel on 5-minute timeframe

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
│   │   ├── base_strategy.py         # Enhanced base class with adapter support
│   │   ├── trend_following.py       # Main trading strategy
│   │   ├── regime_adaptive_strategy.py # Regime-based adaptive strategy
│   │   ├── meta_strategy.py         # Meta-strategy for dynamic strategy selection
│   │   ├── strategy_registry.py     # Registry pattern for strategy management
│   │   └── adapters/
│   │       ├── __init__.py
│   │       ├── bbrsiadx_adapter.py  # BB-RSI-ADX momentum adapter
│   │       ├── tema_adapter.py      # TEMA trend following adapter
│   │       └── quod_adapter.py      # Quod mean reversion adapter
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

## Recent Implementations (July 2025)

### 1. Strategy Adapter Pattern ✅
- **Achievement**: Full implementation with 3 momentum strategies
- **Components**: BaseStrategy enhancement, StrategyRegistry, Multi-timeframe support
- **Impact**: Clean architecture for testing sophisticated strategies

### 2. 5-Minute Data Support ✅
- **Achievement**: Complete integration of intraday data
- **Components**: Data loading, timeframe-aware metrics, proper CAGR/Sharpe calculations
- **Impact**: Momentum strategies achieve 1.40-1.53 Sharpe ratios

### 3. Performance Metrics Fixes ✅
- **Problem**: CAGR showing 0.00% for 5-minute data
- **Solution**: Timeframe-aware calculations (19,656 periods/year for 5-min)
- **Impact**: Accurate performance measurement across all timeframes

### 4. Meta-Strategy Framework ✅
- **Achievement**: Dynamic strategy selection based on performance
- **Components**: 
  - MetaStrategy class with performance tracking
  - Strategy switching with cooldown periods
  - Performance window optimization (390 bars for 5-min data)
  - Registry-based performance statistics
- **Impact**: Foundation for intelligent strategy orchestration

## Key Technical Details

### Models/Strategies Available
1. **ML Models**: Decision Tree, Random Forest, XGBoost (need optimization for 5-min data)
2. **Ensemble Models**: Stacking, Transformer, Hybrid
3. **Momentum Strategies**: 
   - BB-RSI-ADX: 1.40 Sharpe on 5-min data
   - TEMA: 0.76 Sharpe on 5-min data  
   - Quod: 1.53 Sharpe on 5-min data ✅
4. **Meta-Strategy**: Dynamic selection between strategies based on:
   - Performance-based selection (default): Tracks Sharpe ratio over rolling window
   - Regime-based selection (planned): Maps market regimes to optimal strategies

### Supported Timeframes
1. **Daily**: Traditional ML models trained on this
2. **5-Minute**: Momentum strategies excel here

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

## Latest Performance Results (July 28, 2025)

### Daily Data (Poor Performance)
- Decision Tree: -38.44% return, -19.36 Sharpe
- Random Forest: -35.62% return, -17.42 Sharpe
- XGBoost: -1.68% return, -0.39 Sharpe

### 5-Minute Data (Excellent Performance)
- BB-RSI-ADX: 8.25% return, 1.40 Sharpe, 935 trades
- TEMA: 5.81% return, 0.76 Sharpe, 1,904 trades
- Quod: 14.98% return, 1.53 Sharpe, 3,195 trades ✅

## Main Issues to Address
1. **ML Strategy Optimization**: Need to retrain for 5-minute data patterns
2. **Trading-Focused Optimization**: Use Sharpe ratio instead of accuracy
3. **Regime Detection**: Implement regime-based strategy selection for meta-strategy
4. **Meta-Strategy Optimization**: Fine-tune performance window and switching parameters

## Next Steps (Priority Order)

### 1. Test and Optimize Meta-Strategy (IMMEDIATE)
- Run comprehensive tests with performance tracking enabled
- Fine-tune performance window (currently 390 bars)
- Optimize switch cooldown (currently 78 bars)
- Validate strategy switching behavior

### 2. Implement Regime-Based Selection (HIGH)
- Complete regime detection integration in meta-strategy
- Map regimes to optimal strategies based on historical performance
- Test regime-based vs performance-based selection

### 3. ML Strategy Optimization for 5-Minute Data (HIGH) 
- Retrain ML models with intraday-specific features
- Use shorter lookback periods appropriate for 5-min data
- Implement trading-focused optimization (Sharpe ratio)

### 4. Meta-Strategy Enhancements (MEDIUM)
- Add parallel strategy tracking (Option 1) for better learning
- Implement strategy confidence weighting
- Add transaction cost awareness to switching logic

### 5. Production Readiness (MEDIUM)
- Add real-time performance monitoring
- Implement strategy health checks
- Create automated retraining pipeline

## Command Examples

### Daily Data Testing
```bash
# Compare all strategies on daily data
python strategy_runner.py --data data/raw --mode compare --include-momentum --output daily_comparison
```

### 5-Minute Data Testing 
```bash
# Test single momentum strategy
python strategy_runner.py --data data/raw --model quod --mode single --output quod_5min --timeframe 5min

# Test meta-strategy with performance tracking
python strategy_runner.py --data data/raw --model meta_strategy --mode single --output meta_test --timeframe 5min

# Compare all strategies on 5-minute data
python strategy_runner.py --data data/raw --mode compare --include-momentum --output 5min_comparison --timeframe 5min
```

### Feature Analysis
```bash
# Run feature importance audit
python strategy_runner.py --data data/raw --mode audit --audit-model random_forest --output feature_analysis
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
2. The system supports both daily and 5-minute SPY data
3. Transaction costs: 0.1% (daily), 0.05% (5-minute)
4. Quod strategy achieved 1.53 Sharpe ratio, exceeding v0.2 target
5. Meta-strategy framework implemented with:
   - Performance tracking (390-bar window for 5-min data)
   - Dynamic strategy switching (78-bar cooldown)
   - Registry-based performance management
6. ML strategies need retraining for 5-minute data patterns
7. Momentum strategies work best on intraday timeframes

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