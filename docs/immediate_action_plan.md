# Immediate Action Plan for Performance Improvement

## Quick Wins (1-2 days each)

### 1. Lower Signal Thresholds
**File**: `config.py`
```python
# Change from:
CONFIDENCE_THRESHOLDS = {
    'BUY': 0.65,
    'SELL': 0.35
}

# To:
CONFIDENCE_THRESHOLDS = {
    'BUY': 0.55,
    'SELL': 0.45
}
```
**Expected Result**: Increase trades from 5-6 to 15-20

### 2. Enable Adaptive Thresholds
**File**: `strategy_runner.py`
```python
# When creating strategies, ensure:
'use_adaptive_thresholds': 'always'  # Not 'auto' or 'never'
'buy_percentile': 70    # More aggressive than 80
'sell_percentile': 30   # More aggressive than 20
```
**Expected Result**: Dynamic threshold adjustment based on market conditions

### 3. Test Focal Loss with XGBoost
**Command**:
```bash
# Create new config with focal loss
python strategy_runner.py \
    --data data/raw \
    --model xgboost \
    --model-params "use_focal_loss=True,focal_gamma=2.0,focal_alpha=auto" \
    --mode single \
    --output xgboost_focal_aggressive
```
**Expected Result**: Better balanced buy/sell signals

### 4. Quick Sharpe Optimization Test
**File**: Create `test_sharpe_optimization.py`
```python
from src.models.hyperparameter_optimization import optimize_hyperparameters
from src.data.preprocessing import preprocess_data
import pandas as pd

# Load and prepare data
df = pd.read_csv('data/raw/historical_data_STOCK_SPY_1_day2010-2025.csv', 
                 index_col=0, parse_dates=True)
df = preprocess_data(df)

# Run optimization with custom scoring
results = optimize_hyperparameters(
    df, 
    'random_forest',
    scoring='sharpe',  # If implemented
    n_trials=50
)

print(f"Best parameters for Sharpe: {results}")
```

## Performance Testing Commands

### Run Comprehensive Comparison
```bash
# Test all improvements
python strategy_runner.py \
    --data data/raw \
    --mode compare \
    --use-optimized \
    --output improved_comparison \
    --configs "xgboost_focal,rf_adaptive,ensemble_voting"
```

### Test Individual Improvements
```bash
# 1. Test lower thresholds
python strategy_runner.py --data data/raw --model random_forest \
    --threshold-buy 0.55 --threshold-sell 0.45 \
    --mode single --output rf_lower_thresholds

# 2. Test adaptive thresholds  
python strategy_runner.py --data data/raw --model random_forest \
    --use-adaptive always --buy-percentile 70 --sell-percentile 30 \
    --mode single --output rf_adaptive_aggressive

# 3. Test focal loss
python strategy_runner.py --data data/raw --model xgboost \
    --focal-loss --focal-gamma 2.0 \
    --mode single --output xgboost_focal
```

## Metrics to Track

### Before (Current Baseline)
- Decision Tree: 42% return, 0.37 Sharpe, 5 trades
- Random Forest: 16% return, 0.01 Sharpe, 6 trades  
- XGBoost: 24% return, 0.06 Sharpe, 63 trades
- Best Sharpe: 0.37 (Decision Tree)

### Target After Quick Wins
- Trades: 20-30 (4-5x increase)
- Sharpe: > 0.50 (35% improvement)
- Win Rate: > 55%
- Annual Return: > 10%

### Ultimate Target (v0.2)
- Annual Return: 20%
- Sharpe Ratio: 0.75
- Max Drawdown: < 20%

## Daily Checklist
1. [ ] Run improved_comparison test
2. [ ] Check number of trades generated
3. [ ] Compare Sharpe ratios
4. [ ] Document any parameter changes
5. [ ] Commit improvements that show positive results

## If Quick Wins Don't Work
1. Implement full Sharpe-based optimization
2. Add market microstructure features
3. Try different base models (LightGBM, CatBoost)
4. Implement proper walk-forward analysis
5. Consider alternative data sources