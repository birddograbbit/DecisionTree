# Feature Analysis Comparison: Daily vs Intraday Data

## Implementation Summary
Successfully implemented timeframe-aware feature analysis with appropriate lookback periods:
- **Daily data**: 10-day lookback period
- **5-minute data**: 78-bar lookback period (1 trading day)
- **1-minute data**: 390-bar lookback period (1 trading day)

## Test Results

### 5-Minute Data Feature Importance
Using lookback period: 78 bars

Top 10 features:
1. **plus_di**: 0.0009 (directional movement indicator)
2. **atr**: 0.0007 (average true range)  
3. **volume_momentum_1d**: 0.0004 (volume momentum)
4. **macd**: -0.0002
5. **stoch_k**: -0.0002
6. **minus_di**: -0.0002
7. **adx**: -0.0003
8. **price_momentum_5d**: -0.0003
9. **adx_momentum**: -0.0004
10. **atr_zscore**: -0.0005

Collinearity detected:
- sma_ratio and macd: 0.916
- bb_position and stoch_k: 0.922

### Daily Data Feature Importance
Using lookback period: 10 days

Top 10 features:
1. **sma_ratio**: 0.0072 (price/SMA ratio)
2. **atr_zscore**: 0.0061 (normalized volatility)
3. **rsi**: 0.0054 (relative strength index)
4. **adx_momentum**: 0.0044
5. **stoch_k**: 0.0043
6. **macd**: 0.0036
7. **std**: 0.0025
8. **adx**: 0.0025
9. **price_momentum_5d**: 0.0023
10. **atr**: 0.0021

Collinearity detected:
- rsi and bb_position: 0.849
- sma_ratio and price_momentum_5d: 0.909
- bb_position and stoch_k: 0.830

## Key Insights

1. **Different Feature Importance Patterns**: 
   - 5-minute data prioritizes directional indicators (plus_di) and short-term volatility
   - Daily data emphasizes trend ratios (sma_ratio) and normalized volatility measures

2. **Magnitude Differences**:
   - 5-minute features show lower importance values (max 0.0009)
   - Daily features show higher importance values (max 0.0072)
   - This suggests daily patterns may be more predictable

3. **Collinearity Patterns**:
   - Both timeframes show high correlation between momentum indicators
   - Different pairs are correlated at different timeframes

## Usage Commands

```bash
# For 5-minute data analysis
python strategy_runner.py --data data/raw --mode audit --audit-model random_forest --output 5min_features --timeframe 5min

# For daily data analysis  
python strategy_runner.py --data data/raw --mode audit --audit-model random_forest --output daily_features
```

## Recommendation
When adding new strategies, run feature analysis on the appropriate timeframe to understand which technical indicators are most predictive. This helps in:
1. Feature selection for ML models
2. Understanding market dynamics at different frequencies
3. Optimizing strategy parameters for each timeframe