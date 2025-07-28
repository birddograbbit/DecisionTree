# Next Steps for DecisionTree Hybrid System Optimization

## Executive Summary
After implementing the Strategy Adapter Pattern and 5-minute data support, momentum strategies show significant performance improvements. The system now has flexible architecture for testing sophisticated strategies across multiple timeframes. Latest results show Quod strategy achieving 1.53 Sharpe ratio on 5-minute data, exceeding our v0.2 target of 0.75.

## Current Performance Status (July 28, 2025)

### Daily Data Results
- **Decision Tree**: -38.44% return, -19.36 Sharpe, 699 trades
- **Random Forest**: -35.62% return, -17.42 Sharpe, 599 trades  
- **XGBoost (Fixed)**: -1.68% return, -0.39 Sharpe, 1,459 trades
- **XGBoost (Confidence)**: -31.04% return, -13.24 Sharpe, 1,195 trades
- **Regime Adaptive RF**: 0.20% return, -19.08 Sharpe, 2 trades

### 5-Minute Data Results ✨
- **BB-RSI-ADX**: 8.25% return, 1.40 Sharpe, 935 trades ✅
- **TEMA**: 5.81% return, 0.76 Sharpe, 1,904 trades ✅
- **Quod**: 14.98% return, 1.53 Sharpe, 3,195 trades ✅✅

### Key Findings
1. **Momentum strategies excel on 5-minute data** - All three momentum strategies achieve positive returns and Sharpe ratios
2. **Quod strategy exceeds v0.2 target** - 1.53 Sharpe ratio surpasses our 0.75 goal
3. **Trading frequency dramatically increased** - From 5-6 trades to thousands with intraday data
4. **ML strategies need optimization** - Still showing negative performance on both timeframes

## Full Operational Procedures

### Testing Procedures

#### 1. Daily Data Testing
```bash
# Single strategy test
python strategy_runner.py --data data/raw --model random_forest --mode single --output daily_test

# Compare all ML strategies
python strategy_runner.py --data data/raw --mode compare --output daily_comparison

# Compare with momentum strategies
python strategy_runner.py --data data/raw --mode compare --include-momentum --output full_comparison
```

#### 2. 5-Minute Data Testing
```bash
# Single momentum strategy with 5-minute data
python strategy_runner.py --data data/raw --model bb_rsi_adx --mode single --output 5min_test --timeframe 5min

# Compare all strategies on 5-minute data
python strategy_runner.py --data data/raw --mode compare --include-momentum --output 5min_comparison --timeframe 5min

# Test specific momentum strategy
python strategy_runner.py --data data/raw --model quod --mode single --output quod_5min --timeframe 5min
```

#### 3. Performance Validation Checklist
- [ ] Verify CAGR is not 0.00% (metrics calculation working)
- [ ] Check Sharpe ratio is properly annualized for timeframe
- [ ] Confirm trade count matches expected frequency
- [ ] Validate win rate is realistic (40-60% range)
- [ ] Ensure max drawdown is within acceptable limits (<20%)

### Data Requirements

#### Daily Data Files
- Training: `data/raw/historical_data_STOCK_SPY_1_day2000-2009.csv`
- Testing: `data/raw/historical_data_STOCK_SPY_1_day2010-2025.csv`

#### 5-Minute Data Files
- Training: `data/raw/historical_data_STOCK_SPY_5_mins_2023-2024.csv`
- Testing: `data/raw/historical_data_STOCK_SPY_5_mins_2025.csv`

### Strategy Configuration

#### Momentum Strategies Available
1. **BB-RSI-ADX** (`--model bb_rsi_adx`)
   - Bollinger Bands with RSI extremes and ADX filtering
   - Best for: Momentum continuation in strong trends
   
2. **TEMA** (`--model tema`)
   - Triple Exponential Moving Average crossovers
   - Best for: Smooth trend following with reduced lag
   
3. **Quod** (`--model quod`)
   - Stochastic reversal and pullback patterns
   - Best for: Mean reversion within trends

## Completed Implementations (July 2025)

### 1. Strategy Adapter Pattern ✅
- Base infrastructure with registry pattern
- Three momentum strategy adapters
- Multi-timeframe feature support
- Advanced order management system

### 2. 5-Minute Data Support ✅
- Data loading and preprocessing for intraday bars
- Timeframe-aware performance metrics
- Proper CAGR and Sharpe calculations for 5-minute data
- Holding period tracking in bars

### 3. Performance Metrics Fixes ✅
- Fixed CAGR showing 0.00% for 5-minute data
- Corrected Sharpe ratio annualization (√19,656 for 5-min)
- Updated holding period calculations
- Simplified transaction cost modeling

## Recommended Next Steps (Priority Order)

### 1. ML Strategy Optimization for 5-Minute Data (HIGH PRIORITY)
**Problem**: ML strategies show poor performance on both daily and 5-minute data.

**Solution**: Retrain and optimize for intraday patterns
```python
# Add intraday-specific features
- Microstructure indicators (bid-ask proxy, volume patterns)
- Time-of-day effects (morning vs afternoon patterns)
- Intraday momentum indicators
- Rolling volatility measures

# Adjust ML parameters for high-frequency data
- Shorter lookback periods (10-50 bars instead of 10-50 days)
- Faster decay in feature importance
- More emphasis on recent price action
```

### 2. Hybrid ML-Momentum Strategies (HIGH PRIORITY)
**Problem**: ML and momentum strategies have complementary strengths.

**Solution**: Combine ML predictions with momentum signals
```python
class HybridMomentumML(BaseStrategy):
    def __init__(self, ml_model, momentum_strategy):
        self.ml_model = ml_model
        self.momentum_strategy = momentum_strategy
    
    def generate_signals(self, features, prices):
        # Get ML predictions
        ml_signals = self.ml_model.predict(features)
        
        # Get momentum signals
        momentum_signals = self.momentum_strategy.generate_signals(features)
        
        # Combine: Enter only when both agree
        combined_signals = ml_signals * momentum_signals
        
        # Or weighted combination
        weight_ml = 0.3
        weight_momentum = 0.7
        combined_probs = (weight_ml * ml_probs + 
                         weight_momentum * momentum_probs)
        
        return combined_signals
```

### 3. Trading-Focused Hyperparameter Optimization (HIGH PRIORITY)
**Problem**: Current optimization uses accuracy, not trading metrics.

**Solution**: Optimize directly for Sharpe ratio
```python
def objective(trial):
    # Sample hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    }
    
    # Train model and generate signals
    model = XGBClassifier(**params)
    # ... training code ...
    
    # Backtest on validation data
    backtest_results = run_backtest(signals, val_data, timeframe='5min')
    
    # Return Sharpe ratio (to maximize)
    return backtest_results['performance']['sharpe_ratio']
```

### 4. Ensemble of Timeframes (MEDIUM PRIORITY)
**Problem**: Different timeframes capture different market dynamics.

**Solution**: Combine signals from multiple timeframes
```python
# Run strategies on multiple timeframes
timeframes = ['5min', '15min', '1h', 'daily']
all_signals = []

for tf in timeframes:
    signals = strategy.generate_signals(data_resampled[tf])
    all_signals.append(signals)

# Voting or weighted average
final_signal = np.mean(all_signals, axis=0)
```

### 5. Regime-Aware Strategy Selection (MEDIUM PRIORITY)
**Problem**: Different strategies work better in different market conditions.

**Solution**: Dynamically select strategy based on market regime
```python
class RegimeAwareSelector:
    def __init__(self, strategies):
        self.strategies = strategies
        self.regime_detector = MarketRegimeDetector()
    
    def select_strategy(self, market_data):
        regime = self.regime_detector.detect(market_data)
        
        if regime == 'trending':
            return self.strategies['tema']  # Trend following
        elif regime == 'ranging':
            return self.strategies['quod']  # Mean reversion
        elif regime == 'volatile':
            return self.strategies['bb_rsi_adx']  # Momentum
        else:
            return self.strategies['ensemble']  # Default
```

## Implementation Timeline

### Week 1: ML Strategy Optimization
1. Add intraday-specific features
2. Retrain ML models on 5-minute data
3. Implement Sharpe-focused optimization
4. Test performance improvements

### Week 2: Hybrid Strategies
1. Create ML-momentum hybrid framework
2. Test different combination methods
3. Optimize weighting schemes
4. Validate on out-of-sample data

### Week 3: Advanced Techniques
1. Implement multi-timeframe ensemble
2. Build regime detection system
3. Create dynamic strategy selector
4. Comprehensive backtesting

## Success Metrics
- **Primary**: Achieve consistent Sharpe > 1.0 across strategies
- **Secondary**: Annual return > 15% on 5-minute data
- **Tertiary**: 
  - Win rate: 48-52% (realistic for high-frequency)
  - Max drawdown: < 10%
  - Trade frequency: 500-5000 trades/year

## Testing Commands Reference

### Quick Performance Check
```bash
# Test best performer (Quod on 5-min)
python strategy_runner.py --data data/raw --model quod --mode single --output quick_test --timeframe 5min
```

### Full Comparison
```bash
# Compare all strategies on 5-minute data
python strategy_runner.py --data data/raw --mode compare --include-momentum --output full_5min_comparison --timeframe 5min
```

### Feature Importance Analysis
```bash
# Analyze which features matter most
# For daily data
python strategy_runner.py --data data/raw --mode audit --audit-model random_forest --output feature_analysis

# For 5-minute data
python strategy_runner.py --data data/raw --mode audit --audit-model random_forest --output 5min_features --timeframe 5min   
```

## Key Insights from Latest Results

1. **Timeframe Matters**: Momentum strategies designed for intraday trading perform poorly on daily data but excel on 5-minute data
2. **Strategy-Data Fit**: Matching strategy design to data frequency is crucial
3. **ML Needs Adaptation**: Current ML models trained on daily patterns need retraining for intraday dynamics
4. **Achieved Goal**: Quod strategy's 1.53 Sharpe exceeds our 0.75 target

## Project Status Summary (Updated July 28, 2025)

### Completed Work
1. **Strategy Adapter Pattern** ✅
   - Full implementation with 3 momentum strategies
   - Registry pattern for clean management
   - Multi-timeframe support
   
2. **5-Minute Data Integration** ✅
   - Data loading and preprocessing
   - Timeframe-aware metrics
   - Proper performance calculations

3. **Performance Metrics Fixes** ✅
   - CAGR calculation for all timeframes
   - Sharpe ratio annualization
   - Holding period tracking

### Current Capabilities
- **8 strategies**: decision_tree, random_forest, xgboost, regime_adaptive, bb_rsi_adx, tema, quod, stacking
- **2 timeframes**: daily and 5-minute
- **Advanced features**: Multi-timeframe analysis, order management, registry pattern
- **Performance**: Quod achieves 1.53 Sharpe on 5-minute data

### Next Phase Focus
1. Optimize ML strategies for 5-minute data
2. Create hybrid ML-momentum approaches  
3. Implement ensemble techniques
4. Build production-ready system

## Conclusion
The system has successfully demonstrated that sophisticated momentum strategies can achieve the v0.2 performance targets when applied to appropriate timeframe data. The next phase focuses on bringing ML strategies up to similar performance levels and creating robust ensemble approaches for consistent results.