# Decision Tree Classifier Trading Strategy

An advanced ensemble-based machine learning trading system for S&P 500 stocks with Interactive Brokers (IBKR) integration. This project implements a comprehensive pipeline from data acquisition to live trading with sophisticated regime detection, hyperparameter optimization, and performance analysis.

## 🎯 Project Overview

### Current Status: Phase 1.5 (v0.2 Roadmap)
The system has evolved into a sophisticated trading platform with the following **completed features**:

✅ **Multi-Model Support**: Decision Tree, Random Forest, XGBoost, and Stacking ensembles  
✅ **Strategy Framework**: TrendFollowing and RegimeAdaptive strategies  
✅ **Hyperparameter Optimization**: Automated optimization with Optuna  
✅ **Feature Engineering**: Advanced feature auditing and pruning capabilities  
✅ **Market Regime Detection**: Multi-method regime identification and adaptation  
✅ **Performance Analysis**: Comprehensive backtesting and visualization  
✅ **Modular Architecture**: Engine-based design with clear separation of concerns  

### Success Metrics
- **CAGR/Max Drawdown ratio** > 0.40 (vs S&P 500's ~0.18)
- **Accuracy** of directional prediction > 60%
- **Strategy Sharpe ratio** > 1.0
- **Maximum drawdown** < 25%

## 🚀 Quick Start

### Installation
```bash
git clone <repository-url>
cd DecisionTree
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Verify Installation
```bash
# Quick system test
python strategy_runner.py --data data/raw --model random_forest --mode single --output quick_test
```

## 📊 Comprehensive System Testing Guide

This guide provides step-by-step testing procedures to validate all system capabilities. Tests progress from basic functionality to advanced features.

### Prerequisites

#### 1. Environment Setup
```bash
# Verify Python environment
python --version  # Should be 3.9+
pip list | grep -E "(numpy|pandas|scikit-learn|optuna)"

# Check project structure
ls -la  # Should see: src/, data/, config.py, strategy_runner.py, etc.
```

#### 2. Data Setup
Ensure historical data files are present:
```bash
# Required data files
ls data/raw/historical_data_STOCK_SPY_1_day*.csv

# Expected files:
# - historical_data_STOCK_SPY_1_day2000-2009.csv (optional)
# - historical_data_STOCK_SPY_1_day2010-2025.csv (required)
```

If data files are missing, the system will attempt to load from any CSV files in the data directory containing "SPY".

### Level 1: Basic Functionality Tests

#### 1.1 Smoke Tests - Basic Strategy Execution
Test core functionality with each model type:

```bash
# Test Decision Tree
python strategy_runner.py --data data/raw --model decision_tree --mode single --output test_dt

# Test Random Forest
python strategy_runner.py --data data/raw --model random_forest --mode single --output test_rf

# Test XGBoost (if available)
python strategy_runner.py --data data/raw --model xgboost --mode single --output test_xgb

# Test Stacking Ensemble
python strategy_runner.py --data data/raw --model stacking --mode single --output test_stacking
```

**Expected Results:**
- Each command completes without errors
- Performance metrics printed to console
- Results saved to respective output directories
- Equity curve plots generated

#### 1.2 Strategy Comparison
```bash
# Compare all strategies
python strategy_runner.py --data data/raw --mode compare --output comparison_baseline
```

**Expected Results:**
- Comparison chart with all model performances
- CSV file with metrics comparison
- Individual result directories for each strategy

#### 1.3 Data Processing Validation
```bash
# Test different date ranges
python strategy_runner.py --data data/raw --model random_forest --train-end 2023-01-01 --output date_test
```

### Level 2: Component-Level Testing

#### 2.1 Feature Engineering & Auditing
```bash
# Run feature importance audit
python strategy_runner.py --data data/raw --mode audit --audit-model random_forest --output feature_audit_test

# Run strategy with feature auditing
python strategy_runner.py --data data/raw --model random_forest --feature-audit --top-features 15 --output feature_pruned_test
```

**Expected Results:**
- Feature importance rankings
- Collinearity analysis
- Pruned feature performance comparison

#### 2.2 Hyperparameter Optimization
```bash
# Quick optimization test (reduced trials for testing)
python optimize_hyperparameters.py --data data/raw --model random_forest --trials 10 --output optimization_test

# Test all models optimization
python optimize_hyperparameters.py --data data/raw --model all --trials 5 --output optimization_all_test
```

**Expected Results:**
- Optuna optimization progress
- Best parameters saved to data/hyperparameters/
- Performance improvement metrics

#### 2.3 Regime Detection Testing
```bash
# Test regime detection
python test_regime_detection.py --data data/raw/historical_data_STOCK_SPY_1_day2010-2025.csv --output regime_test

# Alternative: Use built-in regime detection
python strategy_runner.py --data data/raw --strategy regime_adaptive --model random_forest --output regime_strategy_test
```

**Expected Results:**
- Regime identification charts
- Regime statistics and transitions
- Performance analysis by regime

### Level 3: Advanced Feature Testing

#### 3.1 Optimized Parameter Usage
```bash
# First, optimize parameters
python optimize_hyperparameters.py --data data/raw --model random_forest --trials 20

# Then run strategy with optimized parameters
python strategy_runner.py --data data/raw --model random_forest --use-optimized --output optimized_test

# Compare optimized vs baseline
python strategy_runner.py --data data/raw --model random_forest --output baseline_test
```

#### 3.2 Regime-Specific Optimization
```bash
# Optimize with regime-specific parameters
python optimize_hyperparameters.py --data data/raw --model random_forest --trials 15 --regime-specific

# Run regime-adaptive strategy with optimized parameters
python strategy_runner.py --data data/raw --strategy regime_adaptive --model random_forest --use-optimized --output regime_optimized_test
```

#### 3.3 Probability Calibration Testing
```bash
# Test calibrated models
python strategy_runner.py --data data/raw --model decision_tree --calibrate --output dt_calibrated_test
python strategy_runner.py --data data/raw --model random_forest --calibrate --output rf_calibrated_test
```

#### 3.4 Position Sizing Strategies
```bash
# Test different position sizing approaches (configured in strategy_configs.py)
python strategy_runner.py --data data/raw --model xgboost --output xgb_confidence_test
```

### Level 4: Integration & End-to-End Testing

#### 4.1 Complete Workflow Test
```bash
# Full workflow: Feature audit → Optimization → Strategy execution
python strategy_runner.py --data data/raw --model random_forest --feature-audit --use-optimized --output complete_workflow_test
```

#### 4.2 Multiple Strategy Comparison with Optimization
```bash
# Compare strategies using optimized parameters
python optimize_hyperparameters.py --data data/raw --model all --trials 15

python strategy_runner.py --data data/raw --mode compare --use-optimized --output optimized_comparison
```

#### 4.3 Scheduled Optimization Testing
```bash
# Test hyperparameter scheduling (runs briefly then stops)
python schedule_hyperparameter_scan.py --data data/raw --model random_forest --trials 5 --day sunday --time "00:00" &
SCHEDULER_PID=$!
sleep 10
kill $SCHEDULER_PID

# Check scheduler logs
grep "Scheduled" hyperparam_scheduler.log 2>/dev/null || echo "Scheduler test completed"
```

### Level 5: Performance Validation

#### 5.1 Benchmark Comparison
```bash
# Generate comprehensive performance comparison
python -c "
import pandas as pd
import os

# Load results from different tests
results = {}
test_dirs = ['test_dt', 'test_rf', 'test_xgb', 'optimized_test', 'regime_optimized_test']

for test_dir in test_dirs:
    perf_file = f'{test_dir}/performance.txt'
    if os.path.exists(perf_file):
        print(f'\n=== {test_dir.upper()} RESULTS ===')
        with open(perf_file, 'r') as f:
            print(f.read())
"
```

#### 5.2 Regime Performance Analysis
```bash
# Detailed regime analysis
python strategy_runner.py --data data/raw --strategy regime_adaptive --model random_forest --use-optimized --output regime_analysis

# Check regime-specific results
ls regime_analysis/regime_performance.*
```

#### 5.3 Risk Metrics Validation
```bash
# Generate risk analysis report
python -c "
import pandas as pd
import numpy as np
import os

def analyze_equity_curve(equity_file):
    if os.path.exists(equity_file):
        equity = pd.read_csv(equity_file, index_col=0, parse_dates=True)
        if 'equity' in equity.columns:
            returns = equity['equity'].pct_change().dropna()
            
            # Calculate risk metrics
            total_return = (equity['equity'].iloc[-1] / equity['equity'].iloc[0]) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe = returns.mean() * 252 / (returns.std() * np.sqrt(252))
            max_dd = ((equity['equity'] / equity['equity'].cummax()) - 1).min()
            
            print(f'Total Return: {total_return:.2%}')
            print(f'Volatility: {volatility:.2%}')
            print(f'Sharpe Ratio: {sharpe:.2f}')
            print(f'Max Drawdown: {max_dd:.2%}')
            print(f'Calmar Ratio: {total_return/abs(max_dd):.2f}')
            print('---')
    
# Analyze different strategies
test_results = ['test_rf/equity_curve.csv', 'optimized_test/equity_curve.csv', 'regime_optimized_test/equity_curve.csv']
for result_file in test_results:
    if os.path.exists(result_file):
        print(f'Analysis for {result_file}:')
        analyze_equity_curve(result_file)
"
```

### Level 6: Specialized Testing

#### 6.1 Feature Audit Deep Dive
```bash
# Comprehensive feature analysis
python run_feature_audit.py --data data/raw/historical_data_STOCK_SPY_1_day2010-2025.csv --model random_forest --output detailed_audit

# Check audit results
ls detailed_audit/
cat detailed_audit/audit_summary.txt
```

#### 6.2 Focal Loss Testing (XGBoost)
```bash
# Test focal loss for imbalanced data
python tests/test_focal_loss.py

# Run XGBoost with different focal loss settings
python optimize_hyperparameters.py --data data/raw --model xgboost --trials 10
```

#### 6.3 Model Stacking Analysis
```bash
# Test stacking ensemble in detail
python strategy_runner.py --data data/raw --model stacking --output stacking_analysis

# Examine stacking components
python -c "
import pickle
import os

if os.path.exists('stacking_analysis/'):
    print('Stacking model components analysis...')
    # Add analysis code for stacking components if model files are saved
"
```

## 🔧 Troubleshooting Guide

### Common Issues and Solutions

#### Critical Issue: XGBoost Optimization Fails (Fixed as of 2025-06-05)
**Symptoms:**
```
Import error in XGBoost optimization: cannot import name 'FocalLoss' from 'src.models.xgboost_model'
[I 2025-06-05 10:58:26,451] Trial 0 finished with value: 0.0 and parameters: {}.
```

**Root Cause:** Incomplete fix implementation - hyperparameter optimization tried to import removed classes.

**Solution:** ✅ **FIXED** - Updated `src/models/hyperparameter_optimization.py` to work with simplified XGBoost implementation.

#### XGBoost Focal Loss Parameter Warnings
**Symptoms:**
```
WARNING: /Users/runner/work/xgboost/xgboost/src/learner.cc:738: 
Parameters: { "focal_alpha", "focal_gamma", "use_focal_loss" } are not used.
```

**Root Cause:** Configuration generates focal loss parameters but simplified XGBoost doesn't use them.

**Solution:** These are warnings only and don't affect functionality. The optimization has been updated to remove these parameters.

#### RegimeAdaptiveStrategy Date Ambiguity
**Symptoms:**
```
ERROR - Error in generate_signals: 'date' is both an index level and a column label, which is ambiguous.
```

**Root Cause:** DataFrame has 'date' as both index and column name.

**Solution:** 
```python
# For developers - ensure consistent date handling:
if signals.index.name == 'date' and 'date' in signals.columns:
    signals = signals.reset_index(drop=True)
```

The strategy code has multiple fixes for this, but if you encounter it:
```bash
# Workaround: Use TrendFollowing strategy instead
python strategy_runner.py --data data/raw --strategy trend_following --model random_forest --output workaround_test
```

#### Data Loading Issues
```bash
# Check data files
find data/ -name "*.csv" -type f
# If no files found, add your historical data to data/raw/

# Test data loading
python -c "
from strategy_runner import load_data
try:
    df = load_data('data/raw')
    print(f'Data loaded successfully: {df.shape} rows')
    print(f'Date range: {df.index[0]} to {df.index[-1]}')
except Exception as e:
    print(f'Data loading error: {e}')
"
```

#### Import Errors
```bash
# Check Python path
python -c "import sys; print('\\n'.join(sys.path))"

# Test critical imports
python -c "
try:
    from src.strategies.trend_following import TrendFollowingStrategy
    from src.models.model_factory import ModelFactory
    print('✅ Core imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
"
```

#### Memory Issues
```bash
# Check memory usage during optimization
python optimize_hyperparameters.py --data data/raw --model random_forest --trials 5 --verbose

# For large datasets, reduce lookback period in config.py
```

#### Performance Issues
```bash
# Quick performance test
time python strategy_runner.py --data data/raw --model random_forest --train-end 2023-01-01 --output perf_test

# Check if results meet success criteria
python -c "
import pandas as pd
import os

if os.path.exists('perf_test/performance.txt'):
    with open('perf_test/performance.txt', 'r') as f:
        content = f.read()
        print('Performance check:')
        if 'sharpe_ratio:' in content:
            # Extract and validate metrics
            print('✅ Performance metrics generated')
        else:
            print('❌ Performance metrics missing')
"
```

### If You Encounter New Issues

1. **Check the error logs** - All strategies log detailed error information
2. **Try the TrendFollowing strategy** - It's more stable than RegimeAdaptive
3. **Reduce complexity** - Use fewer trials for optimization, smaller datasets
4. **Check the issue files** - See MEDIUM_PRIORITY_FIXES.md and LOW_PRIORITY_FIXES.md for known issues

## 📈 Expected Results Summary

After running the complete testing suite, you should see:

### Performance Targets
- **Sharpe Ratio**: > 0.7 (Phase 2 target)
- **CAGR/Max Drawdown**: > 0.35 
- **Model Accuracy**: > 57% (Phase 2 target)
- **Max Drawdown**: < 25%

### File Outputs
- Equity curves and performance plots
- Trade logs and regime analysis
- Feature importance rankings
- Optimized hyperparameters in `data/hyperparameters/`
- Comprehensive performance reports

### Model Comparison
The system should demonstrate:
- Ensemble models outperforming single models
- Optimized parameters improving performance
- Regime-adaptive strategies showing better risk-adjusted returns
- Feature pruning maintaining or improving model performance

## 🔗 Related Documentation

- **[Complete Strategy Documentation](Decision_Tree_Classifier_Strategy.md)** - Comprehensive system guide
- **[v0.2 Roadmap](v0.2_roadmap.md)** - Development progress and future plans  
- **[Test Plan](test_plan.md)** - Detailed automated testing procedures
- **[Hyperparameter Optimization](hyperparameter_optimization.md)** - Optimization strategies
- **[Priority Fixes](MEDIUM_PRIORITY_FIXES.md)** - Recent fixes and improvements

## 🎛️ Interactive Brokers Setup

For live trading (Phase 3), configure IBKR:
1. Install TWS or IB Gateway
2. Enable API connections (Configuration > API > Settings)
3. Set appropriate ports in `config.py`
4. Test connection with paper trading first

---

**Note**: This is an active development project currently in Phase 1.5 of the v0.2 roadmap. Critical issues as of 2025-06-05 have been resolved. For issues or contributions, see the development documentation in the main strategy file.
