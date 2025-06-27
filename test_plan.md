# Decision Tree Strategy System Test Plan

This document outlines a comprehensive testing plan for the Decision Tree Strategy system, covering all features implemented through Phase 1.5 of the v0.2 roadmap. These tests are designed to verify individual component functionality and integration between components.

## 1. Baseline Functionality Tests

### 1.1 Basic Strategy Execution

```bash
# Test basic strategy with Decision Tree model
python strategy_runner.py --data data/raw --model decision_tree --mode single --output baseline_results

# Test basic strategy with Random Forest model
python strategy_runner.py --data data/raw --model random_forest --mode single --output baseline_results

# Test basic strategy with XGBoost model (if available)
python strategy_runner.py --data data/raw --model xgboost --mode single --output baseline_results
```

### 1.2 Strategy Comparison

```bash
# Compare performance of different model types
python strategy_runner.py --data data/raw --mode compare --output comparison_results
```

### 1.3 Data Processing

```bash
# Test data loading and preprocessing with different date ranges
python strategy_runner.py --data data/raw --model random_forest --train-end 2023-01-01 --output data_test_results
```

## 2. Phase 1 Feature Tests

### 2.1 Hyperparameter Optimization with Optuna

```bash
# Test hyperparameter optimization for Decision Tree (small trial count for quick testing)
python optimize_hyperparameters.py --data data/raw --model decision_tree --trials 10

# Test hyperparameter optimization for Random Forest
python optimize_hyperparameters.py --data data/raw --model random_forest --trials 10

# Test hyperparameter optimization for XGBoost (if available)
python optimize_hyperparameters.py --data data/raw --model xgboost --trials 10

# Test optimization of all models (reduced trials for testing)
python optimize_hyperparameters.py --data data/raw --model all --trials 5
```

### 2.2 Feature Importance and Pruning

```bash
# Test feature importance calculation (create a specific test script)
python -c "
from src.data.preprocessing import preprocess_data
from src.features.feature_engineering import engineer_features
from src.models.model_factory import ModelFactory
import pandas as pd

# Load data
data = pd.read_csv('data/raw/historical_data_STOCK_SPY_1_day2010-2025.csv', index_col=0, parse_dates=True)
data = preprocess_data(data)

# Engineer features
X, y, features = engineer_features(data)

# Create and train model
model = ModelFactory.create_model('random_forest')
model.train(X, y)

# Get feature importance
importance = model.get_feature_importance()
print(importance.sort_values(ascending=False).head(10))
"
```

### 2.3 Momentum-Volatility Hybrid Features

```bash
# Test regime detection with volatility features
python test_regime_detection.py --data data/raw/historical_data_STOCK_SPY_1_day2010-2025.csv --output regime_detection_results --method volatility_regimes
```

### 2.4 Class Imbalance Handling

```bash
# Test XGBoost with focal loss
python optimize_hyperparameters.py --data data/raw --model xgboost --trials 5
```

## 3. Phase 1.5 Feature Tests

### 3.1 Auto-Parameter Loading System

```bash
# Run strategy with optimized parameters
python strategy_runner.py --data data/raw --model random_forest --use-optimized --output optimized_results

# Compare baseline vs optimized parameters
python strategy_runner.py --data data/raw --model random_forest --mode single --output baseline_results
python strategy_runner.py --data data/raw --model random_forest --use-optimized --output optimized_results
python -c "
import pandas as pd
baseline = pd.read_pickle('baseline_results/performance_metrics.pkl')
optimized = pd.read_pickle('optimized_results/performance_metrics.pkl')
print('Baseline metrics:', baseline)
print('Optimized metrics:', optimized)
"
```

### 3.2 Model Parameter Persistence

```bash
# Verify parameter persistence across runs
python optimize_hyperparameters.py --data data/raw --model decision_tree --trials 5
python -c "
import os
import pickle
from pathlib import Path
import config

# List versioned hyperparameters
hyperparams_dir = Path(config.HYPERPARAMS_DIR)
versioned_dir = hyperparams_dir / 'versioned'
files = sorted(os.listdir(versioned_dir))
print(f'Found {len(files)} versioned hyperparameter files')
for f in files:
    if f.endswith('.pkl'):
        print(f)
        
# Load latest parameters for comparison
latest_file = [f for f in files if f.endswith('.pkl')][-1]
with open(versioned_dir / latest_file, 'rb') as f:
    params = pickle.load(f)
print('Latest hyperparameters:', params)
"
```

### 3.3 Optimization Preprocessing CLI

```bash
# Test CLI with different parameter combinations
python optimize_hyperparameters.py --data data/raw --model random_forest --trials 5 --lookback 30
python optimize_hyperparameters.py --data data/raw --model random_forest --trials 5 --lookback 50
```

### 3.4 Regime-Specific Hyperparameter Optimization

```bash
# Test regime-specific optimization
python optimize_hyperparameters.py --data data/raw --model random_forest --trials 10 --regime-specific

# Verify regime-specific parameters
python -c "
import os
import pickle
from pathlib import Path
import config

# List regime-specific hyperparameters
hyperparams_dir = Path(config.HYPERPARAMS_DIR)
regime_dir = hyperparams_dir / 'regimes'
files = sorted(os.listdir(regime_dir))
print(f'Found {len(files)} regime-specific parameter files')
for f in files:
    if f.endswith('.pkl'):
        print(f)
        
# Load parameters for one regime
if files:
    with open(regime_dir / files[0], 'rb') as f:
        params = pickle.load(f)
    print('Regime-specific parameters sample:', params)
"
```

### 3.5 Regime-Adaptive Strategy with Optimized Parameters

```bash
# Test regime-adaptive strategy with default parameters
python test_regime_adaptive_strategy.py --data data/raw/historical_data_STOCK_SPY_1_day2010-2025.csv --output regime_default_results

# Test regime-adaptive strategy with optimized parameters
python test_regime_adaptive_strategy.py --data data/raw/historical_data_STOCK_SPY_1_day2010-2025.csv --output regime_optimized_results --use-optimized
```

### 3.6 Adaptive Probability Thresholds

```bash
# Test regular thresholds
python strategy_runner.py --data data/raw --model random_forest --strategy trend_following --output standard_threshold_results

# Test adaptive thresholds with regime-adaptive strategy
python strategy_runner.py --data data/raw --model random_forest --strategy regime_adaptive --output adaptive_threshold_results
```

### 3.7 Hyperparameter Scanning Scheduler

```bash
# Test scheduler initialization (run for 10 seconds then terminate)
python schedule_hyperparameter_scan.py --data data/raw --model random_forest --trials 5 --day sunday --time "00:00" &
PID=$!
sleep 10
kill $PID

# Verify scheduler logs
grep "Scheduled hyperparameter scan" hyperparam_scheduler.log
```

## 4. Integration Tests

### 4.1 End-to-End Workflow Test

```bash
# Optimize parameters
python optimize_hyperparameters.py --data data/raw --model random_forest --trials 10

# Run strategy with optimized parameters
python strategy_runner.py --data data/raw --model random_forest --use-optimized --output full_workflow_results

# Analyze results
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# Load results
results = pd.read_pickle('full_workflow_results/backtest_results.pkl')

# Plot equity curve
plt.figure(figsize=(12, 6))
plt.plot(results['equity_curve'])
plt.title('Equity Curve')
plt.xlabel('Date')
plt.ylabel('Equity')
plt.savefig('full_workflow_results/equity_curve.png')

# Print key metrics
metrics = results['performance']
print('Sharpe Ratio:', metrics.get('sharpe_ratio'))
print('CAGR:', metrics.get('cagr'))
print('Max Drawdown:', metrics.get('max_drawdown'))
print('CAGR/MaxDD:', metrics.get('cagr') / metrics.get('max_drawdown') if metrics.get('max_drawdown') else 'N/A')
"
```

### 4.2 Regime-Specific Models with Optimization

```bash
# Optimize regime-specific parameters
python optimize_hyperparameters.py --data data/raw --model random_forest --trials 10 --regime-specific

# Run regime-adaptive strategy with optimized parameters
python strategy_runner.py --data data/raw --model random_forest --strategy regime_adaptive --use-optimized --output regime_integration_results
```

## 5. Regression Tests

### 5.1 Fixed Issues Verification

```bash
# Test RegimeAdaptiveStrategy fixes (PR #78)
python test_regime_adaptive_strategy.py --data data/raw/historical_data_STOCK_SPY_1_day2010-2025.csv --output regression_regime_results

# Test pipeline fixes (PR #74, #75)
python optimize_hyperparameters.py --data data/raw --model xgboost --trials 5
```

### 5.2 Performance Comparison Against Baseline

```bash
# Compare performance metrics before and after Phase 1.5
python -c "
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

# Load results from different phases
baseline_file = 'baseline_results/performance_metrics.pkl'
phase1_file = 'optimized_results/performance_metrics.pkl'
phase15_file = 'regime_integration_results/performance_metrics.pkl'

if all(os.path.exists(f) for f in [baseline_file, phase1_file, phase15_file]):
    baseline = pd.read_pickle(baseline_file)
    phase1 = pd.read_pickle(phase1_file)
    phase15 = pd.read_pickle(phase15_file)
    
    # Create comparison table
    comparison = pd.DataFrame({
        'Baseline': [baseline.get('sharpe_ratio'), baseline.get('max_drawdown'), baseline.get('cagr')],
        'Phase 1': [phase1.get('sharpe_ratio'), phase1.get('max_drawdown'), phase1.get('cagr')],
        'Phase 1.5': [phase15.get('sharpe_ratio'), phase15.get('max_drawdown'), phase15.get('cagr')]
    }, index=['Sharpe Ratio', 'Max Drawdown', 'CAGR'])
    
    print('Performance Comparison:')
    print(comparison)
else:
    print('One or more result files not found. Run the corresponding tests first.')
"
```

## 6. Success Criteria

For each test, the following criteria indicate successful implementation:

1. **Basic Functionality**: All commands execute without errors
2. **Phase 1**:
   - Optuna optimization improves model performance over default parameters
   - Feature importance correctly identifies relevant features 
   - Momentum-volatility features are present in feature lists
   - Class imbalance handling improves classification metrics for minority classes

3. **Phase 1.5**:
   - Parameter loading and persistence save/load parameters correctly
   - Regime-specific optimization creates separate parameter files for each regime
   - Regime-adaptive strategy adapts behavior based on detected regime
   - Adaptive thresholds generate reasonable trade counts across regimes
   - Hyperparameter scanning scheduler initializes correctly

4. **Integration**:
   - End-to-end workflow produces expected results
   - Performance metrics improve from baseline to Phase 1.5

These tests should provide comprehensive validation of all implemented features and ensure the system is ready to proceed to Phase 2 of the roadmap.

## Test Results (2025-05-20)

All automated tests and manual commands were executed in the Codex environment. Key results:

- `pytest` passed (tests/test_focal_loss.py)
- Baseline strategy runs completed for Decision Tree, Random Forest and XGBoost models.
- Strategy comparison and data range tests executed successfully.
- Hyperparameter optimization commands ran for individual models and `all` option.
- Feature importance script executed without errors after fixing model pipelines.
- Regime-specific optimization succeeded after fixing index selection logic.
- Regime-adaptive strategy ran with optimized parameters, producing summary metrics.

No failing tests were encountered after applying the above fixes.

## Pre-Production Test Results (2025-06-28)

### Phase 1: Environment Setup ✅

All dependencies installed successfully:
- **PyTorch**: 2.7.1
- **Scikit-learn**: 1.7.0
- **pytest**: 8.4.1
- **All imports verified**: Transformer, Hybrid, and Model Factory components

### Phase 2: Smoke Tests ✅

#### Basic Imports
- ✅ Transformer imports work
- ✅ Hybrid imports work  
- ✅ Model factory imports work

#### Decision Tree Baseline Test
```
Performance Summary:
Total Return: 41.96%
CAGR: 4.97%
Max Drawdown: 16.67%
Sharpe Ratio: 0.37
CAGR/Max DD: 0.30
Win Rate: 80.00%
Number of Trades: 5
```

#### Initial Transformer Tests
- ✅ test_forward_pass PASSED
- ✅ Sequence preparation: 4/4 tests PASSED
- ⚠️ Coverage: 8.09% (vs 25% required)

### Phase 3: Unit Testing (Partial) ⚠️

#### Transformer Components
- **test_transformer_model.py**: 6/6 PASSED
  - Model initialization ✅
  - Forward pass ✅
  - Gradient flow ✅
  - Save/load ✅
  - Positional encoding ✅
  - Multiple configs ✅

- **test_transformer_wrapper.py**: 2/2 PASSED
  - Train/predict ✅
  - Missing data handling ✅

- **test_sequence_preparation.py**: 4/4 PASSED
- **test_technical_indicators.py**: 1/1 PASSED

#### Auxiliary Components
- **GPU Optimizer**: 1 PASSED, 1 SKIPPED (quantization not supported on macOS)
- **Online Learning**: 1/1 PASSED
- **Multi-task**: 1/1 PASSED
- **Versioning**: 1/1 PASSED
- **Error Recovery**: 1/1 PASSED

#### Combined Transformer Tests
- 16/16 tests PASSED
- Coverage improved to 13.63% (still below 25% threshold)

#### Critical Issue: Hybrid Strategy Test
- **Status**: ❌ FAILED - Segmentation Fault
- **Error Location**: PyTorch multi_head_attention_forward
- **Impact**: Cannot test hybrid strategy integration

### Test Coverage Analysis

| Module | Coverage | Status |
|--------|----------|--------|
| transformer_model.py | 89.13% | ✅ Good |
| transformer_wrapper.py | 89.39% | ✅ Good |
| sequence_preparation.py | 70.83% | ✅ Good |
| error_recovery.py | 72.73% | ✅ Good |
| versioning.py | 94.44% | ✅ Excellent |
| gpu_optimizer.py | 81.58% | ✅ Good |
| multi_task.py | 93.33% | ✅ Excellent |
| online_learning.py | 56.00% | ⚠️ Moderate |
| **Overall System** | **13.63%** | ❌ Below threshold |

### Key Findings

1. **Individual Components**: Most transformer components work well in isolation with good test coverage
2. **Integration Issue**: Critical segmentation fault when testing hybrid strategy
3. **Coverage Gap**: Overall system coverage (13.63%) significantly below 25% requirement
4. **Platform Limitation**: Quantization not supported on macOS (ARM architecture)

### Recommendations

1. **Immediate Actions**:
   - Debug segmentation fault in hybrid strategy
   - Investigate PyTorch compatibility with multi-head attention on macOS ARM
   - Consider testing on x86_64 Linux environment

2. **Coverage Improvements**:
   - Add tests for untested modules (strategies, backtesting, feature engineering)
   - Expand integration tests once hybrid issue is resolved

3. **Next Steps**:
   - Fix hybrid strategy crash before proceeding
   - Complete remaining Phase 3-10 tests after resolution
   - Consider containerized testing environment for consistency

### Status Summary

- **Environment Setup**: ✅ Complete
- **Basic Functionality**: ✅ Working
- **Unit Tests**: ⚠️ Partial (most pass, critical failure in hybrid)
- **Integration Tests**: ❌ Blocked by hybrid strategy crash
- **System Ready for Production**: ❌ No - critical issues must be resolved

The system shows promise with individual components working well, but the hybrid strategy segmentation fault is a blocking issue that must be resolved before proceeding with further testing or production deployment.
