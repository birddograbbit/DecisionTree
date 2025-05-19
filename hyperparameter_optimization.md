# Hyperparameter Optimization Integration

This document describes the hyperparameter optimization features implemented in Phase 1.5 of the v0.2 roadmap.

## Overview

The hyperparameter optimization integration adds the following capabilities to the trading system:

1. **Auto-parameter loading** - Automatically load the best hyperparameters for models
2. **Parameter persistence** - Store optimized parameters in versioned files
3. **Optimization preprocessing CLI** - Run hyperparameter optimization before strategy execution
4. **Regime-specific optimization** - Use different hyperparameters for different market regimes
5. **Adaptive probability thresholds** - Automatically adjust thresholds for calibrated models

## Components

### HyperparameterManager

The `HyperparameterManager` class (`src/models/hyperparameter_manager.py`) provides the following functionality:

- Load best parameters for a model type
- Save optimized parameters to disk
- Create optimized model instances
- Support regime-specific parameter sets

```python
# Example usage
from src.models.hyperparameter_manager import HyperparameterManager

# Create instance
manager = HyperparameterManager()

# Get best parameters
params = manager.get_best_params('random_forest')

# Create optimized model
model = manager.create_optimized_model('random_forest', X, y)

# Get regime-specific models
regime_models = manager.get_regime_specific_models(X, y, 'random_forest', regimes)
```

### ModelFactory Integration

The `ModelFactory` class has been updated to integrate with `HyperparameterManager`:

- Added support for the `use_optimized` parameter to automatically use optimized hyperparameters
- Added support for regime-specific model creation
- Simplified the `create_optimized_model` method to use `HyperparameterManager`

```python
# Example usage
from src.models.model_factory import ModelFactory

# Create model with optimized hyperparameters
model = ModelFactory.create_model('random_forest', use_optimized=True)

# Create regime-specific model
model = ModelFactory.create_model('random_forest', use_optimized=True, regime='strong_uptrend')
```

### Hyperparameter Optimization CLI

The `optimize_hyperparameters.py` script provides a command-line interface for hyperparameter optimization:

```bash
# Optimize all model types
python optimize_hyperparameters.py --data data/raw --model all

# Optimize a specific model type
python optimize_hyperparameters.py --data data/raw --model random_forest --trials 100

# Optimize for specific regimes
python optimize_hyperparameters.py --data data/raw --model random_forest --regime-specific
```

### RegimeAdaptiveStrategy Enhancements

The `RegimeAdaptiveStrategy` class has been enhanced to support:

- Using regime-specific models with optimized hyperparameters
- Dynamically adjusting probability thresholds based on market regime
- More robust regime-specific signal generation

## Usage

### Running Strategy with Optimized Parameters

```bash
# Optimize hyperparameters first
python optimize_hyperparameters.py --data data/raw --model all

# Run strategy with optimized hyperparameters
python strategy_runner.py --data data/raw --model random_forest --use-optimized
```

### Running Strategy with Regime-Specific Parameters

```bash
# Optimize regime-specific hyperparameters
python optimize_hyperparameters.py --data data/raw --model random_forest --regime-specific

# Run regime-adaptive strategy with optimized parameters
python strategy_runner.py --data data/raw --model random_forest --strategy regime_adaptive --use-optimized
```

## Configuration

The hyperparameter optimization framework uses several configuration parameters defined in `config.py`:

- `OPTUNA_TRIALS` - Number of trials for hyperparameter optimization (default: 100)
- `TIMESERIES_CV_SPLITS` - Number of splits for time series cross-validation (default: 5)
- `RANDOM_STATE` - Random seed for reproducibility (default: 42)

Additionally, the following parameters define the hyperparameter search spaces:

- `DECISION_TREE_PARAMS` - Search space for Decision Tree parameters
- `RANDOM_FOREST_PARAMS` - Search space for Random Forest parameters
- `XGBOOST_PARAMS` - Search space for XGBoost parameters

## Directory Structure

Hyperparameters are stored in the following directories:

```
data/
├── hyperparameters/         # Base directory for hyperparameters
│   ├── decision_tree_hyperparameters.pkl     # Latest parameters for Decision Tree
│   ├── random_forest_hyperparameters.pkl     # Latest parameters for Random Forest
│   ├── xgboost_hyperparameters.pkl           # Latest parameters for XGBoost
│   ├── versioned/                            # Versioned hyperparameters
│   │   ├── decision_tree_hyperparameters_v20250519_123456.pkl
│   │   ├── random_forest_hyperparameters_v20250519_123456.pkl
│   │   └── xgboost_hyperparameters_v20250519_123456.pkl
│   └── regimes/                              # Regime-specific hyperparameters
│       ├── decision_tree_strong_uptrend_hyperparameters.pkl
│       ├── random_forest_strong_uptrend_hyperparameters.pkl
│       └── xgboost_strong_uptrend_hyperparameters.pkl
```

## Future Enhancements

Future enhancements to the hyperparameter optimization framework include:

1. Adding support for automated hyperparameter scheduling
2. Implementing more sophisticated optimization algorithms
3. Providing a web-based interface for hyperparameter visualization
4. Adding support for parallel optimization to reduce runtime
