# Operational Workflow

This document outlines the recommended workflow for running and evaluating strategies
in the DecisionTree project after the latest updates.

## 1. Prepare Data
- Place historical CSV files under the `data/` directory.
- For intraday tests, ensure 5-minute or 1-minute files follow the naming
  conventions used by `strategy_runner.py`.

## 2. Optional Feature Audit
```
python strategy_runner.py --mode audit --data <data_path> --output <out_dir> \
    --audit-model random_forest --timeframe 5min
```
This generates feature importance reports in `<out_dir>/feature_audit` without
splitting data manually.

## 3. Run Strategy Comparison
```
python strategy_runner.py --mode compare --data <data_path> \
    --include-momentum --timeframe 5min
```
This evaluates multiple strategies using the new default thresholds (0.55/0.45)
and applies commission and slippage from `config.py` to momentum strategies.

## 4. Run a Single Strategy
```
python strategy_runner.py --mode single --data <data_path> \
    --audit-model random_forest --timeframe 5min
```
The script automatically splits data 70/30 for training and testing and uses the
central `ThresholdManager` for signal generation.

## 5. Retrain and Optimise Models
Use `optimize_hyperparameters.py` for Sharpe-ratio driven tuning:
```
python optimize_hyperparameters.py --model random_forest --timeframe 5min
```
Store resulting parameters in `data/hyperparameters/` for reuse.

## 6. Testing
Run targeted tests to verify critical components:
```
pytest tests/test_threshold_manager.py
```
Additional tests reside under `tests/` and can be executed with `pytest`.

## 7. Configuration
Adjust thresholds and transaction cost settings in `config.py`:
- `BUY_THRESHOLD` / `SELL_THRESHOLD`
- `TRANSACTION_COST` / `SLIPPAGE_RATE` (daily)
- `TRANSACTION_COST_5MIN` / `SLIPPAGE_5MIN` (intraday)

Default configuration:
```python
BUY_THRESHOLD = 0.55
SELL_THRESHOLD = 0.45
USE_ADAPTIVE_THRESHOLDS = 'always'
BUY_PERCENTILE = 70
SELL_PERCENTILE = 30
```

These values are consumed by `ThresholdManager` and momentum adapters during
backtests.

## 8. Symbols and Timeframes

`strategy_runner.py` operates on a single symbol and timeframe per invocation.
Specify them explicitly:

```bash
python strategy_runner.py --mode compare --data <data_path> --symbol SPX --timeframe 1min
```

To analyse multiple symbols or timeframes, run the script in a loop or separate calls.

## 9. Transformer Notes

The transformer model now exposes an sklearn-compatible interface. Cross-validation
works through the `ModelEngine`, but training remains CPU-intensive on macOS.

