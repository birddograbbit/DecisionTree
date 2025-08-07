# Advanced Usage

This section outlines custom model setups and multi‑GPU training.

## Custom Architectures
Adjust parameters when creating the model:
```python
model = ModelFactory.create_model('transformer', d_model=128, n_heads=4, n_layers=4)
```

## Hyperparameter Tuning
Use `optimize_hyperparameters.py` to run Optuna sweeps.

## Multi‑GPU Training
Set the wrapper's `device` argument to a CUDA device id and enable DistributedDataParallel if needed.

## Intraday Feature Set
Models can be trained on 5‑minute data by supplying `--timeframe 5min` to `strategy_runner.py`. Make sure 5‑minute SPY CSV files exist in `data/raw` (e.g. `historical_data_STOCK_SPY_5_mins_2023-2024.csv`). The intraday pipeline adds hour/minute features, short-window RSI/EMA, rolling volatility, and lagged returns for improved high-frequency performance.

```bash
python strategy_runner.py --data data/raw --model decision_tree --timeframe 5min --output dt_intraday
```

## Meta-Strategy Tuning
Use `strategy_runner.py` with the `meta_strategy` model and adjust the new CLI parameters:
```bash
python strategy_runner.py --data data/raw --model meta_strategy --timeframe 5min \
    --performance-window 390 --switch-cooldown 78 --output meta_run
```
For automated sweeps across common parameter combinations, run:
```bash
python meta_strategy_perf_test/param_sweep.py
```
