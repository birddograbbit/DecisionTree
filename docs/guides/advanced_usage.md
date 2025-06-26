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
