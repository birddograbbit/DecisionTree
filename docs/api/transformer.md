# Transformer Model API

This document describes the API for the Transformer components integrated with the DecisionTree system.

## Classes

### `TimeSeriesTransformer`
- Located at `src/models/transformer/transformer_model.py`.
- Provides a PyTorch implementation of a transformer encoder for time series forecasting.

### `TransformerModelWrapper`
- Located at `src/models/transformer/transformer_wrapper.py`.
- Exposes `train`, `predict`, `predict_large_dataset`, `save`, and `load` methods compatible with the existing `BaseModel` interface.

### `GPUOptimizedTransformer`
- Located at `src/models/transformer/gpu_optimizer.py`.
- Enables mixed precision training on GPUs.

### `BatchPredictor`
- Located at `src/models/transformer/batch_predictor.py`.
- Provides efficient batch predictions for large arrays.

## Usage

```python
from src.models.model_factory import ModelFactory
from config import TRANSFORMER_CONFIG

model = ModelFactory.create_model('transformer', **TRANSFORMER_CONFIG['default'])
model.train(X_train, y_train)
probs = model.predict(X_test)
```

### Quantization

```python
from src.models.transformer.quantization import quantize_transformer
quantized = quantize_transformer(model.model, calibration_loader)
```
