# Transformer Model API

This document describes the API for the Transformer components integrated with the DecisionTree system.

## Classes

### `TimeSeriesTransformer`
- Located at `src/models/transformer/transformer_model.py`.
- Provides a PyTorch implementation of a transformer encoder for time series forecasting.

### `TransformerModelWrapper`
- Located at `src/models/transformer/transformer_wrapper.py`.
- Exposes `train`, `predict`, `save_checkpoint`, and `load_checkpoint` methods compatible with the existing `BaseModel` interface.

## Usage

```python
from src.models.model_factory import ModelFactory
from config import TRANSFORMER_CONFIG

model = ModelFactory.create_model('transformer', **TRANSFORMER_CONFIG['default'])
```
