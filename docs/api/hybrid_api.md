# Hybrid Strategy API

This reference covers the classes combining decision trees and transformers.

## `HybridMLStrategy`
* Location: `src/models/ensemble/hybrid_strategy.py`
* Combines predictions from a decision tree model and a transformer model.
* Important methods: `generate_signals`, `predict`, `backtest`.
* Supports regime based weighting via the `weight_config` parameter.

## Creating Hybrid Models
Use `ModelFactory.create_model('hybrid', dt_params={}, tf_params={})` to build a hybrid instance.
