# Transformer API Reference

This document describes the main classes for the transformer subsystem.

## `TimeSeriesTransformer`
* Location: `src/models/transformer/transformer_model.py`
* Core PyTorch module implementing the encoder architecture.
* Methods: `forward`, `predict_proba`, `save_checkpoint`, `load_checkpoint`.

## `TransformerModelWrapper`
* Location: `src/models/transformer/transformer_wrapper.py`
* Provides a high level interface compatible with the rest of the system.
* Key methods: `train`, `predict`, `predict_large_dataset`, `save`, `load`.

## Utility Modules
* `gpu_optimizer.GPUOptimizedTransformer` – mixed precision training helper.
* `quantization.quantize_transformer` – static quantization helper.
* `online_learning.OnlineTransformer` – streaming updates.
* `multi_task.MultiTaskTransformer` – multi‑output head wrapper.
* `versioning.ModelVersionManager` – simple model registry.

