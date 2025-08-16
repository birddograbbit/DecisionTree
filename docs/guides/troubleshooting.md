# Troubleshooting

Common issues when running transformer or hybrid models.

* **CUDA not available** – ensure the correct CUDA toolkit is installed and `torch.cuda.is_available()` returns True.
* **Missing data errors** – check that input DataFrames contain all required feature columns.
* **Slow inference** – try quantizing the model with `quantization.quantize_transformer` or reduce sequence length.
* **Data validation warnings** – loaders warn about duplicate timestamps and intraday gaps; examine the messages to confirm they reflect expected market closures.
