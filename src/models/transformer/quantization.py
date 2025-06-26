"""Utilities for quantizing transformer models."""

import torch
import torch.quantization as quantization


def quantize_transformer(model, calibration_data):
    """Quantize a transformer model using static quantization."""
    model.eval()
    model.qconfig = quantization.get_default_qconfig("fbgemm")
    quantization.prepare(model, inplace=True)
    with torch.no_grad():
        for batch in calibration_data:
            model(batch)
    quantized = quantization.convert(model, inplace=False)
    return quantized
