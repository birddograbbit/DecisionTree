import torch
from src.models.transformer.transformer_model import TimeSeriesTransformer
from src.models.transformer.quantization import quantize_transformer


def test_quantize_model():
    model = TimeSeriesTransformer()
    data = [torch.randn(1,30,9) for _ in range(2)]
    q = quantize_transformer(model, data)
    assert q is not None
