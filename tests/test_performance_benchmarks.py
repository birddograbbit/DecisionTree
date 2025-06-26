import time
import torch
from src.models.transformer.transformer_model import TimeSeriesTransformer


def test_inference_latency():
    model = TimeSeriesTransformer()
    x = torch.randn(1,30,9)
    start = time.time()
    for _ in range(20):
        model(x)
    latency = (time.time() - start)/20
    assert latency < 0.1
