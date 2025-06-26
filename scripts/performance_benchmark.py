"""Simple inference benchmark for the transformer model."""
import time
import numpy as np
from src.models.transformer.transformer_model import TimeSeriesTransformer

def benchmark(batch=32, seq_length=30, n_features=9, runs=50):
    model = TimeSeriesTransformer(feature_size=n_features, seq_length=seq_length)
    x = np.random.randn(batch, seq_length, n_features).astype('float32')
    import torch
    x_t = torch.from_numpy(x)
    start = time.time()
    for _ in range(runs):
        _ = model(x_t)
    end = time.time()
    latency_ms = (end - start) / runs * 1000
    print(f"Average latency: {latency_ms:.3f}ms over {runs} runs")

if __name__ == '__main__':
    benchmark()
