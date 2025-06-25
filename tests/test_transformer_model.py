import torch
import numpy as np
from src.models.transformer.transformer_model import TimeSeriesTransformer

class TestTimeSeriesTransformer:
    def test_model_initialization(self):
        configs = [
            {'feature_size': 10, 'num_layers': 2, 'd_model': 64},
            {'feature_size': 5, 'num_layers': 4, 'd_model': 128},
        ]
        for config in configs:
            model = TimeSeriesTransformer(**config)
            assert model is not None

    def test_forward_pass(self):
        model = TimeSeriesTransformer(feature_size=9, seq_length=30)
        for batch in [1, 32]:
            x = torch.randn(batch, 30, 9)
            out = model(x)
            assert out.shape == (batch, 1)

    def test_gradient_flow(self):
        model = TimeSeriesTransformer(feature_size=9)
        x = torch.randn(16, 30, 9, requires_grad=True)
        out = model(x)
        loss = out.mean()
        loss.backward()
        assert x.grad is not None
