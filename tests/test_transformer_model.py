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

    def test_save_load(self, tmp_path):
        model = TimeSeriesTransformer(feature_size=3, seq_length=4, dropout=0.0)
        x = torch.randn(2,4,3)
        model.eval()
        _ = model(x)
        ckpt = tmp_path / "model.pt"
        model.save_checkpoint(ckpt)
        loaded = TimeSeriesTransformer.load_checkpoint(ckpt)
        loaded.eval()
        out1 = model(x)
        out2 = loaded(x)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_positional_encoding_shape(self):
        model = TimeSeriesTransformer(feature_size=2, seq_length=15)
        assert model.pos_embedding.shape == (1, 15, model.d_model)

    def test_multiple_configs(self):
        params = [
            {'d_model':32,'nhead':2,'num_layers':1},
            {'d_model':64,'nhead':4,'num_layers':2},
        ]
        for p in params:
            m = TimeSeriesTransformer(feature_size=4, seq_length=10, **p)
            out = m(torch.randn(3,10,4))
            assert out.shape == (3,1)
