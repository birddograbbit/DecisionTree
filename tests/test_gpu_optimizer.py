import torch
from src.models.transformer.transformer_model import TimeSeriesTransformer
from src.models.transformer.gpu_optimizer import GPUOptimizedTransformer


def test_mixed_precision_training():
    model = TimeSeriesTransformer(feature_size=3, seq_length=2)
    opt = torch.optim.Adam(model.parameters())
    gpu_opt = GPUOptimizedTransformer(model, device='cpu')
    X = torch.randn(4, 2, 3)
    y = torch.randn(4, 1)
    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    gpu_opt.fit(loader, opt, epochs=1)
