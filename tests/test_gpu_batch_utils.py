import numpy as np
import torch
from src.models.transformer.transformer_model import TimeSeriesTransformer
from src.models.transformer.gpu_optimizer import GPUOptimizedTransformer
from src.models.transformer.batch_predictor import BatchPredictor


def test_gpu_optimizer_train_epoch():
    model = TimeSeriesTransformer(feature_size=3, seq_length=2)
    crit = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters())
    gpu_opt = GPUOptimizedTransformer(model, crit, device='cpu')
    X = torch.randn(4,2,3)
    y = torch.randn(4,1)
    ds = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(ds, batch_size=2)
    gpu_opt.train_epoch(loader, opt)


def test_batch_predictor():
    model = torch.nn.Identity()
    predictor = BatchPredictor(model, batch_size=2)
    data = np.random.randn(4,1)
    preds = predictor.predict(data, device='cpu')
    assert preds.shape[0] == 4
