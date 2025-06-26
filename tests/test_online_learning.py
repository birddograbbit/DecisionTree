import torch
from src.models.transformer.transformer_model import TimeSeriesTransformer
from src.models.transformer.online_learning import OnlineTransformer


def test_online_learning_updates():
    model = TimeSeriesTransformer(feature_size=3, seq_length=2)
    crit = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters())
    online = OnlineTransformer(model, crit, opt, buffer_size=10)
    for _ in range(20):
        x = torch.randn(2,2,3)
        y = torch.randn(2)
        online.add_experience(x, y)
    assert len(online.buffer) <= 10
