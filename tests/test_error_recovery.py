import torch
from src.models.transformer.error_recovery import RobustTransformer


def test_fallback_prediction():
    class Dummy:
        def __call__(self, x):
            raise RuntimeError('fail')
    fallback_called = {'v': False}
    def fallback(x):
        fallback_called['v'] = True
        return torch.zeros(len(x))
    rob = RobustTransformer(Dummy())
    x = torch.randn(2,3)
    out = rob.predict_with_fallback(x, fallback)
    assert fallback_called['v']
