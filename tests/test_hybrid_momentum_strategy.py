import pandas as pd
import numpy as np

from src.strategies.hybrid_momentum_strategy import HybridMomentumMLStrategy
from src.strategies.base_strategy import BaseStrategy


class DummyMLStrategy(BaseStrategy):
    def initialize(self, config):
        super().initialize(config)
    def generate_features(self, data):
        return data, None, data.index
    def generate_signals(self, features, predictions, dates):
        return pd.DataFrame({'signal': [1, -1, 1]}, index=dates)
    def train(self, data):
        pass
    def predict(self, data):
        idx = data.index
        signals = pd.DataFrame({'signal': [1, -1, 1]}, index=idx)
        preds = np.array([0.8, 0.2, 0.6])
        return signals, preds
    def backtest(self, data):
        pass


class DummyMomentumStrategy(BaseStrategy):
    def initialize(self, config):
        super().initialize(config)
    def generate_features(self, data):
        return data, None, data.index
    def generate_signals(self, features, predictions, dates):
        signals = pd.DataFrame({'signal': [1, -1, 0]}, index=dates)
        return signals
    def train(self, data):
        pass
    def predict(self, data):
        return self.generate_signals(data, None, data.index)
    def backtest(self, data):
        pass


def test_agreement_only_mode():
    data = pd.DataFrame(index=pd.date_range('2023-01-01', periods=3, freq='D'))
    ml = DummyMLStrategy()
    mom = DummyMomentumStrategy()
    strat = HybridMomentumMLStrategy(ml_strategy=ml, momentum_strategy=mom)
    strat.initialize({'agree_only': True})
    strat.is_trained = True
    signals, probs = strat.predict(data)
    assert signals['signal'].tolist() == [1, -1, 0]
    assert np.allclose(probs, [1.0, 0.0, 0.5])


def test_weighted_fusion_mode():
    data = pd.DataFrame(index=pd.date_range('2023-01-01', periods=3, freq='D'))
    ml = DummyMLStrategy()
    mom = DummyMomentumStrategy()
    strat = HybridMomentumMLStrategy(ml_strategy=ml, momentum_strategy=mom)
    strat.initialize({'agree_only': False, 'weights': (0.3, 0.7)})
    strat.is_trained = True
    signals, probs = strat.predict(data)
    assert signals['signal'].tolist() == [1, -1, 0]
    assert np.allclose(probs, [0.94, 0.06, 0.53])
