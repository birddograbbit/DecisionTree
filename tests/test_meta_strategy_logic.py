import pandas as pd
import numpy as np
import pytest

from src.strategies.meta_strategy import MetaStrategy
from src.strategies.strategy_registry import StrategyRegistry
from src.strategies.base_strategy import BaseStrategy


class DummyStrategy(BaseStrategy):
    """Minimal strategy for testing meta-strategy behaviour."""

    def __init__(self, name):
        super().__init__()
        self.name = name

    def initialize(self, config):
        self.config = config or {}

    def generate_features(self, data):
        X = data.copy()
        y = np.zeros(len(data))
        return X, y, data.index

    def generate_signals(self, features, predictions, dates):
        return pd.DataFrame({'signal': np.zeros(len(dates))}, index=dates)

    def backtest(self, data, train_data=None, test_data=None, timeframe='daily'):
        return {}


@pytest.fixture
def dummy_data():
    return pd.DataFrame({'feature': [0]}, index=pd.date_range('2020', periods=1))


def test_meta_strategy_selects_best_performance(monkeypatch, dummy_data):
    sharpe_map = {
        's1': {'sharpe_ratio': 0.0, 'insufficient_data': False},
        's2': {'sharpe_ratio': 1.0, 'insufficient_data': False},
    }

    def fake_create_strategy(name, config=None):
        return DummyStrategy(name)

    def fake_get_performance_stats(name, window):
        return sharpe_map.get(name, {'sharpe_ratio': 0, 'insufficient_data': True})

    monkeypatch.setattr(StrategyRegistry, 'create_strategy', classmethod(lambda cls, name, config=None: fake_create_strategy(name, config)))
    monkeypatch.setattr(StrategyRegistry, 'get_performance_stats', classmethod(lambda cls, name, window=100: fake_get_performance_stats(name, window)))

    config = {
        'strategies': ['s1', 's2'],
        'selection_method': 'performance',
        'performance_window': 10,
        'switch_cooldown': 0,
    }
    meta = MetaStrategy(config)

    features = dummy_data
    dates = dummy_data.index
    predictions = np.zeros(len(features))

    meta.generate_signals(features, predictions, dates)

    assert meta.current_strategy_name == 's2'


def test_meta_strategy_respects_cooldown(monkeypatch, dummy_data):
    sharpe_map = {
        's1': {'sharpe_ratio': 0.0, 'insufficient_data': False},
        's2': {'sharpe_ratio': 1.0, 'insufficient_data': False},
    }

    monkeypatch.setattr(StrategyRegistry, 'create_strategy', classmethod(lambda cls, name, config=None: DummyStrategy(name)))
    monkeypatch.setattr(StrategyRegistry, 'get_performance_stats', classmethod(lambda cls, name, window=100: sharpe_map[name]))

    config = {
        'strategies': ['s1', 's2'],
        'selection_method': 'performance',
        'performance_window': 10,
        'switch_cooldown': 5,
    }
    meta = MetaStrategy(config)

    features = dummy_data
    dates = dummy_data.index
    predictions = np.zeros(len(features))

    # First call should not switch due to cooldown
    meta.generate_signals(features, predictions, dates)
    assert meta.current_strategy_name == 's1'

    # Set bars_since_switch to cooldown threshold and call again
    meta.bars_since_switch = meta.switch_cooldown
    meta.generate_signals(features, predictions, dates)
    assert meta.current_strategy_name == 's2'


def test_meta_strategy_regime_selection(monkeypatch, dummy_data):
    def fake_create_strategy(name, config=None):
        return DummyStrategy(name)

    monkeypatch.setattr(StrategyRegistry, 'create_strategy', classmethod(lambda cls, name, config=None: fake_create_strategy(name, config)))

    config = {
        'strategies': ['s1', 's2'],
        'selection_method': 'regime',
        'regime_map': {'bull': 's2', 'neutral': 's1'},
    }
    meta = MetaStrategy(config)

    # Patch regime detector to return "bull"
    monkeypatch.setattr(meta.regime_detector, 'get_current_regime', lambda: {'regime_label': 'bull'})

    selected = meta._select_strategy(dummy_data, dummy_data.index)
    assert selected == 's2'
