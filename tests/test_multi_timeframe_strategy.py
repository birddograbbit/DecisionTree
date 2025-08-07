import numpy as np
import pandas as pd

from src.strategies.multi_timeframe_strategy import MultiTimeframeStrategy
from src.strategies.base_strategy import BaseStrategy
from src.backtesting.engine import BacktestEngine

class DummyStrategy(BaseStrategy):
    def initialize(self, config):
        self.config = config
    def generate_features(self, data):
        return pd.DataFrame(index=data.index), None, data.index
    def generate_signals(self, features, predictions, dates):
        sig = np.where(np.arange(len(dates)) % 2 == 0, 1, -1)
        return pd.DataFrame({'signal': sig}, index=dates)
    def backtest(self, data, train_data=None, test_data=None, timeframe='5min'):
        features, _, dates = self.generate_features(data)
        signals = self.generate_signals(features, None, dates)
        sig_df = signals.copy()
        sig_df['symbol'] = 'SPY'
        sig_df = sig_df.reset_index().rename(columns={'index': 'date'})
        engine = BacktestEngine()
        return engine.run_backtest(sig_df, data)


def test_multi_timeframe_strategy_runs():
    from src.strategies.strategy_registry import StrategyRegistry
    StrategyRegistry.register_strategy('dummy', DummyStrategy)
    dates = pd.date_range('2024-01-01', periods=20, freq='5min')
    data = pd.DataFrame({
        'open': np.linspace(100, 101, 20),
        'high': np.linspace(100, 101, 20),
        'low': np.linspace(100, 101, 20),
        'close': np.linspace(100, 101, 20),
        'volume': np.ones(20)
    }, index=dates)
    config = {
        'name': 'MultiTF Dummy',
        'model_type': 'multi_timeframe',
        'base_strategy': 'dummy',
        'timeframes': ['5min', '15min']
    }
    strategy = StrategyRegistry.get_strategy('multi_timeframe', config)
    results = strategy.backtest(data, timeframe='5min')
    assert 'performance' in results
