"""Multi-timeframe strategy wrapper for combining signals across timeframes."""

from typing import List, Optional
import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy
from src.backtesting.engine import BacktestEngine
import config


class MultiTimeframeStrategy(BaseStrategy):
    """Run a base strategy on multiple timeframes and aggregate the signals."""

    def __init__(self):
        super().__init__()
        self.base_strategy_name: str = 'tema'
        self.timeframes: List[str] = []
        self.combine_method: str = 'average'
        self.weights: Optional[List[float]] = None
        self.sub_strategies = []

    def initialize(self, config: dict):
        """Initialize multi-timeframe strategy with configuration."""
        from .strategy_registry import StrategyRegistry  # local import to avoid circular
        super().initialize(config)
        self.base_strategy_name = config.get('base_strategy', 'tema')
        self.timeframes = config.get('timeframes', ['5min', '15min', '1h', '1D'])
        self.combine_method = config.get('combine_method', 'average')
        self.weights = config.get('weights')

        self.sub_strategies = []
        for tf in self.timeframes:
            strat = StrategyRegistry.get_strategy(self.base_strategy_name)
            strat_config = config.copy()
            strat_config['primary_timeframe'] = tf
            strat.initialize(strat_config)
            self.sub_strategies.append((tf, strat))

    def generate_features(self, data):  # pragma: no cover - not used directly
        raise NotImplementedError("MultiTimeframeStrategy delegates feature generation to sub-strategies")

    def generate_signals(self, features, predictions, dates):  # pragma: no cover - not used
        raise NotImplementedError("MultiTimeframeStrategy delegates signal generation to sub-strategies")

    def _resample(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if timeframe in ['5min', '5T', '5m']:
            return data
        agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        rules = {k: v for k, v in agg.items() if k in data.columns}
        return data.resample(timeframe).agg(rules).dropna()

    def _run_signals(self, strat: BaseStrategy, data: pd.DataFrame) -> pd.Series:
        features, _, dates = strat.generate_features(data)
        signals = strat.generate_signals(features, None, dates)
        if 'date' in signals.columns:
            signals = signals.set_index('date')
        return signals['signal']

    def backtest(self, data: pd.DataFrame, train_data: pd.DataFrame = None,
                 test_data: pd.DataFrame = None, timeframe: str = '5min'):
        base_index = data.index
        symbol = self.config.get('symbol', 'SPY')
        weights = self.weights or [1.0] * len(self.sub_strategies)
        total_weight = float(sum(weights))
        combined = np.zeros(len(base_index))

        for (tf, strat), weight in zip(self.sub_strategies, weights):
            tf_data = self._resample(data, tf) if tf != timeframe else data
            sig = self._run_signals(strat, tf_data)
            sig = sig.reindex(base_index, method='ffill').fillna(0)
            combined += sig.values * weight

        if self.combine_method == 'vote':
            final_signal = np.sign(combined)
        else:  # average
            avg = combined / total_weight if total_weight else combined
            final_signal = np.where(avg > 0.5, 1, np.where(avg < -0.5, -1, 0))

        signals_df = pd.DataFrame({'date': base_index, 'symbol': symbol, 'signal': final_signal})
        price_df = data[['close']].copy()
        engine = BacktestEngine(initial_capital=config.INITIAL_CAPITAL,
                                commission=self.config.get('commission', 0.0005),
                                slippage=self.config.get('slippage', 0.0001))
        return engine.run_backtest(signals_df, {symbol: price_df}, timeframe)
