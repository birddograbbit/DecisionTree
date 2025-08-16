import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy
from src.features.indicators import calculate_rsi, calculate_atr
from src.features.multi_timeframe_features import MultiTimeframeAggregator
import config

logger = logging.getLogger(__name__)


class MPO3TFAdapter(BaseStrategy):
    """Multi-timeframe momentum strategy using RSI alignment."""

    def __init__(self) -> None:
        super().__init__()
        self.rsi_period = 14
        self.rsi_upper = 60
        self.rsi_lower = 40
        self.atr_length = 14
        self.atr_stop_loss = 3
        self.atr_take_profit = 3
        self.position_size = 0.1
        self.aggregator = MultiTimeframeAggregator()

    def initialize(self, config: Dict) -> None:
        super().initialize(config)
        self.rsi_period = config.get('rsi_period', self.rsi_period)
        self.rsi_upper = config.get('rsi_upper', self.rsi_upper)
        self.rsi_lower = config.get('rsi_lower', self.rsi_lower)
        self.atr_length = config.get('atr_length', self.atr_length)
        self.atr_stop_loss = config.get('atr_stop_loss', self.atr_stop_loss)
        self.atr_take_profit = config.get('atr_take_profit', self.atr_take_profit)
        self.position_size = config.get('position_size', self.position_size)
        logger.info("MPO3TF adapter initialized with config")

    def get_required_features(self) -> List[str]:
        return ['close', 'high', 'low', 'volume']

    def get_required_timeframes(self) -> List[str]:
        timeframe = '1min'
        if hasattr(self, 'config') and self.config:
            timeframe = self.config.get('timeframe', timeframe)
        return [timeframe]

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['rsi_1m'] = calculate_rsi(df, window=self.rsi_period)

        data_5 = self.aggregator.resample_data(df, '5T')
        data_10 = self.aggregator.resample_data(df, '10T')
        data_15 = self.aggregator.resample_data(df, '15T')

        rsi_5 = calculate_rsi(data_5, window=self.rsi_period).reindex(df.index, method='ffill')
        rsi_10 = calculate_rsi(data_10, window=self.rsi_period).reindex(df.index, method='ffill')
        rsi_15 = calculate_rsi(data_15, window=self.rsi_period).reindex(df.index, method='ffill')

        df['rsi_5m'] = rsi_5
        df['rsi_10m'] = rsi_10
        df['rsi_15m'] = rsi_15
        df['atr'] = calculate_atr(df, self.atr_length)
        return df

    def generate_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
        df = self._add_indicators(data)
        features = df[['close', 'rsi_1m', 'rsi_5m', 'rsi_10m', 'rsi_15m', 'atr']].copy()
        target = (df['close'].shift(-1) > df['close']).astype(int)
        valid = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid]
        target = target[valid]
        dates = df.index[valid]
        return features, target, dates

    def generate_signals(self, features: pd.DataFrame, predictions: Optional[np.ndarray],
                        dates: pd.DatetimeIndex) -> pd.DataFrame:
        signals = pd.DataFrame(index=dates)
        signals['date'] = dates
        signals['symbol'] = self.config.get('symbol', 'SPY') if hasattr(self, 'config') and self.config else 'SPY'
        signals['signal'] = 0
        signals['entry_price'] = features['close']
        atr = features['atr']
        close = features['close']

        long_cond = (
            (features['rsi_1m'] > self.rsi_upper) &
            (features['rsi_5m'] > self.rsi_upper) &
            (features['rsi_10m'] > self.rsi_upper) &
            (features['rsi_15m'] > self.rsi_upper)
        )
        short_cond = (
            (features['rsi_1m'] < self.rsi_lower) &
            (features['rsi_5m'] < self.rsi_lower) &
            (features['rsi_10m'] < self.rsi_lower) &
            (features['rsi_15m'] < self.rsi_lower)
        )
        signals.loc[long_cond, 'signal'] = 1
        signals.loc[short_cond, 'signal'] = -1

        signals['stop_loss'] = np.where(
            signals['signal'] == 1,
            close - atr * self.atr_stop_loss,
            np.where(signals['signal'] == -1, close + atr * self.atr_stop_loss, np.nan)
        )
        signals['take_profit'] = np.where(
            signals['signal'] == 1,
            close + atr * self.atr_take_profit,
            np.where(signals['signal'] == -1, close - atr * self.atr_take_profit, np.nan)
        )
        signals['position_size'] = np.where(signals['signal'] != 0, self.position_size, 0)
        return signals

    def apply_risk_management(self, signals: pd.DataFrame, prices: pd.DataFrame,
                              features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return signals

    def _calculate_backtest_metrics(self, signals: pd.DataFrame, prices: pd.DataFrame) -> Dict[str, any]:
        aligned_prices = prices.loc[signals.index]
        position = signals['signal'].shift(1).fillna(0)
        returns = position * aligned_prices['close'].pct_change()

        timeframe = self.config.get('timeframe', '1min') if hasattr(self, 'config') and self.config else '1min'
        default_commission = (
            config.TRANSACTION_COST_5MIN
            if timeframe in ['5min', '5T', '1min', '1T']
            else config.TRANSACTION_COST
        )
        default_slippage = (
            config.SLIPPAGE_5MIN
            if timeframe in ['5min', '5T', '1min', '1T']
            else config.SLIPPAGE_RATE
        )
        commission = self.config.get('commission', default_commission)
        slippage = self.config.get('slippage', default_slippage)
        trade_changes = signals['signal'].diff().abs().fillna(0)
        returns -= trade_changes * (commission + slippage)
        if timeframe in ['1min', '1T', '1m']:
            annualization_factor = np.sqrt(390 * 252)
            periods_per_year = 390 * 252
        elif timeframe in ['5min', '5T', '5m']:
            annualization_factor = np.sqrt(78 * 252)
            periods_per_year = 78 * 252
        else:
            annualization_factor = np.sqrt(252)
            periods_per_year = 252

        total_return = (1 + returns).prod() - 1
        sharpe_ratio = returns.mean() / returns.std() * annualization_factor if returns.std() > 0 else 0
        max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
        num_trades = (signals['signal'] != signals['signal'].shift(1)).sum()
        win_rate = (returns > 0).sum() / (returns != 0).sum() if (returns != 0).sum() > 0 else 0
        years = max(len(returns) / periods_per_year, 0.01)
        ann_return = (1 + total_return) ** (1 / years) - 1

        return {
            'total_return': total_return,
            'ann_return': ann_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'strategy': 'MPO-3TF'
        }

    def backtest(self, data: pd.DataFrame, train_data: Optional[pd.DataFrame] = None,
                 test_data: Optional[pd.DataFrame] = None, timeframe: str = 'daily') -> Dict[str, any]:
        if test_data is None:
            test_data = data
        features, _, dates = self.generate_features(test_data)
        signals = self.generate_signals(features, None, dates)
        signals = self.apply_risk_management(signals, test_data, features)
        metrics = self._calculate_backtest_metrics(signals, test_data)
        result = {
            **metrics,
            'trades': signals[signals['signal'] != 0],
            'equity_curve': pd.DataFrame({'equity': (1 + metrics.get('total_return', 0))}, index=[test_data.index[-1]])
        }
        result['performance'] = metrics
        return result
