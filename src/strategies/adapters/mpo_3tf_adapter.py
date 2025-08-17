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
        # Entry control parameters
        self.use_entry1 = True
        self.use_entry2 = True
        self.entry2_min_lm = 1  # Minimum local momentum for entry2
        
        # RSI parameters for base timeframe
        self.rsi_period = 14
        
        # Multi-timeframe overbought/oversold thresholds
        self.ob1 = 93.15  # Timeframe 1 overbought
        self.ob2 = 60.64  # Timeframe 2 overbought
        self.ob3 = 64.15  # Timeframe 3 overbought
        self.os1 = 38.76  # Timeframe 1 oversold
        self.os2 = 32.63  # Timeframe 2 oversold
        self.os3 = 53.44  # Timeframe 3 oversold
        
        # MBRSI gate parameters
        self.use_mbrsi_gate = False
        self.mbrsi_thresh = 47.16
        
        # Risk management
        self.atr_length = 14
        self.sl_mult = 0.50  # Stop loss multiplier
        self.tp_mult = 4.31  # Take profit multiplier
        
        # Minimum bars warmup
        self.min_bars_warmup = 49
        
        # Position sizing
        self.position_size = 0.1
        
        # Multi-timeframe aggregator
        self.aggregator = MultiTimeframeAggregator()

    def initialize(self, config: Dict) -> None:
        super().initialize(config)
        # Entry control parameters
        self.use_entry1 = config.get('use_entry1', self.use_entry1)
        self.use_entry2 = config.get('use_entry2', self.use_entry2)
        self.entry2_min_lm = config.get('entry2_min_lm', self.entry2_min_lm)
        
        # RSI parameters
        self.rsi_period = config.get('rsi_period', self.rsi_period)
        
        # Multi-timeframe thresholds
        self.ob1 = config.get('ob1', self.ob1)
        self.ob2 = config.get('ob2', self.ob2)
        self.ob3 = config.get('ob3', self.ob3)
        self.os1 = config.get('os1', self.os1)
        self.os2 = config.get('os2', self.os2)
        self.os3 = config.get('os3', self.os3)
        
        # MBRSI gate parameters
        self.use_mbrsi_gate = config.get('use_mbrsi_gate', self.use_mbrsi_gate)
        self.mbrsi_thresh = config.get('mbrsi_thresh', self.mbrsi_thresh)
        
        # Risk management
        self.atr_length = config.get('atr_length', self.atr_length)
        self.sl_mult = config.get('sl_mult', self.sl_mult)
        self.tp_mult = config.get('tp_mult', self.tp_mult)
        
        # Minimum bars warmup
        self.min_bars_warmup = config.get('min_bars_warmup', self.min_bars_warmup)
        
        # Position sizing
        self.position_size = config.get('position_size', self.position_size)
        logger.info("MPO3TF adapter initialized with optimized config")

    def get_required_features(self) -> List[str]:
        return ['close', 'high', 'low', 'volume']

    def get_required_timeframes(self) -> List[str]:
        timeframe = '1min'
        if hasattr(self, 'config') and self.config:
            timeframe = self.config.get('timeframe', timeframe)
        return [timeframe]

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Calculate RSI for 1-minute (base) timeframe
        df['rsi_1m'] = calculate_rsi(df, window=self.rsi_period)
        
        # Resample to higher timeframes
        data_5 = self.aggregator.resample_data(df, '5T')
        data_10 = self.aggregator.resample_data(df, '10T')
        data_15 = self.aggregator.resample_data(df, '15T')
        
        # Calculate RSI for each timeframe
        rsi_5 = calculate_rsi(data_5, window=self.rsi_period).reindex(df.index, method='ffill')
        rsi_10 = calculate_rsi(data_10, window=self.rsi_period).reindex(df.index, method='ffill')
        rsi_15 = calculate_rsi(data_15, window=self.rsi_period).reindex(df.index, method='ffill')
        
        df['rsi_5m'] = rsi_5
        df['rsi_10m'] = rsi_10
        df['rsi_15m'] = rsi_15
        
        # Calculate ATR
        df['atr'] = calculate_atr(df, self.atr_length)
        
        # Calculate MBRSI (Money Flow RSI) if needed
        if self.use_mbrsi_gate:
            # Calculate money flow
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            raw_money_flow = typical_price * df['volume']
            
            # Separate positive and negative money flow
            positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
            negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
            
            # Calculate money flow ratio
            positive_mf = pd.Series(positive_flow).rolling(self.rsi_period).sum()
            negative_mf = pd.Series(negative_flow).rolling(self.rsi_period).sum()
            
            # Calculate MBRSI
            mf_ratio = positive_mf / (negative_mf + 1e-10)  # Avoid division by zero
            df['mbrsi'] = 100 - (100 / (1 + mf_ratio))
        else:
            df['mbrsi'] = 50  # Neutral value when not used
        
        # Calculate local momentum for entry2
        df['local_momentum'] = df['close'].diff(self.entry2_min_lm)
        
        # Track bars since start for warmup
        df['bar_count'] = range(len(df))
        
        return df

    def generate_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
        df = self._add_indicators(data)
        
        # Build feature list
        feature_cols = ['close', 'rsi_1m', 'rsi_5m', 'rsi_10m', 'rsi_15m', 'atr']
        
        # Add optional features
        if self.use_mbrsi_gate and 'mbrsi' in df.columns:
            feature_cols.append('mbrsi')
        
        if 'local_momentum' in df.columns:
            feature_cols.append('local_momentum')
        
        if 'bar_count' in df.columns:
            feature_cols.append('bar_count')
        
        # Only keep columns that exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        features = df[feature_cols].copy()
        
        # Create target (next bar direction)
        target = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Filter out invalid rows
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
        
        # Skip early bars during warmup period
        warmup_mask = features.get('bar_count', pd.Series(range(len(features)))) >= self.min_bars_warmup
        
        # Entry 1: Multi-timeframe RSI alignment
        entry1_long = False
        entry1_short = False
        
        if self.use_entry1:
            # Long conditions: All RSIs above their respective overbought thresholds
            entry1_long = (
                (features['rsi_1m'] > self.ob1) &
                (features['rsi_5m'] > self.ob2) &
                (features['rsi_10m'] > self.ob3) &
                warmup_mask
            )
            
            # Short conditions: All RSIs below their respective oversold thresholds
            entry1_short = (
                (features['rsi_1m'] < self.os1) &
                (features['rsi_5m'] < self.os2) &
                (features['rsi_10m'] < self.os3) &
                warmup_mask
            )
        
        # Entry 2: With local momentum filter
        entry2_long = False
        entry2_short = False
        
        if self.use_entry2:
            # Entry 2 uses relaxed thresholds but requires momentum confirmation
            local_momentum = features.get('local_momentum', pd.Series(0, index=features.index))
            
            # Long: RSIs moderately bullish + positive momentum
            entry2_long = (
                (features['rsi_1m'] > 60) &  # Relaxed from ob1
                (features['rsi_5m'] > 55) &  # Relaxed from ob2
                (features['rsi_10m'] > 50) &  # Relaxed from ob3
                (local_momentum > 0) &  # Positive momentum
                warmup_mask
            )
            
            # Short: RSIs moderately bearish + negative momentum
            entry2_short = (
                (features['rsi_1m'] < 40) &  # Relaxed from os1
                (features['rsi_5m'] < 45) &  # Relaxed from os2
                (features['rsi_10m'] < 50) &  # Relaxed from os3
                (local_momentum < 0) &  # Negative momentum
                warmup_mask
            )
        
        # Combine entry conditions
        long_cond = entry1_long | entry2_long
        short_cond = entry1_short | entry2_short
        
        # Apply MBRSI gate if enabled
        if self.use_mbrsi_gate:
            mbrsi = features.get('mbrsi', pd.Series(50, index=features.index))
            # MBRSI confirmation: above threshold for longs, below for shorts
            long_cond = long_cond & (mbrsi > self.mbrsi_thresh)
            short_cond = short_cond & (mbrsi < (100 - self.mbrsi_thresh))
        
        # Set signals
        signals.loc[long_cond, 'signal'] = 1
        signals.loc[short_cond, 'signal'] = -1
        
        # Risk management with optimized multipliers
        signals['stop_loss'] = np.where(
            signals['signal'] == 1,
            close - atr * self.sl_mult,
            np.where(signals['signal'] == -1, close + atr * self.sl_mult, np.nan)
        )
        
        signals['take_profit'] = np.where(
            signals['signal'] == 1,
            close + atr * self.tp_mult,
            np.where(signals['signal'] == -1, close - atr * self.tp_mult, np.nan)
        )
        
        # Position sizing
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
