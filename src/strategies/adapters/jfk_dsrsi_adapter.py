import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import logging

from src.strategies.base_strategy import BaseStrategy
from src.features.indicators import calculate_rsi, calculate_atr
import config

logger = logging.getLogger(__name__)


def calculate_dsrsi(df: pd.DataFrame, length: int = 14, smoothing: int = 3) -> pd.Series:
    """Compute Double Smoothed RSI."""
    rsi1 = calculate_rsi(df, window=length)
    rsi_df = pd.DataFrame({'close': rsi1})
    return calculate_rsi(rsi_df, window=smoothing)


def calculate_kps(df: pd.DataFrame, length: int = 14, smooth: int = 3) -> pd.Series:
    """Simplified Kase Permission Stochastic indicator."""
    lowest = df['low'].rolling(length).min()
    highest = df['high'].rolling(length).max()
    kps = 100 * (df['close'] - lowest) / (highest - lowest)
    return kps.rolling(smooth).mean()


def calculate_pst(df: pd.DataFrame, source: str, length: int = 14, smooth: int = 3, x_shift: int = 3) -> pd.Series:
    """Calculate Phase Shift Transform (PST) indicator."""
    # Get source price
    if source == 'open':
        price = df['open']
    elif source == 'high':
        price = df['high']
    elif source == 'low':
        price = df['low']
    else:
        price = df['close']
    
    # Apply smoothing (simplified version of Jurik-like smoothing)
    ema1 = price.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=smooth, adjust=False).mean()
    
    # Apply phase shift
    pst = ema2.shift(-x_shift)
    
    # Fill NaN values at the end with the last valid value
    pst = pst.ffill()
    
    return pst


class JFKDSRSIAdapter(BaseStrategy):
    """JFK DSRsi momentum strategy adapter."""

    def __init__(self) -> None:
        super().__init__()
        # Core DSRSI parameters
        self.dsrsi_length = 30  # Optimized from 14
        self.smoothing_period = 2  # Optimized from 3
        self.source = 'open'  # New: price source
        self.use_dsrsi = False  # New: whether to use DSRSI
        self.volume_weighted = False  # New: volume weighting
        
        # KPS parameters
        self.kps_length = 14
        self.kps_smooth_period = 8  # Optimized from 3
        self.kps_long_exit_threshold = 84  # New: exit threshold for longs
        self.kps_short_exit_threshold = 5  # New: exit threshold for shorts
        
        # PST (Phase Shift Transform) parameters - New
        self.pst_length = 17
        self.pst_smooth = 6
        self.pst_x = 3
        self.jphase = 78.91  # Jurik phase
        
        # Entry/Exit parameters
        self.rsi_long_threshold = 60
        self.rsi_short_threshold = 40
        self.exit_at_opposite_signal = True  # New: exit when opposite signal
        
        # ATR and risk management
        self.atr_length = 18  # Optimized from 14
        self.use_sl = True  # New: use stop loss
        self.use_tp = False  # New: use take profit
        self.use_ts = True  # New: use trailing stop
        self.sl_atr_ratio = 2.62  # Optimized from 3
        self.tp_sl_ratio = 1.77  # New: TP/SL ratio
        
        # Trailing stop parameters - New
        self.ts_method = 'ATR'  # 'ATR' or 'Percent'
        self.ts_atr_multiplier = 0.54
        self.ts_percent = 3.34
        self.ts_source = 'Open'  # 'Open', 'Close', 'SwingHL'
        self.ts_swing_lookback = 15
        
        # Position sizing
        self.position_size = 0.1

    def initialize(self, config: Dict) -> None:
        super().initialize(config)
        # Core DSRSI parameters
        self.dsrsi_length = config.get('dsrsi_length', self.dsrsi_length)
        self.smoothing_period = config.get('smoothing_period', self.smoothing_period)
        self.source = config.get('source', self.source)
        self.use_dsrsi = config.get('use_dsrsi', self.use_dsrsi)
        self.volume_weighted = config.get('volume_weighted', self.volume_weighted)
        
        # KPS parameters
        self.kps_length = config.get('kps_length', self.kps_length)
        self.kps_smooth_period = config.get('kps_smooth_period', self.kps_smooth_period)
        self.kps_long_exit_threshold = config.get('kps_long_exit_threshold', self.kps_long_exit_threshold)
        self.kps_short_exit_threshold = config.get('kps_short_exit_threshold', self.kps_short_exit_threshold)
        
        # PST parameters
        self.pst_length = config.get('pst_length', self.pst_length)
        self.pst_smooth = config.get('pst_smooth', self.pst_smooth)
        self.pst_x = config.get('pst_x', self.pst_x)
        self.jphase = config.get('jphase', self.jphase)
        
        # Entry/Exit parameters
        self.rsi_long_threshold = config.get('rsi_long_threshold', self.rsi_long_threshold)
        self.rsi_short_threshold = config.get('rsi_short_threshold', self.rsi_short_threshold)
        self.exit_at_opposite_signal = config.get('exit_at_opposite_signal', self.exit_at_opposite_signal)
        
        # ATR and risk management
        self.atr_length = config.get('atr_length', self.atr_length)
        self.use_sl = config.get('use_sl', self.use_sl)
        self.use_tp = config.get('use_tp', self.use_tp)
        self.use_ts = config.get('use_ts', self.use_ts)
        self.sl_atr_ratio = config.get('sl_atr_ratio', self.sl_atr_ratio)
        self.tp_sl_ratio = config.get('tp_sl_ratio', self.tp_sl_ratio)
        
        # Trailing stop parameters
        self.ts_method = config.get('ts_method', self.ts_method)
        self.ts_atr_multiplier = config.get('ts_atr_multiplier', self.ts_atr_multiplier)
        self.ts_percent = config.get('ts_percent', self.ts_percent)
        self.ts_source = config.get('ts_source', self.ts_source)
        self.ts_swing_lookback = config.get('ts_swing_lookback', self.ts_swing_lookback)
        
        # Position sizing
        self.position_size = config.get('position_size', self.position_size)
        logger.info("JFKDSRSI adapter initialized with optimized config")

    def get_required_features(self) -> List[str]:
        return ['close', 'high', 'low', 'volume']

    def get_required_timeframes(self) -> List[str]:
        timeframe = '5min'
        if hasattr(self, 'config') and self.config:
            timeframe = self.config.get('timeframe', timeframe)
        return [timeframe]

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        # Get source price
        if self.source == 'open':
            source_price = df['open']
        elif self.source == 'high':
            source_price = df['high']
        elif self.source == 'low':
            source_price = df['low']
        else:
            source_price = df['close']
        
        # Calculate DSRSI if enabled
        if self.use_dsrsi:
            if self.volume_weighted:
                # Volume-weighted typical price
                typical = (df['high'] + df['low'] + df['close']) / 3
                volume_norm = df['volume'] / df['volume'].rolling(20).mean()
                weighted_price = typical * volume_norm
                dsrsi_df = pd.DataFrame({'close': weighted_price})
            else:
                dsrsi_df = pd.DataFrame({'close': source_price})
            df['dsrsi'] = calculate_dsrsi(dsrsi_df, self.dsrsi_length, self.smoothing_period)
        else:
            # Use regular RSI if DSRSI is disabled
            df['dsrsi'] = calculate_rsi(df, window=self.dsrsi_length)
        
        # Calculate KPS with optimized smoothing
        df['kps'] = calculate_kps(df, self.kps_length, self.kps_smooth_period)
        
        # Calculate PST
        df['pst'] = calculate_pst(df, self.source, self.pst_length, self.pst_smooth, self.pst_x)
        
        # Calculate ATR
        df['atr'] = calculate_atr(df, self.atr_length)
        
        # Calculate swing high/low for trailing stop
        if self.ts_source == 'SwingHL':
            df['swing_high'] = df['high'].rolling(self.ts_swing_lookback).max()
            df['swing_low'] = df['low'].rolling(self.ts_swing_lookback).min()
        
        return df

    def generate_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
        df = self._add_indicators(data)
        
        # Select features based on configuration
        feature_cols = ['close', 'dsrsi', 'kps', 'pst', 'atr']
        
        # Add volume if volume-weighted
        if self.volume_weighted and 'volume' in df.columns:
            feature_cols.append('volume')
        
        # Add source price if different from close
        if self.source != 'close' and self.source in df.columns:
            feature_cols.append(self.source)
        
        # Add swing levels if needed for trailing stop
        if self.ts_source == 'SwingHL':
            if 'swing_high' in df.columns:
                feature_cols.append('swing_high')
            if 'swing_low' in df.columns:
                feature_cols.append('swing_low')
        
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
        
        # Entry conditions
        if self.use_dsrsi:
            # Use DSRSI with KPS confirmation
            long_cond = (features['dsrsi'] > self.rsi_long_threshold) & (features['kps'] > 50)
            short_cond = (features['dsrsi'] < self.rsi_short_threshold) & (features['kps'] < 50)
        else:
            # Use KPS and PST trend
            kps_rising = features['kps'] > features['kps'].shift(1)
            kps_falling = features['kps'] < features['kps'].shift(1)
            pst_above = features['pst'] > features['close']
            pst_below = features['pst'] < features['close']
            
            long_cond = kps_rising & pst_above & (features['kps'] > 20)
            short_cond = kps_falling & pst_below & (features['kps'] < 80)
        
        # Exit conditions based on KPS thresholds
        exit_long_cond = features['kps'] > self.kps_long_exit_threshold
        exit_short_cond = features['kps'] < self.kps_short_exit_threshold
        
        # Apply signals
        signals.loc[long_cond, 'signal'] = 1
        signals.loc[short_cond, 'signal'] = -1
        
        # Handle exit at opposite signal
        if self.exit_at_opposite_signal:
            # When a short signal appears, exit longs
            signals.loc[short_cond & (signals['signal'].shift(1) == 1), 'signal'] = 0
            # When a long signal appears, exit shorts
            signals.loc[long_cond & (signals['signal'].shift(1) == -1), 'signal'] = 0
        
        # Apply KPS exit thresholds
        signals.loc[exit_long_cond & (signals['signal'].shift(1) == 1), 'signal'] = 0
        signals.loc[exit_short_cond & (signals['signal'].shift(1) == -1), 'signal'] = 0
        
        # Stop loss calculation
        if self.use_sl:
            signals['stop_loss'] = np.where(
                signals['signal'] == 1,
                close - atr * self.sl_atr_ratio,
                np.where(signals['signal'] == -1, close + atr * self.sl_atr_ratio, np.nan)
            )
        else:
            signals['stop_loss'] = np.nan
        
        # Take profit calculation
        if self.use_tp:
            tp_distance = atr * self.sl_atr_ratio * self.tp_sl_ratio
            signals['take_profit'] = np.where(
                signals['signal'] == 1,
                close + tp_distance,
                np.where(signals['signal'] == -1, close - tp_distance, np.nan)
            )
        else:
            signals['take_profit'] = np.nan
        
        # Trailing stop calculation
        if self.use_ts:
            if self.ts_method == 'ATR':
                trail_offset = atr * self.ts_atr_multiplier
            else:  # Percent method
                if self.ts_source == 'Open':
                    base_price = features['open'] if 'open' in features.columns else close
                elif self.ts_source == 'SwingHL':
                    # Use swing high for shorts, swing low for longs
                    base_price = np.where(
                        signals['signal'] == 1,
                        features['swing_low'] if 'swing_low' in features.columns else close,
                        features['swing_high'] if 'swing_high' in features.columns else close
                    )
                else:  # Close
                    base_price = close
                trail_offset = base_price * (self.ts_percent / 100)
            
            # Add trailing stop activation and offset
            signals['trail_activation'] = np.where(
                signals['signal'] == 1,
                close + trail_offset,  # Activation level for longs
                np.where(signals['signal'] == -1, close - trail_offset, np.nan)
            )
            signals['trail_offset'] = trail_offset
        
        signals['position_size'] = np.where(signals['signal'] != 0, self.position_size, 0)
        return signals

    def apply_risk_management(self, signals: pd.DataFrame, prices: pd.DataFrame,
                              features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        return signals

    def _calculate_backtest_metrics(self, signals: pd.DataFrame, prices: pd.DataFrame) -> Dict[str, any]:
        aligned_prices = prices.loc[signals.index]
        position = signals['signal'].shift(1).fillna(0)
        returns = position * aligned_prices['close'].pct_change()

        timeframe = self.config.get('timeframe', '5min') if hasattr(self, 'config') and self.config else '5min'
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
        if timeframe in ['5min', '5T', '5m']:
            annualization_factor = np.sqrt(78 * 252)
            periods_per_year = 78 * 252
        elif timeframe in ['1min', '1T', '1m']:
            annualization_factor = np.sqrt(390 * 252)
            periods_per_year = 390 * 252
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
            'strategy': 'JFK-DSRSI'
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
