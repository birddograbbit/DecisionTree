"""
Full JFK_DSRSI adapter with complete reference implementation logic.

This adapter implements the exact logic from the reference optimization,
including Jurik filtering, Phase Shift Transform, and complex entry/exit conditions.
"""

import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy
from src.features.indicators import calculate_atr
from src.features.indicators_advanced import (
    jurik_moving_average,
    phase_shift_transform,
    kase_permission_stochastic,
    calculate_dsrsi
)
import config

logger = logging.getLogger(__name__)


class SignalState(Enum):
    """Signal state machine states."""
    IDLE = 0
    ARMED_LONG = 1
    ARMED_SHORT = 2
    IN_POSITION = 3


class JFKDSRSIFullAdapter(BaseStrategy):
    """Full JFK DSRsi momentum strategy adapter with reference implementation."""

    def __init__(self) -> None:
        super().__init__()
        
        # Core DSRSI parameters (Optimized)
        self.dsrsi_length = 30
        self.smoothing_period = 2
        self.source = 'open'
        self.use_dsrsi = False
        self.volume_weighted = False
        
        # JFKPS parameters (Optimized)
        self.pst_length = 17
        self.pst_smooth = 6
        self.pst_x = 3
        self.jphase = 78.91
        self.kps_smooth_period = 8
        
        # KPS thresholds (Optimized)
        self.kps_long_entry = 50  # KPS > 50 for long
        self.kps_short_entry = 50  # KPS < 50 for short
        self.kps_long_exit_threshold = 84
        self.kps_short_exit_threshold = 5
        
        # Entry/Exit parameters
        self.exit_at_opposite_signal = True
        
        # Risk management (Optimized)
        self.atr_length = 18
        self.use_sl = True
        self.use_tp = False
        self.use_ts = True
        self.sl_atr_ratio = 2.62
        self.tp_sl_ratio = 1.77
        
        # Trailing stop (Optimized)
        self.ts_method = 'ATR'
        self.ts_atr_multiplier = 0.54
        self.ts_percent = 3.34
        self.ts_source = 'Open'
        self.ts_swing_lookback = 15
        
        # Position sizing
        self.position_size = 0.1
        
        # State tracking
        self.signal_state = SignalState.IDLE
        self.min_bars_warmup = 50

    def initialize(self, config: Dict) -> None:
        """Initialize with configuration."""
        super().initialize(config)
        
        # Update all parameters from config
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        logger.info("JFK-DSRSI Full adapter initialized with optimized config")

    def get_required_features(self) -> List[str]:
        """Get required feature columns."""
        return ['open', 'high', 'low', 'close', 'volume']

    def get_required_timeframes(self) -> List[str]:
        """Get required timeframes."""
        timeframe = '5min'
        if hasattr(self, 'config') and self.config:
            timeframe = self.config.get('timeframe', timeframe)
        return [timeframe]

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators."""
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
        
        # Calculate DSRSI
        df['dsrsi'] = calculate_dsrsi(
            df, 
            source=self.source,
            length=self.dsrsi_length,
            smoothing=self.smoothing_period,
            volume_weighted=self.volume_weighted
        )
        
        # Calculate KPS with PST
        df['kps'] = kase_permission_stochastic(
            df,
            length=14,  # Standard stoch length
            smooth=self.kps_smooth_period,
            pst_length=self.pst_length,
            pst_smooth=self.pst_smooth,
            pst_x=self.pst_x,
            jphase=self.jphase
        )
        
        # Calculate PST for trend
        df['pst'] = phase_shift_transform(
            source_price,
            self.source,
            length=self.pst_length,
            smooth=self.pst_smooth,
            x_shift=self.pst_x,
            jphase=self.jphase
        )
        
        # Calculate ATR
        df['atr'] = calculate_atr(df, self.atr_length)
        
        # Calculate swing points for trailing stop
        if self.ts_source == 'SwingHL':
            df['swing_high'] = df['high'].rolling(self.ts_swing_lookback).max()
            df['swing_low'] = df['low'].rolling(self.ts_swing_lookback).min()
        
        # Track bars for warmup
        df['bar_count'] = range(len(df))
        
        return df

    def _calculate_entry_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate entry signals with state machine logic."""
        signals = pd.DataFrame(index=df.index)
        signals['long_entry'] = False
        signals['short_entry'] = False
        
        # Skip warmup period
        warmup_mask = df['bar_count'] >= self.min_bars_warmup
        
        if self.use_dsrsi:
            # DSRSI mode: Use DSRSI with KPS confirmation
            dsrsi = df['dsrsi']
            kps = df['kps']
            
            # Long conditions: DSRSI bullish + KPS rising above 50
            dsrsi_bull = dsrsi > 60
            kps_bull = (kps > self.kps_long_entry) & (kps > kps.shift(1))
            
            # Short conditions: DSRSI bearish + KPS falling below 50
            dsrsi_bear = dsrsi < 40
            kps_bear = (kps < self.kps_short_entry) & (kps < kps.shift(1))
            
            signals.loc[warmup_mask & dsrsi_bull & kps_bull, 'long_entry'] = True
            signals.loc[warmup_mask & dsrsi_bear & kps_bear, 'short_entry'] = True
            
        else:
            # KPS-only mode with PST trend filter
            kps = df['kps']
            pst = df['pst']
            close = df['close']
            
            # KPS momentum
            kps_rising = kps > kps.shift(1)
            kps_falling = kps < kps.shift(1)
            
            # PST trend filter
            pst_bullish = pst > close
            pst_bearish = pst < close
            
            # Entry conditions
            long_cond = kps_rising & pst_bullish & (kps > 20) & (kps < 80)
            short_cond = kps_falling & pst_bearish & (kps < 80) & (kps > 20)
            
            signals.loc[warmup_mask & long_cond, 'long_entry'] = True
            signals.loc[warmup_mask & short_cond, 'short_entry'] = True
        
        return signals

    def _calculate_exit_signals(self, df: pd.DataFrame, entry_signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate exit signals based on KPS thresholds."""
        signals = pd.DataFrame(index=df.index)
        signals['long_exit'] = False
        signals['short_exit'] = False
        
        kps = df['kps']
        
        # KPS threshold exits
        signals.loc[kps > self.kps_long_exit_threshold, 'long_exit'] = True
        signals.loc[kps < self.kps_short_exit_threshold, 'short_exit'] = True
        
        # Exit at opposite signal if enabled
        if self.exit_at_opposite_signal:
            signals.loc[entry_signals['short_entry'], 'long_exit'] = True
            signals.loc[entry_signals['long_entry'], 'short_exit'] = True
        
        return signals

    def generate_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
        """Generate features for the strategy."""
        df = self._add_indicators(data)
        
        # Select feature columns
        feature_cols = ['open', 'high', 'low', 'close', 'dsrsi', 'kps', 'pst', 'atr']
        
        if 'volume' in df.columns:
            feature_cols.append('volume')
        
        if 'swing_high' in df.columns:
            feature_cols.extend(['swing_high', 'swing_low'])
        
        if 'bar_count' in df.columns:
            feature_cols.append('bar_count')
        
        # Only keep existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        features = df[feature_cols].copy()
        
        # Create target
        target = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Filter valid rows
        valid = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid]
        target = target[valid]
        dates = df.index[valid]
        
        return features, target, dates

    def generate_signals(self, features: pd.DataFrame, predictions: Optional[np.ndarray],
                        dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Generate trading signals."""
        # Calculate entry and exit signals
        entry_signals = self._calculate_entry_signals(features)
        exit_signals = self._calculate_exit_signals(features, entry_signals)
        
        # Build final signals DataFrame
        signals = pd.DataFrame(index=dates)
        signals['date'] = dates
        signals['symbol'] = self.config.get('symbol', 'SPY') if hasattr(self, 'config') and self.config else 'SPY'
        signals['signal'] = 0
        signals['entry_price'] = features['close']
        
        # Process signals with proper position tracking
        position = 0
        for i in range(len(signals)):
            if position == 0:
                # Check for entries
                if entry_signals.iloc[i]['long_entry']:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    position = 1
                elif entry_signals.iloc[i]['short_entry']:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    position = -1
            elif position == 1:
                # Check for long exits
                if exit_signals.iloc[i]['long_exit']:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0
                    position = 0
                else:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
            elif position == -1:
                # Check for short exits
                if exit_signals.iloc[i]['short_exit']:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0
                    position = 0
                else:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
        
        # Add risk management levels
        atr = features['atr']
        close = features['close']
        
        # Stop loss
        if self.use_sl:
            signals['stop_loss'] = np.where(
                signals['signal'] == 1,
                close - atr * self.sl_atr_ratio,
                np.where(signals['signal'] == -1, close + atr * self.sl_atr_ratio, np.nan)
            )
        else:
            signals['stop_loss'] = np.nan
        
        # Take profit
        if self.use_tp:
            tp_distance = atr * self.sl_atr_ratio * self.tp_sl_ratio
            signals['take_profit'] = np.where(
                signals['signal'] == 1,
                close + tp_distance,
                np.where(signals['signal'] == -1, close - tp_distance, np.nan)
            )
        else:
            signals['take_profit'] = np.nan
        
        # Trailing stop
        if self.use_ts:
            if self.ts_method == 'ATR':
                trail_offset = atr * self.ts_atr_multiplier
            else:  # Percent method
                if self.ts_source == 'Open':
                    base_price = features['open']
                elif self.ts_source == 'SwingHL':
                    base_price = np.where(
                        signals['signal'] == 1,
                        features.get('swing_low', close),
                        features.get('swing_high', close)
                    )
                else:  # Close
                    base_price = close
                trail_offset = base_price * (self.ts_percent / 100)
            
            signals['trail_activation'] = np.where(
                signals['signal'] == 1,
                close + trail_offset,
                np.where(signals['signal'] == -1, close - trail_offset, np.nan)
            )
            signals['trail_offset'] = trail_offset
        
        signals['position_size'] = np.where(signals['signal'] != 0, self.position_size, 0)
        
        return signals

    def apply_risk_management(self, signals: pd.DataFrame, prices: pd.DataFrame,
                              features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Apply risk management rules."""
        # Risk management is handled in generate_signals
        return signals

    def _calculate_backtest_metrics(self, signals: pd.DataFrame, prices: pd.DataFrame) -> Dict[str, any]:
        """Calculate backtest metrics."""
        aligned_prices = prices.loc[signals.index]
        position = signals['signal'].shift(1).fillna(0)
        returns = position * aligned_prices['close'].pct_change()
        
        # Get timeframe for proper annualization
        timeframe = self.config.get('timeframe', '5min') if hasattr(self, 'config') and self.config else '5min'
        
        # Set commission and slippage based on timeframe
        if timeframe in ['5min', '5T', '1min', '1T']:
            commission = config.TRANSACTION_COST_5MIN
            slippage = config.SLIPPAGE_5MIN
            if timeframe in ['5min', '5T']:
                annualization_factor = np.sqrt(78 * 252)
                periods_per_year = 78 * 252
            else:  # 1min
                annualization_factor = np.sqrt(390 * 252)
                periods_per_year = 390 * 252
        else:
            commission = config.TRANSACTION_COST
            slippage = config.SLIPPAGE_RATE
            annualization_factor = np.sqrt(252)
            periods_per_year = 252
        
        # Apply transaction costs
        trade_changes = signals['signal'].diff().abs().fillna(0)
        returns -= trade_changes * (commission + slippage)
        
        # Calculate metrics
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
            'strategy': 'JFK-DSRSI-Full'
        }

    def backtest(self, data: pd.DataFrame, train_data: Optional[pd.DataFrame] = None,
                 test_data: Optional[pd.DataFrame] = None, timeframe: str = 'daily') -> Dict[str, any]:
        """Run backtest on the strategy."""
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