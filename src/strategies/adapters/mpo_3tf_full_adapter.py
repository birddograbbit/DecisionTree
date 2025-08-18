"""
Full MPO-3TF adapter with complete reference implementation logic.

This adapter implements the exact logic from the reference optimization,
including MPO calculation (RSI + Stochastic + MFI), multi-timeframe analysis,
and complex entry/exit conditions with arming logic.
"""

import logging
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

from src.strategies.base_strategy import BaseStrategy
from src.features.indicators import calculate_atr
from src.features.indicators_advanced import calculate_mpo, calculate_mbrsi
from src.features.multi_timeframe_features import MultiTimeframeAggregator
import config

logger = logging.getLogger(__name__)


class EntryState(Enum):
    """Entry state machine states."""
    IDLE = 0
    LOOK_LONG1 = 1
    LOOK_SHORT1 = 2
    LOOK_LONG2 = 3
    LOOK_SHORT2 = 4


class MPO3TFFullAdapter(BaseStrategy):
    """Full MPO-3TF multi-timeframe strategy with reference implementation."""

    def __init__(self) -> None:
        super().__init__()
        
        # Entry control (Optimized)
        self.use_entry1 = True
        self.use_entry2 = True
        self.entry2_min_lm = 1  # Minimum local momentum crossovers
        
        # Multi-timeframe thresholds (Optimized)
        self.ob1 = 93.15  # TF1 overbought
        self.ob2 = 60.64  # TF2 overbought
        self.ob3 = 64.15  # TF3 overbought
        self.os1 = 38.76  # TF1 oversold
        self.os2 = 32.63  # TF2 oversold
        self.os3 = 53.44  # TF3 oversold
        
        # MBRSI gate (Optimized)
        self.use_mbrsi_gate = False
        self.mbrsi_thresh = 47.16
        self.mbrsi_fast = 9
        self.mbrsi_slow = 21
        self.mbrsi_rsi_len = 12
        
        # Risk management (Optimized)
        self.atr_length = 14
        self.sl_mult = 0.50
        self.tp_mult = 4.31
        
        # Dynamic ATR and trailing
        self.dynamic_atr = False
        self.trail_mult = 0.0
        
        # Warmup and position sizing
        self.min_bars_warmup = 49
        self.position_size = 0.1
        self.diagnostics = False
        
        # Multi-timeframe aggregator
        self.aggregator = MultiTimeframeAggregator()
        
        # State tracking
        self.entry_state = EntryState.IDLE
        self.ob1_armed = False
        self.ob2_armed = False
        self.ob3_armed = False
        self.os1_armed = False
        self.os2_armed = False
        self.os3_armed = False

    def initialize(self, config: Dict) -> None:
        """Initialize with configuration."""
        super().initialize(config)
        
        # Update all parameters from config
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.diagnostics = config.get('diagnostics', self.diagnostics)

        logger.info("MPO-3TF Full adapter initialized with optimized config")

    def get_required_features(self) -> List[str]:
        """Get required feature columns."""
        return ['open', 'high', 'low', 'close', 'volume']

    def get_required_timeframes(self) -> List[str]:
        """Get required timeframes."""
        timeframe = '1min'
        if hasattr(self, 'config') and self.config:
            timeframe = self.config.get('timeframe', timeframe)
        return [timeframe]

    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all required indicators including multi-timeframe MPO."""
        df = data.copy()
        
        # Calculate MPO for base timeframe (1min)
        df['mpo_1m'] = calculate_mpo(df, length=14)
        
        # Resample to higher timeframes
        # For 1-minute base: 5min, 10min, 15min
        # For 5-minute base: 10min, 15min, 30min
        base_tf = self.get_required_timeframes()[0]
        if base_tf in ['1min', '1T']:
            tf2, tf3, tf4 = '5T', '10T', '15T'
        else:  # 5min base
            tf2, tf3, tf4 = '10T', '15T', '30T'
        
        # Resample and calculate MPO for each timeframe
        data_5 = self.aggregator.resample_data(df, tf2)
        data_10 = self.aggregator.resample_data(df, tf3)
        data_15 = self.aggregator.resample_data(df, tf4)
        
        # Calculate MPO for each timeframe
        mpo_5 = calculate_mpo(data_5, length=14).reindex(df.index, method='ffill')
        mpo_10 = calculate_mpo(data_10, length=14).reindex(df.index, method='ffill')
        mpo_15 = calculate_mpo(data_15, length=14).reindex(df.index, method='ffill')
        
        df['mpo_5m'] = mpo_5
        df['mpo_10m'] = mpo_10
        df['mpo_15m'] = mpo_15
        
        # Calculate ATR
        df['atr'] = calculate_atr(df, self.atr_length)
        
        # Calculate MBRSI if gate is enabled
        if self.use_mbrsi_gate:
            # Calculate on highest timeframe for stability
            mbrsi_15 = calculate_mbrsi(data_15, self.mbrsi_rsi_len, 
                                       self.mbrsi_fast, self.mbrsi_slow)
            df['mbrsi'] = mbrsi_15.reindex(df.index, method='ffill')
        else:
            df['mbrsi'] = 50  # Neutral value
        
        # Track bars for warmup
        df['bar_count'] = range(len(df))
        
        return df

    def _update_armed_states(self, row: pd.Series) -> None:
        """Update armed states based on threshold crossings."""
        # Check if oscillators cross thresholds (inclusive arming)
        if row['mpo_1m'] >= self.ob1:
            self.ob1_armed = True
        if row['mpo_5m'] >= self.ob2:
            self.ob2_armed = True
        if row['mpo_10m'] >= self.ob3:
            self.ob3_armed = True
        
        if row['mpo_1m'] <= self.os1:
            self.os1_armed = True
        if row['mpo_5m'] <= self.os2:
            self.os2_armed = True
        if row['mpo_10m'] <= self.os3:
            self.os3_armed = True

    def _reset_armed_states(self) -> None:
        """Reset all armed states after entry."""
        self.ob1_armed = False
        self.ob2_armed = False
        self.ob3_armed = False
        self.os1_armed = False
        self.os2_armed = False
        self.os3_armed = False
        self.entry_state = EntryState.IDLE

    def _calculate_entry_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate entry signals with state machine and arming logic."""
        signals = pd.DataFrame(index=df.index)
        signals['long_entry'] = False
        signals['short_entry'] = False
        
        # Skip warmup period
        warmup_mask = df['bar_count'] >= self.min_bars_warmup
        
        for i in range(len(df)):
            if not warmup_mask.iloc[i]:
                continue

            row = df.iloc[i]
            prev_row = df.iloc[i-1] if i > 0 else row

            prev_state = self.entry_state
            # Update armed states
            self._update_armed_states(row)
            if self.diagnostics and prev_state != self.entry_state:
                logger.debug(f"State {prev_state.name} -> {self.entry_state.name} at {df.index[i]}")
            
            # Gate checks
            bull_gate = True
            bear_gate = True
            if self.use_mbrsi_gate:
                bull_gate = row['mbrsi'] >= self.mbrsi_thresh
                bear_gate = row['mbrsi'] <= self.mbrsi_thresh
            
            # Entry 1: Bias + pullback
            if self.use_entry1 and self.entry_state == EntryState.IDLE:
                # Check for bias setup
                below50_3 = row['mpo_15m'] < 50
                above50_3 = row['mpo_15m'] > 50
                above50_1_or_2 = (row['mpo_1m'] > 50) or (row['mpo_5m'] > 50)
                below50_1_or_2 = (row['mpo_1m'] < 50) or (row['mpo_5m'] < 50)
                
                if below50_3 and above50_1_or_2 and bear_gate:
                    self.entry_state = EntryState.LOOK_SHORT1
                elif above50_3 and below50_1_or_2 and bull_gate:
                    self.entry_state = EntryState.LOOK_LONG1
            
            # Cancel bias if TF3 reverses
            if self.entry_state == EntryState.LOOK_SHORT1 and row['mpo_15m'] > 50:
                self.entry_state = EntryState.IDLE
            if self.entry_state == EntryState.LOOK_LONG1 and row['mpo_15m'] < 50:
                self.entry_state = EntryState.IDLE
            
            # Check for pullback triggers (cross 50)
            co1_50 = (prev_row['mpo_1m'] < 50) and (row['mpo_1m'] >= 50)
            cu1_50 = (prev_row['mpo_1m'] > 50) and (row['mpo_1m'] <= 50)
            co2_50 = (prev_row['mpo_5m'] < 50) and (row['mpo_5m'] >= 50)
            cu2_50 = (prev_row['mpo_5m'] > 50) and (row['mpo_5m'] <= 50)
            
            # Entry 1 signals
            if self.use_entry1:
                if self.entry_state == EntryState.LOOK_LONG1 and (co1_50 or co2_50) and bull_gate:
                    signals.iloc[i, signals.columns.get_loc('long_entry')] = True
                    self._reset_armed_states()
                elif self.entry_state == EntryState.LOOK_SHORT1 and (cu1_50 or cu2_50) and bear_gate:
                    signals.iloc[i, signals.columns.get_loc('short_entry')] = True
                    self._reset_armed_states()
            
            # Entry 2: 3x extreme fade
            if self.use_entry2:
                # Check if all armed for extreme setup
                if self.entry_state == EntryState.IDLE:
                    if self.ob1_armed and self.ob2_armed and self.ob3_armed and bear_gate:
                        self.entry_state = EntryState.LOOK_SHORT2
                    elif self.os1_armed and self.os2_armed and self.os3_armed and bull_gate:
                        self.entry_state = EntryState.LOOK_LONG2
                
                # Count pullback triggers
                lm_up_count = 0  # Leaving oversold
                if (prev_row['mpo_1m'] < self.os1) and (row['mpo_1m'] >= self.os1):
                    lm_up_count += 1
                if (prev_row['mpo_5m'] < self.os2) and (row['mpo_5m'] >= self.os2):
                    lm_up_count += 1
                
                lm_dn_count = 0  # Leaving overbought
                if (prev_row['mpo_1m'] > self.ob1) and (row['mpo_1m'] <= self.ob1):
                    lm_dn_count += 1
                if (prev_row['mpo_5m'] > self.ob2) and (row['mpo_5m'] <= self.ob2):
                    lm_dn_count += 1
                
                # Entry 2 signals
                if self.entry_state == EntryState.LOOK_LONG2 and (lm_up_count >= self.entry2_min_lm) and bull_gate:
                    signals.iloc[i, signals.columns.get_loc('long_entry')] = True
                    self._reset_armed_states()
                elif self.entry_state == EntryState.LOOK_SHORT2 and (lm_dn_count >= self.entry2_min_lm) and bear_gate:
                    signals.iloc[i, signals.columns.get_loc('short_entry')] = True
                    self._reset_armed_states()
        
        return signals

    def generate_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
        """Generate features for the strategy."""
        df = self._add_indicators(data)
        
        # Select feature columns
        feature_cols = ['open', 'high', 'low', 'close', 'mpo_1m', 'mpo_5m', 'mpo_10m', 'mpo_15m', 'atr']
        
        if self.use_mbrsi_gate:
            feature_cols.append('mbrsi')
        
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
        # Calculate entry signals
        entry_signals = self._calculate_entry_signals(features)
        
        # Build final signals DataFrame
        signals = pd.DataFrame(index=dates)
        signals['date'] = dates
        signals['symbol'] = self.config.get('symbol', 'SPY') if hasattr(self, 'config') and self.config else 'SPY'
        signals['signal'] = 0
        signals['entry_price'] = features['close']

        # Process signals
        position = 0
        for i in range(len(signals)):
            current_time = dates[i]
            if current_time.hour == 15 and current_time.minute == 59:
                signals.iloc[i, signals.columns.get_loc('signal')] = 0
                position = 0
                continue
            if position == 0:
                # Check for entries
                if entry_signals.iloc[i]['long_entry']:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    position = 1
                elif entry_signals.iloc[i]['short_entry']:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    position = -1
            elif position == 1:
                if entry_signals.iloc[i]['short_entry']:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
                    position = -1
                else:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
            else:
                if entry_signals.iloc[i]['long_entry']:
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                    position = 1
                else:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
        
        # Add risk management levels
        atr = features['atr']
        close = features['close']
        
        # Stop loss and take profit with optimized multipliers
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
        
        # Trailing stop if enabled
        if self.trail_mult > 0:
            signals['trail_activation'] = np.where(
                signals['signal'] == 1,
                close + atr * self.trail_mult,
                np.where(signals['signal'] == -1, close - atr * self.trail_mult, np.nan)
            )
            signals['trail_offset'] = atr * self.trail_mult

        signals['position_size'] = np.where(signals['signal'] != 0, self.position_size, 0)
        if self.diagnostics:
            long_entries = (signals['signal'].diff() == 1).sum()
            short_entries = (signals['signal'].diff() == -1).sum()
            logger.info(f"MPO3TFFullAdapter entries - long: {long_entries}, short: {short_entries}")

        return signals

    def apply_risk_management(self, signals: pd.DataFrame, prices: pd.DataFrame,
                              features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Apply risk management rules intrabar."""
        managed = signals.copy()
        position = 0
        stop = target = trail_act = trail_off = None
        for i in range(len(managed)):
            bar = prices.iloc[i]
            sig = managed.iloc[i]['signal']
            ts = prices.index[i]
            if ts.hour == 15 and ts.minute == 59:
                managed.iloc[i, managed.columns.get_loc('signal')] = 0
                position = 0
                stop = target = trail_act = trail_off = None
                if self.diagnostics:
                    logger.debug(f"EOD flatten at {ts}")
                continue
            if position == 0:
                if sig != 0:
                    position = sig
                    stop = managed.iloc[i].get('stop_loss')
                    target = managed.iloc[i].get('take_profit')
                    trail_act = managed.iloc[i].get('trail_activation')
                    trail_off = managed.iloc[i].get('trail_offset')
                    if self.diagnostics:
                        logger.debug(f"Position opened {position} at {ts}")
                continue

            if trail_act is not None and not np.isnan(trail_act):
                if position == 1 and bar['high'] >= trail_act:
                    stop = max(stop if stop is not None else -np.inf, trail_act - trail_off)
                    trail_act = bar['high'] + trail_off
                elif position == -1 and bar['low'] <= trail_act:
                    stop = min(stop if stop is not None else np.inf, trail_act + trail_off)
                    trail_act = bar['low'] - trail_off

            exit_trade = False
            if position == 1:
                if stop is not None and bar['low'] <= stop:
                    exit_trade = True
                elif target is not None and bar['high'] >= target:
                    exit_trade = True
            else:
                if stop is not None and bar['high'] >= stop:
                    exit_trade = True
                elif target is not None and bar['low'] <= target:
                    exit_trade = True

            if exit_trade:
                managed.iloc[i, managed.columns.get_loc('signal')] = 0
                if self.diagnostics:
                    logger.debug(f"Position closed at {ts}")
                position = 0
                stop = target = trail_act = trail_off = None
            else:
                if position != sig and self.diagnostics:
                    logger.debug(f"Position changed from {position} to {sig} at {ts}")
                managed.iloc[i, managed.columns.get_loc('signal')] = position

        if self.diagnostics:
            exits = (managed['signal'].diff() == -1).sum() + (managed['signal'].diff() == 1).sum()
            logger.info(f"MPO3TFFullAdapter exits: {exits}")
        return managed

    def _calculate_backtest_metrics(self, signals: pd.DataFrame, prices: pd.DataFrame) -> Dict[str, any]:
        """Calculate backtest metrics."""
        aligned_prices = prices.loc[signals.index]
        position = signals['signal'].shift(1).fillna(0)
        returns = position * aligned_prices['open'].pct_change()
        
        # Get timeframe for proper annualization
        timeframe = self.config.get('timeframe', '1min') if hasattr(self, 'config') and self.config else '1min'
        
        # Set commission and slippage based on timeframe
        if timeframe in ['1min', '1T']:
            commission = config.TRANSACTION_COST_5MIN
            slippage = config.SLIPPAGE_5MIN
            annualization_factor = np.sqrt(390 * 252)
            periods_per_year = 390 * 252
        elif timeframe in ['5min', '5T']:
            commission = config.TRANSACTION_COST_5MIN
            slippage = config.SLIPPAGE_5MIN
            annualization_factor = np.sqrt(78 * 252)
            periods_per_year = 78 * 252
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
            'strategy': 'MPO-3TF-Full'
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