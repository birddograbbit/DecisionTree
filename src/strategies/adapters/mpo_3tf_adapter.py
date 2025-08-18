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
    IDLE = 0
    LOOK_LONG1 = 1
    LOOK_SHORT1 = 2
    LOOK_LONG2 = 3
    LOOK_SHORT2 = 4


class MPO3TFAdapter(BaseStrategy):
    """Multi-timeframe momentum strategy using MPO with state machine."""

    def __init__(self) -> None:
        super().__init__()
        # Entry control parameters
        self.use_entry1 = True
        self.use_entry2 = True
        self.entry2_min_lm = 1  # Minimum local momentum for entry2

        # MPO parameters
        self.mpo_length = 14
        
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
        self.diagnostics = False

        # Multi-timeframe aggregator and state tracking
        self.aggregator = MultiTimeframeAggregator()
        self.entry_state = EntryState.IDLE
        self.ob1_armed = self.ob2_armed = self.ob3_armed = False
        self.os1_armed = self.os2_armed = self.os3_armed = False

    def initialize(self, config: Dict) -> None:
        super().initialize(config)
        # Entry control parameters
        self.use_entry1 = config.get('use_entry1', self.use_entry1)
        self.use_entry2 = config.get('use_entry2', self.use_entry2)
        self.entry2_min_lm = config.get('entry2_min_lm', self.entry2_min_lm)
        
        # MPO parameters
        self.mpo_length = config.get('mpo_length', self.mpo_length)
        
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
        self.diagnostics = config.get('diagnostics', self.diagnostics)
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
        
        # Calculate MPO for 1-minute timeframe
        df['mpo_1m'] = calculate_mpo(df, length=self.mpo_length)

        # Resample to higher timeframes
        data_5 = self.aggregator.resample_data(df, '5T')
        data_10 = self.aggregator.resample_data(df, '10T')
        data_15 = self.aggregator.resample_data(df, '15T')

        # Calculate MPO for each timeframe
        mpo_5 = calculate_mpo(data_5, length=self.mpo_length).reindex(df.index, method='ffill')
        mpo_10 = calculate_mpo(data_10, length=self.mpo_length).reindex(df.index, method='ffill')
        mpo_15 = calculate_mpo(data_15, length=self.mpo_length).reindex(df.index, method='ffill')

        df['mpo_5m'] = mpo_5
        df['mpo_10m'] = mpo_10
        df['mpo_15m'] = mpo_15

        # Calculate ATR
        df['atr'] = calculate_atr(df, self.atr_length)

        # Calculate MBRSI gate if enabled
        if self.use_mbrsi_gate:
            mbrsi_15 = calculate_mbrsi(data_15)
            df['mbrsi'] = mbrsi_15.reindex(df.index, method='ffill')
        else:
            df['mbrsi'] = 50

        # Track bars since start for warmup
        df['bar_count'] = range(len(df))

        return df

    def _update_armed_states(self, row: pd.Series) -> None:
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
        self.ob1_armed = self.ob2_armed = self.ob3_armed = False
        self.os1_armed = self.os2_armed = self.os3_armed = False
        self.entry_state = EntryState.IDLE

    def _calculate_entry_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(index=df.index)
        signals['long_entry'] = False
        signals['short_entry'] = False

        warmup_mask = df['bar_count'] >= self.min_bars_warmup

        for i in range(len(df)):
            if not warmup_mask.iloc[i]:
                continue
            row = df.iloc[i]
            prev_row = df.iloc[i-1] if i > 0 else row

            prev_state = self.entry_state
            self._update_armed_states(row)
            if self.diagnostics and prev_state != self.entry_state:
                logger.debug(f"State {prev_state.name} -> {self.entry_state.name} at {df.index[i]}")

            bull_gate = True
            bear_gate = True
            if self.use_mbrsi_gate:
                bull_gate = row['mbrsi'] >= self.mbrsi_thresh
                bear_gate = row['mbrsi'] <= self.mbrsi_thresh

            if self.use_entry1 and self.entry_state == EntryState.IDLE:
                below50_3 = row['mpo_15m'] < 50
                above50_3 = row['mpo_15m'] > 50
                above50_1_or_2 = (row['mpo_1m'] > 50) or (row['mpo_5m'] > 50)
                below50_1_or_2 = (row['mpo_1m'] < 50) or (row['mpo_5m'] < 50)

                if below50_3 and above50_1_or_2 and bear_gate:
                    self.entry_state = EntryState.LOOK_SHORT1
                elif above50_3 and below50_1_or_2 and bull_gate:
                    self.entry_state = EntryState.LOOK_LONG1

            if self.entry_state == EntryState.LOOK_SHORT1 and row['mpo_15m'] > 50:
                self.entry_state = EntryState.IDLE
            if self.entry_state == EntryState.LOOK_LONG1 and row['mpo_15m'] < 50:
                self.entry_state = EntryState.IDLE

            co1_50 = (prev_row['mpo_1m'] < 50) and (row['mpo_1m'] >= 50)
            cu1_50 = (prev_row['mpo_1m'] > 50) and (row['mpo_1m'] <= 50)
            co2_50 = (prev_row['mpo_5m'] < 50) and (row['mpo_5m'] >= 50)
            cu2_50 = (prev_row['mpo_5m'] > 50) and (row['mpo_5m'] <= 50)

            if self.use_entry1:
                if self.entry_state == EntryState.LOOK_LONG1 and (co1_50 or co2_50) and bull_gate:
                    signals.iloc[i, signals.columns.get_loc('long_entry')] = True
                    self._reset_armed_states()
                elif self.entry_state == EntryState.LOOK_SHORT1 and (cu1_50 or cu2_50) and bear_gate:
                    signals.iloc[i, signals.columns.get_loc('short_entry')] = True
                    self._reset_armed_states()

            if self.use_entry2:
                if self.entry_state == EntryState.IDLE:
                    if self.ob1_armed and self.ob2_armed and self.ob3_armed and bear_gate:
                        self.entry_state = EntryState.LOOK_SHORT2
                    elif self.os1_armed and self.os2_armed and self.os3_armed and bull_gate:
                        self.entry_state = EntryState.LOOK_LONG2

                lm_up_count = 0
                if (prev_row['mpo_1m'] < self.os1) and (row['mpo_1m'] >= self.os1):
                    lm_up_count += 1
                if (prev_row['mpo_5m'] < self.os2) and (row['mpo_5m'] >= self.os2):
                    lm_up_count += 1

                lm_dn_count = 0
                if (prev_row['mpo_1m'] > self.ob1) and (row['mpo_1m'] <= self.ob1):
                    lm_dn_count += 1
                if (prev_row['mpo_5m'] > self.ob2) and (row['mpo_5m'] <= self.ob2):
                    lm_dn_count += 1

                if self.entry_state == EntryState.LOOK_LONG2 and (lm_up_count >= self.entry2_min_lm) and bull_gate:
                    signals.iloc[i, signals.columns.get_loc('long_entry')] = True
                    self._reset_armed_states()
                elif self.entry_state == EntryState.LOOK_SHORT2 and (lm_dn_count >= self.entry2_min_lm) and bear_gate:
                    signals.iloc[i, signals.columns.get_loc('short_entry')] = True
                    self._reset_armed_states()

        return signals

    def generate_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
        df = self._add_indicators(data)
        
        # Build feature list
        feature_cols = ['close', 'mpo_1m', 'mpo_5m', 'mpo_10m', 'mpo_15m', 'atr']

        if self.use_mbrsi_gate and 'mbrsi' in df.columns:
            feature_cols.append('mbrsi')

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
        entry_signals = self._calculate_entry_signals(features)

        signals = pd.DataFrame(index=dates)
        signals['date'] = dates
        signals['symbol'] = self.config.get('symbol', 'SPY') if hasattr(self, 'config') and self.config else 'SPY'
        signals['signal'] = 0
        signals['entry_price'] = features['close']

        position = 0
        for i in range(len(signals)):
            current_time = dates[i]
            if current_time.hour == 15 and current_time.minute == 59:
                signals.iloc[i, signals.columns.get_loc('signal')] = 0
                position = 0
                continue
            if position == 0:
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

        atr = features['atr']
        close = features['close']

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

        signals['position_size'] = np.where(signals['signal'] != 0, self.position_size, 0)
        if self.diagnostics:
            long_entries = (signals['signal'].diff() == 1).sum()
            short_entries = (signals['signal'].diff() == -1).sum()
            logger.info(f"MPO3TFAdapter entries - long: {long_entries}, short: {short_entries}")

        return signals

    def apply_risk_management(self, signals: pd.DataFrame, prices: pd.DataFrame,
                              features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        managed = signals.copy()
        position = 0
        stop = target = None
        for i in range(len(managed)):
            bar = prices.iloc[i]
            sig = managed.iloc[i]['signal']
            ts = prices.index[i]
            if ts.hour == 15 and ts.minute == 59:
                managed.iloc[i, managed.columns.get_loc('signal')] = 0
                position = 0
                stop = target = None
                if self.diagnostics:
                    logger.debug(f"EOD flatten at {ts}")
                continue
            if position == 0:
                if sig != 0:
                    position = sig
                    stop = managed.iloc[i].get('stop_loss')
                    target = managed.iloc[i].get('take_profit')
                    if self.diagnostics:
                        logger.debug(f"Position opened {position} at {ts}")
                continue

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
                stop = target = None
            else:
                if position != sig and self.diagnostics:
                    logger.debug(f"Position changed from {position} to {sig} at {ts}")
                managed.iloc[i, managed.columns.get_loc('signal')] = position

        if self.diagnostics:
            exits = (managed['signal'].diff() == -1).sum() + (managed['signal'].diff() == 1).sum()
            logger.info(f"MPO3TFAdapter exits: {exits}")
        return managed

    def _calculate_backtest_metrics(self, signals: pd.DataFrame, prices: pd.DataFrame) -> Dict[str, any]:
        aligned_prices = prices.loc[signals.index]
        position = signals['signal'].shift(1).fillna(0)
        returns = position * aligned_prices['open'].pct_change()

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
