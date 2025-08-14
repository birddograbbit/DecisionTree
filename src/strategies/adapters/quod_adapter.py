"""
Quod Stochastic Strategy Adapter.

This adapter implements the Quod stochastic-based strategy that:
- Has reversal and pullback modes
- Uses D60 trend logic for entry/exit  
- Includes sophisticated position management with trailing stops
- Has end-of-day force exit capabilities
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import time
from src.strategies.base_strategy import BaseStrategy
from src.strategies.order_management import OrderManagementSystem, OrderType
from src.features.indicators import calculate_stochastic, calculate_atr
from src.features.multi_timeframe_features import MultiTimeframeAggregator
import config

# Configure logging
logger = logging.getLogger(__name__)


class QuodAdapter(BaseStrategy):
    """
    Quod stochastic strategy adapter.
    
    This strategy has multiple operating modes and sophisticated
    position management capabilities.
    """
    
    def __init__(self):
        """Initialize the Quod adapter."""
        super().__init__()
        
        # Strategy modes
        self.use_stoch_reversal = True
        self.use_stoch_pullback = True
        self.use_d60_trend_entry = False
        self.use_d60_trend_exit = True
        self.use_trailing_stop = False
        self.use_force_eod = False
        
        # Stochastic parameters
        self.stoch_k_period = 14
        self.stoch_d_period = 3
        self.stoch_overbought = 80
        self.stoch_oversold = 20
        
        # D60 trend parameters
        self.d60_lookback = 60
        self.trend_threshold = 0.02  # 2% trend strength
        
        # Position management
        self.long_tp_perc = 1.01  # 1% take profit
        self.long_sl_perc = 0.99  # 1% stop loss
        self.long_trail_activation_perc = 1.005  # 0.5% activation
        self.long_trail_offset_ticks = 100
        
        # Exit thresholds
        self.rev_long_exit_count = 3
        self.rev_long_exit_threshold = 80.0
        self.rev_short_exit_count = 3
        self.rev_short_exit_threshold = 20.0
        self.pullback_long_exit_count = 2
        self.pullback_long_exit_threshold = 80.0
        self.pullback_short_exit_count = 2
        self.pullback_short_exit_threshold = 20.0
        
        # End of day exit
        self.end_of_day_hour = 16
        self.end_of_day_minute = 0
        
        # Order management
        self.order_manager = None
        self.aggregator = MultiTimeframeAggregator()
        
    def initialize(self, config: Dict):
        """
        Initialize strategy with configuration.
        
        Parameters:
        -----------
        config : dict
            Strategy configuration
        """
        super().initialize(config)
        
        # Update modes from config
        self.use_stoch_reversal = config.get('use_stoch_reversal', self.use_stoch_reversal)
        self.use_stoch_pullback = config.get('use_stoch_pullback', self.use_stoch_pullback)
        self.use_d60_trend_entry = config.get('use_d60_trend_entry', self.use_d60_trend_entry)
        self.use_d60_trend_exit = config.get('use_d60_trend_exit', self.use_d60_trend_exit)
        self.use_trailing_stop = config.get('use_trailing_stop', self.use_trailing_stop)
        self.use_force_eod = config.get('use_force_eod', self.use_force_eod)
        
        # Update parameters from config
        self.stoch_k_period = config.get('stoch_k_period', self.stoch_k_period)
        self.stoch_d_period = config.get('stoch_d_period', self.stoch_d_period)
        self.stoch_overbought = config.get('stoch_overbought', self.stoch_overbought)
        self.stoch_oversold = config.get('stoch_oversold', self.stoch_oversold)
        
        # Position management parameters
        self.long_tp_perc = config.get('long_tp_perc', self.long_tp_perc)
        self.long_sl_perc = config.get('long_sl_perc', self.long_sl_perc)
        self.long_trail_activation_perc = config.get('long_trail_activation_perc', self.long_trail_activation_perc)
        self.long_trail_offset_ticks = config.get('long_trail_offset_ticks', self.long_trail_offset_ticks)
        
        # Initialize order manager
        allow_same_bar_exit = config.get('allow_same_bar_exit', True)
        self.order_manager = OrderManagementSystem(allow_same_bar_exit)
        
        logger.info("Quod adapter initialized with config")
        
    def get_required_features(self) -> List[str]:
        """
        Get list of required features for this strategy.
        
        Returns:
        --------
        List[str]
            List of required feature names
        """
        return [
            'close', 'high', 'low', 'volume',
            'stoch_k', 'stoch_d', 'atr',
            'd60_trend', 'd60_slope'
        ]
    
    def get_required_timeframes(self) -> List[str]:
        """
        Get list of required timeframes.
        
        Returns:
        --------
        List[str]
            List of required timeframes
        """
        # Single timeframe strategy
        primary = '5T'  # Default 5-minute timeframe
        if hasattr(self, 'config') and self.config:
            primary = self.config.get('primary_timeframe', '5T')
        return [primary]
    
    def get_order_management_config(self) -> Dict[str, any]:
        """
        Get order management configuration.
        
        Returns:
        --------
        Dict[str, any]
            Order management configuration
        """
        return {
            'order_type': 'market',
            'limit_offset_atr': None,
            'order_persistence_bars': 1,
            'allow_same_bar_exit': True,
            'use_trailing_stop': self.use_trailing_stop
        }
    
    def generate_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
        """
        Generate features for the strategy.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
            
        Returns:
        --------
        tuple
            (features, target, dates)
        """
        # Calculate indicators
        data = self._add_indicators(data)
        
        # Create feature matrix
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Stochastic features
        features['stoch_k'] = data['stoch_k']
        features['stoch_d'] = data['stoch_d']
        features['stoch_diff'] = data['stoch_k'] - data['stoch_d']
        
        # D60 trend features
        features['d60_trend'] = data['d60_trend']
        features['d60_slope'] = data['d60_slope']
        features['d60_strength'] = abs(data['d60_slope'])
        
        # Volatility features
        features['atr_normalized'] = data['atr'] / data['close']
        features['high_low_range'] = (data['high'] - data['low']) / data['close']
        
        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Create target (next bar return)
        target = (data['close'].shift(-1) > data['close']).astype(int)
        
        # Drop NaN values
        valid_idx = ~(features.isna().any(axis=1) | target.isna())
        features = features[valid_idx]
        target = target[valid_idx]
        dates = data.index[valid_idx]
        
        return features, target, dates
    
    def generate_signals(self, features: pd.DataFrame, predictions: np.ndarray, 
                        dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Generate trading signals based on Quod logic.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix
        predictions : np.ndarray
            Model predictions (not used for this strategy)
        dates : pd.DatetimeIndex
            Signal dates
            
        Returns:
        --------
        pd.DataFrame
            Trading signals
        """
        # Initialize signals
        signals = pd.DataFrame(index=dates)
        signals['date'] = dates
        signals['symbol'] = self.config.get('symbol', 'SPY') if hasattr(self, 'config') and self.config else 'SPY'
        signals['signal'] = 0
        signals['signal_type'] = ''  # 'reversal' or 'pullback'
        signals['stop_loss'] = np.nan
        signals['take_profit'] = np.nan
        
        # Extract features
        stoch_k = features['stoch_k']
        stoch_d = features['stoch_d']
        d60_trend = features['d60_trend']
        d60_strength = features['d60_strength']
        close_price = features.get('close', pd.Series(index=dates))
        
        # Track state for exit logic
        bars_in_position = 0
        current_position = 0
        exit_counter = 0
        
        # Generate signals
        for i in range(len(signals)):
            # End of day exit check
            if self.use_force_eod and hasattr(dates[i], 'hour'):
                if (dates[i].hour == self.end_of_day_hour and 
                    dates[i].minute >= self.end_of_day_minute - 5):
                    if current_position != 0:
                        signals.iloc[i, signals.columns.get_loc('signal')] = -current_position
                        signals.iloc[i, signals.columns.get_loc('signal_type')] = 'eod_exit'
                        current_position = 0
                        bars_in_position = 0
                        continue
            
            # Skip if missing data
            if pd.isna(stoch_k.iloc[i]) or pd.isna(stoch_d.iloc[i]):
                continue
            
            # Stochastic reversal mode
            if self.use_stoch_reversal and current_position == 0:
                # Long reversal: Stochastic oversold
                if stoch_k.iloc[i] < self.stoch_oversold and stoch_d.iloc[i] < self.stoch_oversold:
                    # Check D60 trend if enabled
                    if self.use_d60_trend_entry:
                        if d60_trend.iloc[i] > 0 and d60_strength.iloc[i] > self.trend_threshold:
                            signals.iloc[i, signals.columns.get_loc('signal')] = 1
                            signals.iloc[i, signals.columns.get_loc('signal_type')] = 'reversal'
                            current_position = 1
                    else:
                        signals.iloc[i, signals.columns.get_loc('signal')] = 1
                        signals.iloc[i, signals.columns.get_loc('signal_type')] = 'reversal'
                        current_position = 1
                
                # Short reversal: Stochastic overbought
                elif stoch_k.iloc[i] > self.stoch_overbought and stoch_d.iloc[i] > self.stoch_overbought:
                    # Check D60 trend if enabled
                    if self.use_d60_trend_entry:
                        if d60_trend.iloc[i] < 0 and d60_strength.iloc[i] > self.trend_threshold:
                            signals.iloc[i, signals.columns.get_loc('signal')] = -1
                            signals.iloc[i, signals.columns.get_loc('signal_type')] = 'reversal'
                            current_position = -1
                    else:
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1
                        signals.iloc[i, signals.columns.get_loc('signal_type')] = 'reversal'
                        current_position = -1
            
            # Stochastic pullback mode
            elif self.use_stoch_pullback and current_position == 0:
                # Long pullback: Stochastic crosses above 20 from oversold
                if (i > 0 and stoch_k.iloc[i-1] < 20 and stoch_k.iloc[i] > 20 and
                    stoch_k.iloc[i] > stoch_d.iloc[i]):
                    # Check D60 trend if enabled
                    if self.use_d60_trend_entry:
                        if d60_trend.iloc[i] > 0:
                            signals.iloc[i, signals.columns.get_loc('signal')] = 1
                            signals.iloc[i, signals.columns.get_loc('signal_type')] = 'pullback'
                            current_position = 1
                    else:
                        signals.iloc[i, signals.columns.get_loc('signal')] = 1
                        signals.iloc[i, signals.columns.get_loc('signal_type')] = 'pullback'
                        current_position = 1
                
                # Short pullback: Stochastic crosses below 80 from overbought
                elif (i > 0 and stoch_k.iloc[i-1] > 80 and stoch_k.iloc[i] < 80 and
                      stoch_k.iloc[i] < stoch_d.iloc[i]):
                    # Check D60 trend if enabled
                    if self.use_d60_trend_entry:
                        if d60_trend.iloc[i] < 0:
                            signals.iloc[i, signals.columns.get_loc('signal')] = -1
                            signals.iloc[i, signals.columns.get_loc('signal_type')] = 'pullback'
                            current_position = -1
                    else:
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1
                        signals.iloc[i, signals.columns.get_loc('signal_type')] = 'pullback'
                        current_position = -1
            
            # Exit logic for existing positions
            if current_position != 0:
                bars_in_position += 1
                
                # Stochastic exit conditions
                exit_signal = False
                
                if current_position == 1:  # Long position
                    # Check stochastic overbought exit
                    if stoch_k.iloc[i] > self.rev_long_exit_threshold:
                        exit_counter += 1
                        if exit_counter >= self.rev_long_exit_count:
                            exit_signal = True
                    else:
                        exit_counter = 0
                    
                    # D60 trend exit
                    if self.use_d60_trend_exit and d60_trend.iloc[i] < 0:
                        exit_signal = True
                        
                elif current_position == -1:  # Short position
                    # Check stochastic oversold exit
                    if stoch_k.iloc[i] < self.rev_short_exit_threshold:
                        exit_counter += 1
                        if exit_counter >= self.rev_short_exit_count:
                            exit_signal = True
                    else:
                        exit_counter = 0
                    
                    # D60 trend exit
                    if self.use_d60_trend_exit and d60_trend.iloc[i] > 0:
                        exit_signal = True
                
                if exit_signal:
                    signals.iloc[i, signals.columns.get_loc('signal')] = -current_position
                    signals.iloc[i, signals.columns.get_loc('signal_type')] = 'exit'
                    current_position = 0
                    bars_in_position = 0
                    exit_counter = 0
            
            # Set position management levels
            if signals.iloc[i, signals.columns.get_loc('signal')] != 0 and not pd.isna(close_price.iloc[i]):
                if signals.iloc[i, signals.columns.get_loc('signal')] == 1:
                    # Long position management
                    signals.iloc[i, signals.columns.get_loc('stop_loss')] = close_price.iloc[i] * self.long_sl_perc
                    signals.iloc[i, signals.columns.get_loc('take_profit')] = close_price.iloc[i] * self.long_tp_perc
                elif signals.iloc[i, signals.columns.get_loc('signal')] == -1:
                    # Short position management (mirror of long)
                    signals.iloc[i, signals.columns.get_loc('stop_loss')] = close_price.iloc[i] * (2 - self.long_sl_perc)
                    signals.iloc[i, signals.columns.get_loc('take_profit')] = close_price.iloc[i] * (2 - self.long_tp_perc)
        
        # Add position sizing
        signals['position_size'] = np.where(
            signals['signal'] != 0,
            self.config.get('position_size', 0.1),
            0
        )
        
        # Add trailing stop configuration if enabled
        if self.use_trailing_stop:
            signals['trail_activation'] = np.where(
                signals['signal'] == 1,
                close_price * self.long_trail_activation_perc,
                np.where(
                    signals['signal'] == -1,
                    close_price * (2 - self.long_trail_activation_perc),
                    np.nan
                )
            )
            signals['trail_offset_ticks'] = self.long_trail_offset_ticks
        
        logger.info(f"Generated {(signals['signal'] != 0).sum()} Quod signals")
        
        return signals
    
    def apply_risk_management(self, signals: pd.DataFrame, prices: pd.DataFrame,
                            features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply advanced risk management including trailing stops.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            Trading signals
        prices : pd.DataFrame
            Price data
        features : pd.DataFrame, optional
            Additional features
            
        Returns:
        --------
        pd.DataFrame
            Modified signals with risk management
        """
        # Risk management is built into the signal generation
        # Additional rules could include:
        # - Maximum daily loss limits
        # - Correlation-based position reduction
        # - Volatility-based position sizing
        
        return signals
    
    def _add_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add required indicators to price data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data
            
        Returns:
        --------
        pd.DataFrame
            Data with indicators
        """
        df = data.copy()
        
        # Stochastic
        df['stoch_k'], df['stoch_d'] = calculate_stochastic(
            df, k_period=self.stoch_k_period, d_period=self.stoch_d_period
        )
        
        # ATR
        df['atr'] = calculate_atr(df, window=14)
        
        # D60 trend (60-period trend analysis)
        df['d60_sma'] = df['close'].rolling(self.d60_lookback).mean()
        df['d60_trend'] = np.where(df['close'] > df['d60_sma'], 1, -1)
        
        # D60 slope (trend strength)
        df['d60_slope'] = df['d60_sma'].ffill().pct_change(10)  # 10-period rate of change
        
        return df
    
    def backtest(self, data: pd.DataFrame, train_data: Optional[pd.DataFrame] = None,
                test_data: Optional[pd.DataFrame] = None, timeframe: str = 'daily') -> Dict[str, any]:
        """
        Run backtest for the strategy.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
        train_data : pd.DataFrame, optional
            Training data (not used for rule-based strategy)
        test_data : pd.DataFrame, optional
            Testing data
            
        Returns:
        --------
        dict
            Backtest results
        """
        # For rule-based strategy, we don't need training
        if test_data is None:
            test_data = data
        
        # Generate features
        features, _, dates = self.generate_features(test_data)
        
        # Generate signals
        signals = self.generate_signals(features, None, dates)
        
        # Apply risk management
        signals = self.apply_risk_management(signals, test_data, features)
        
        # Calculate performance metrics
        metrics = self._calculate_backtest_metrics(signals, test_data)

        # Return both flattened metrics and structured components
        result = {
            **metrics,
            'trades': signals[signals['signal'] != 0],
            'equity_curve': pd.DataFrame(
                {'equity': (1 + metrics.get('total_return', 0))},
                index=[test_data.index[-1]]
            ),
        }
        # Preserve original nested format for backward compatibility
        result['performance'] = metrics
        return result
    
    def _calculate_backtest_metrics(self, signals: pd.DataFrame, 
                                   prices: pd.DataFrame) -> Dict[str, any]:
        """
        Calculate backtest performance metrics.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            Trading signals
        prices : pd.DataFrame
            Price data
            
        Returns:
        --------
        dict
            Performance metrics
        """
        # Align signals with prices
        aligned_prices = prices.loc[signals.index]
        
        # Calculate returns
        position = signals['signal'].shift(1).fillna(0)
        returns = position * aligned_prices['close'].pct_change()

        commission = self.config.get('commission', config.COMMISSION_RATE)
        slippage = self.config.get('slippage', config.SLIPPAGE_RATE)
        trade_changes = signals['signal'].diff().abs().fillna(0)
        returns -= trade_changes * (commission + slippage)
        
        # Determine annualization factor based on timeframe
        timeframe = self.config.get('primary_timeframe', '5T')
        if timeframe in ['5min', '5T', '5m']:
            # 5-minute bars: 78 bars per day * 252 trading days
            annualization_factor = np.sqrt(78 * 252)
            periods_per_year = 78 * 252
        elif timeframe in ['1h', '60min', '60T']:
            # Hourly bars: 6.5 hours per day * 252 trading days
            annualization_factor = np.sqrt(6.5 * 252)
            periods_per_year = 6.5 * 252
        else:
            # Default to daily
            annualization_factor = np.sqrt(252)
            periods_per_year = 252
        
        # Calculate metrics
        total_return = (1 + returns).prod() - 1
        sharpe_ratio = returns.mean() / returns.std() * annualization_factor if returns.std() > 0 else 0
        max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
        
        # Calculate CAGR
        years = max(len(returns) / periods_per_year, 0.01)
        ann_return = (1 + total_return) ** (1 / years) - 1
        
        num_trades = (signals['signal'] != signals['signal'].shift(1)).sum()
        win_rate = (returns > 0).sum() / (returns != 0).sum() if (returns != 0).sum() > 0 else 0
        
        # Count signal types
        reversals = (signals['signal_type'] == 'reversal').sum()
        pullbacks = (signals['signal_type'] == 'pullback').sum()
        
        return {
            'total_return': total_return,
            'ann_return': ann_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'reversal_signals': reversals,
            'pullback_signals': pullbacks,
            'strategy': 'Quod'
        }
    
    def get_metrics(self) -> Dict[str, any]:
        """
        Get strategy performance metrics.
        
        Returns:
        --------
        Dict[str, any]
            Dictionary of performance metrics
        """
        # Start with base metrics
        metrics = super().get_metrics()
        
        # Add Quod specific metrics
        metrics.update({
            'use_stoch_reversal': self.use_stoch_reversal,
            'use_stoch_pullback': self.use_stoch_pullback,
            'use_d60_trend_entry': self.use_d60_trend_entry,
            'use_d60_trend_exit': self.use_d60_trend_exit,
            'use_trailing_stop': self.use_trailing_stop,
            'use_force_eod': self.use_force_eod,
            'stoch_k_period': self.stoch_k_period,
            'stoch_d_period': self.stoch_d_period,
            'stoch_overbought': self.stoch_overbought,
            'stoch_oversold': self.stoch_oversold,
            'd60_lookback': self.d60_lookback,
            'trend_threshold': self.trend_threshold,
            'long_tp_perc': self.long_tp_perc,
            'long_sl_perc': self.long_sl_perc,
            'primary_timeframe': self.get_required_timeframes()[0] if self.get_required_timeframes() else '5T',
        })
        
        # Add signal counts if available
        if hasattr(self, '_signal_counts'):
            metrics.update(self._signal_counts)
        
        return metrics