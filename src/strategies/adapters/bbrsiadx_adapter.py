"""
BB-RSI-ADX Momentum Strategy Adapter.

This adapter implements the BB-RSI-ADX momentum continuation strategy that:
- Uses RSI extremes (>70 for long, <30 for short) to identify momentum
- Confirms with Supertrend for trend direction
- Filters with ADX for trend strength
- Enters on pullbacks to Bollinger Bands
- Requires dual timeframe analysis (primary + 4h)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from src.strategies.base_strategy import BaseStrategy
from src.strategies.order_management import OrderManagementSystem, OrderType
from src.features.indicators import (
    calculate_rsi, calculate_bollinger_bands, calculate_atr,
    calculate_adx, calculate_supertrend
)
from src.features.multi_timeframe_features import MultiTimeframeAggregator

# Configure logging
logger = logging.getLogger(__name__)


class BBRSIADXAdapter(BaseStrategy):
    """
    BB-RSI-ADX momentum continuation strategy adapter.
    
    This is a momentum strategy that trades WITH extreme moves, not against them.
    It combines multiple indicators across timeframes for high-probability entries.
    """
    
    def __init__(self):
        """Initialize the BB-RSI-ADX adapter."""
        super().__init__()
        
        # Strategy parameters
        self.bb_period = 20
        self.bb_std = 2
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.supertrend_period = 10
        self.supertrend_multiplier = 3
        self.adx_primary_threshold = 20
        self.adx_secondary_threshold = 40
        self.atr_stop_multiplier = 6
        self.bars_to_enter = 1  # Order persistence
        
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
        
        # Update parameters from config
        self.bb_period = config.get('bb_period', self.bb_period)
        self.bb_std = config.get('bb_std', self.bb_std)
        self.rsi_period = config.get('rsi_period', self.rsi_period)
        self.rsi_overbought = config.get('rsi_overbought', self.rsi_overbought)
        self.rsi_oversold = config.get('rsi_oversold', self.rsi_oversold)
        self.supertrend_period = config.get('supertrend_period', self.supertrend_period)
        self.supertrend_multiplier = config.get('supertrend_multiplier', self.supertrend_multiplier)
        self.adx_primary_threshold = config.get('adx_primary_threshold', self.adx_primary_threshold)
        self.adx_secondary_threshold = config.get('adx_secondary_threshold', self.adx_secondary_threshold)
        self.atr_stop_multiplier = config.get('atr_stop_multiplier', self.atr_stop_multiplier)
        self.bars_to_enter = config.get('bars_to_enter', self.bars_to_enter)
        
        # Initialize order manager
        allow_same_bar_exit = config.get('allow_same_bar_exit', False)
        self.order_manager = OrderManagementSystem(allow_same_bar_exit)
        
        logger.info("BB-RSI-ADX adapter initialized with config")
        
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
            'rsi', 'bb_upper', 'bb_lower', 'bb_middle',
            'adx', 'plus_di', 'minus_di', 'atr',
            'supertrend', 'supertrend_direction'
        ]
    
    def get_required_timeframes(self) -> List[str]:
        """
        Get list of required timeframes.
        
        Returns:
        --------
        List[str]
            List of required timeframes
        """
        # Primary timeframe (from config) + 4h for confirmation
        primary = '1h'  # Default primary timeframe
        if hasattr(self, 'config') and self.config:
            primary = self.config.get('primary_timeframe', '1h')
        return [primary, '4h']
    
    def get_order_management_config(self) -> Dict[str, any]:
        """
        Get order management configuration.
        
        Returns:
        --------
        Dict[str, any]
            Order management configuration
        """
        return {
            'order_type': 'limit',
            'limit_offset_atr': 0,  # Enter at Bollinger Band
            'order_persistence_bars': self.bars_to_enter,
            'allow_same_bar_exit': False,
            'use_trailing_stop': False
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
        # Calculate indicators for primary timeframe
        data = self._add_indicators(data)
        
        # For multi-timeframe, we'll need the 4h data passed separately
        # This is handled in the strategy runner
        
        # Create feature matrix
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Indicator features
        features['rsi'] = data['rsi']
        features['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        features['adx'] = data['adx']
        features['di_diff'] = data['plus_di'] - data['minus_di']
        features['atr_normalized'] = data['atr'] / data['close']
        features['supertrend_signal'] = data['supertrend_direction']
        
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
        Generate trading signals based on BB-RSI-ADX logic.
        
        This overrides the model predictions with rule-based logic.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix (includes multi-timeframe features)
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
        signals['entry_price'] = np.nan
        signals['stop_loss'] = np.nan
        signals['take_profit'] = np.nan
        
        # Extract primary and secondary timeframe features
        # Assuming features contain both timeframes with suffixes
        primary_tf = self.get_required_timeframes()[0]
        secondary_tf = '4h'
        
        # Get indicators from features
        rsi_primary = features.get(f'rsi_{primary_tf}', features.get('rsi', pd.Series(index=dates)))
        rsi_secondary = features.get(f'rsi_{secondary_tf}', rsi_primary)
        
        adx_primary = features.get(f'adx_{primary_tf}', features.get('adx', pd.Series(index=dates)))
        adx_secondary = features.get(f'adx_{secondary_tf}', adx_primary)
        
        supertrend_dir = features.get(f'supertrend_direction_{secondary_tf}', 
                                     features.get('supertrend_signal', pd.Series(index=dates)))
        
        bb_upper = features.get(f'bb_upper_{primary_tf}', features.get('bb_upper', pd.Series(index=dates)))
        bb_lower = features.get(f'bb_lower_{primary_tf}', features.get('bb_lower', pd.Series(index=dates)))
        
        atr = features.get(f'atr_{primary_tf}', features.get('atr_normalized', pd.Series(index=dates)))
        close_price = features.get(f'close_{primary_tf}', features.get('close', pd.Series(index=dates)))
        
        # Generate signals based on BB-RSI-ADX rules
        for i in range(len(signals)):
            # Skip if missing data
            if pd.isna(rsi_secondary.iloc[i]) or pd.isna(adx_secondary.iloc[i]):
                continue
            
            # Long signal conditions
            long_conditions = (
                rsi_secondary.iloc[i] > self.rsi_overbought and  # Momentum continuation
                supertrend_dir.iloc[i] > 0 and  # Uptrend
                adx_primary.iloc[i] > self.adx_primary_threshold and
                adx_secondary.iloc[i] > self.adx_secondary_threshold
            )
            
            # Short signal conditions
            short_conditions = (
                rsi_secondary.iloc[i] < self.rsi_oversold and  # Momentum continuation
                supertrend_dir.iloc[i] < 0 and  # Downtrend
                adx_primary.iloc[i] > self.adx_primary_threshold and
                adx_secondary.iloc[i] > self.adx_secondary_threshold
            )
            
            if long_conditions:
                signals.iloc[i, signals.columns.get_loc('signal')] = 1
                # Entry at lower Bollinger Band
                signals.iloc[i, signals.columns.get_loc('entry_price')] = bb_lower.iloc[i]
                # Stop loss at 6 ATR below entry
                signals.iloc[i, signals.columns.get_loc('stop_loss')] = (
                    bb_lower.iloc[i] - self.atr_stop_multiplier * atr.iloc[i] * close_price.iloc[i]
                )
                # Take profit at upper Bollinger Band
                signals.iloc[i, signals.columns.get_loc('take_profit')] = bb_upper.iloc[i]
                
            elif short_conditions:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1
                # Entry at upper Bollinger Band
                signals.iloc[i, signals.columns.get_loc('entry_price')] = bb_upper.iloc[i]
                # Stop loss at 6 ATR above entry
                signals.iloc[i, signals.columns.get_loc('stop_loss')] = (
                    bb_upper.iloc[i] + self.atr_stop_multiplier * atr.iloc[i] * close_price.iloc[i]
                )
                # Take profit at lower Bollinger Band
                signals.iloc[i, signals.columns.get_loc('take_profit')] = bb_lower.iloc[i]
        
        # Add position sizing based on signal strength
        signals['position_size'] = np.where(
            signals['signal'] != 0,
            self.config.get('position_size', 0.1),
            0
        )
        
        logger.info(f"Generated {(signals['signal'] != 0).sum()} BB-RSI-ADX signals")
        
        return signals
    
    def apply_risk_management(self, signals: pd.DataFrame, prices: pd.DataFrame,
                            features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply risk management including dynamic exits.
        
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
        # The stop loss and take profit are already set in generate_signals
        # Here we could add additional risk management like:
        # - Maximum position limits
        # - Correlation-based position sizing
        # - Volatility-based adjustments
        
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
        
        # RSI
        df['rsi'] = calculate_rsi(df, window=self.rsi_period)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = calculate_bollinger_bands(
            df, window=self.bb_period, num_std=self.bb_std
        )
        
        # ADX and DI
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df, window=14)
        
        # ATR
        df['atr'] = calculate_atr(df, window=14)
        
        # Supertrend
        supertrend_data = calculate_supertrend(
            df, period=self.supertrend_period, multiplier=self.supertrend_multiplier
        )
        df['supertrend'] = supertrend_data['supertrend']
        df['supertrend_direction'] = supertrend_data['direction']
        
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
        
        # Generate signals (predictions not needed for rule-based)
        signals = self.generate_signals(features, None, dates)
        
        # Apply risk management
        signals = self.apply_risk_management(signals, test_data, features)
        
        # Calculate performance metrics
        metrics = self._calculate_backtest_metrics(signals, test_data)
        
        # Return in expected format
        return {
            'performance': metrics,
            'trades': signals[signals['signal'] != 0],
            'equity_curve': pd.DataFrame({'equity': (1 + metrics.get('total_return', 0))}, index=[test_data.index[-1]])
        }
    
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
        # Simple metrics calculation
        # In production, this would use the BacktestEngine
        
        # Align signals with prices
        aligned_prices = prices.loc[signals.index]
        
        # Calculate returns
        position = signals['signal'].shift(1).fillna(0)
        returns = position * aligned_prices['close'].pct_change()
        
        # Determine annualization factor based on timeframe
        timeframe = self.config.get('primary_timeframe', '1h')
        if timeframe in ['5min', '5T', '5m']:
            # 5-minute bars: 78 bars per day * 252 trading days
            annualization_factor = np.sqrt(78 * 252)
        elif timeframe in ['1h', '60min', '60T']:
            # Hourly bars: 6.5 hours per day * 252 trading days
            annualization_factor = np.sqrt(6.5 * 252)
        else:
            # Default to daily
            annualization_factor = np.sqrt(252)
        
        # Calculate metrics
        total_return = (1 + returns).prod() - 1
        sharpe_ratio = returns.mean() / returns.std() * annualization_factor if returns.std() > 0 else 0
        max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
        
        num_trades = (signals['signal'] != signals['signal'].shift(1)).sum()
        win_rate = (returns > 0).sum() / (returns != 0).sum() if (returns != 0).sum() > 0 else 0
        
        # Calculate CAGR
        if timeframe in ['5min', '5T', '5m']:
            periods_per_year = 78 * 252  # 19,656 bars per year
        elif timeframe in ['1h', '60min', '60T']:
            periods_per_year = 6.5 * 252  # 1,638 hours per year
        else:
            periods_per_year = 252
        
        years = max(len(returns) / periods_per_year, 0.01)
        ann_return = (1 + total_return) ** (1 / years) - 1
        
        return {
            'total_return': total_return,
            'ann_return': ann_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'strategy': 'BB-RSI-ADX'
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
        
        # Add BB-RSI-ADX specific metrics
        metrics.update({
            'bb_period': self.bb_period,
            'rsi_period': self.rsi_period,
            'rsi_overbought': self.rsi_overbought,
            'rsi_oversold': self.rsi_oversold,
            'supertrend_period': self.supertrend_period,
            'supertrend_multiplier': self.supertrend_multiplier,
            'adx_primary_threshold': self.adx_primary_threshold,
            'adx_secondary_threshold': self.adx_secondary_threshold,
            'atr_stop_multiplier': self.atr_stop_multiplier,
            'primary_timeframe': self.get_required_timeframes()[0] if self.get_required_timeframes() else '1h',
            'uses_dual_timeframe': True,
        })
        
        return metrics