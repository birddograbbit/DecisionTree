"""
TEMA Trend Following Strategy Adapter.

This adapter implements the TEMA trend following strategy that:
- Uses TEMA crossovers (fast/slow) for trend detection
- Filters with ADX for trend strength and CMO for momentum
- Supports dual timeframe confirmation
- Uses ATR-based entry, stop loss, and take profit
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from src.strategies.base_strategy import BaseStrategy
from src.strategies.order_management import OrderManagementSystem, OrderType
from src.features.indicators import (
    calculate_tema, calculate_atr, calculate_adx, calculate_cmo
)
from src.features.multi_timeframe_features import MultiTimeframeAggregator
import config

# Configure logging
logger = logging.getLogger(__name__)


class TEMAAdapter(BaseStrategy):
    """
    TEMA trend following strategy adapter.
    
    This strategy uses Triple Exponential Moving Average crossovers
    to identify trends and enters with limit orders.
    """
    
    def __init__(self):
        """Initialize the TEMA adapter."""
        super().__init__()
        
        # Strategy parameters
        self.tema_primary_fast = 10
        self.tema_primary_slow = 80
        self.tema_secondary_fast = 20
        self.tema_secondary_slow = 70
        self.adx_threshold = 40
        self.cmo_long_threshold = 40
        self.cmo_short_threshold = -40
        self.atr_entry_offset = 1
        self.atr_stop_loss = 3
        self.atr_take_profit = 3
        self.bars_to_enter = 6  # Order persistence
        
        # Dual timeframe mode
        self.use_dual_timeframe = True
        
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
        self.tema_primary_fast = config.get('tema_primary_fast', self.tema_primary_fast)
        self.tema_primary_slow = config.get('tema_primary_slow', self.tema_primary_slow)
        self.tema_secondary_fast = config.get('tema_secondary_fast', self.tema_secondary_fast)
        self.tema_secondary_slow = config.get('tema_secondary_slow', self.tema_secondary_slow)
        self.adx_threshold = config.get('adx_threshold', self.adx_threshold)
        self.cmo_long_threshold = config.get('cmo_long_threshold', self.cmo_long_threshold)
        self.cmo_short_threshold = config.get('cmo_short_threshold', self.cmo_short_threshold)
        self.atr_entry_offset = config.get('atr_entry_offset', self.atr_entry_offset)
        self.atr_stop_loss = config.get('atr_stop_loss', self.atr_stop_loss)
        self.atr_take_profit = config.get('atr_take_profit', self.atr_take_profit)
        self.bars_to_enter = config.get('bars_to_enter', self.bars_to_enter)
        self.use_dual_timeframe = config.get('use_dual_timeframe', self.use_dual_timeframe)
        
        # Initialize order manager
        allow_same_bar_exit = config.get('allow_same_bar_exit', False)
        self.order_manager = OrderManagementSystem(allow_same_bar_exit)
        
        logger.info("TEMA adapter initialized with config")
        
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
            'tema_fast', 'tema_slow', 'adx', 'cmo', 'atr'
        ]
    
    def get_required_timeframes(self) -> List[str]:
        """
        Get list of required timeframes.
        
        Returns:
        --------
        List[str]
            List of required timeframes
        """
        primary = '1h'  # Default primary timeframe
        if hasattr(self, 'config') and self.config:
            primary = self.config.get('primary_timeframe', '1h')
        if self.use_dual_timeframe:
            return [primary, '4h']
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
            'order_type': 'limit',
            'limit_offset_atr': self.atr_entry_offset,
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
        
        # Create feature matrix
        features = pd.DataFrame(index=data.index)
        
        # Price features
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # TEMA features
        features['tema_fast'] = data['tema_fast']
        features['tema_slow'] = data['tema_slow']
        features['tema_diff'] = data['tema_fast'] - data['tema_slow']
        features['tema_diff_pct'] = features['tema_diff'] / data['close']
        
        # Momentum and trend features
        features['adx'] = data['adx']
        features['cmo'] = data['cmo']
        features['atr_normalized'] = data['atr'] / data['close']
        
        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Price position relative to TEMA
        features['price_to_tema_fast'] = data['close'] / data['tema_fast'] - 1
        features['price_to_tema_slow'] = data['close'] / data['tema_slow'] - 1
        
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
        Generate trading signals based on TEMA logic.
        
        This overrides the model predictions with rule-based logic.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix (includes multi-timeframe features if enabled)
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
        
        # Extract timeframe features
        primary_tf = self.get_required_timeframes()[0]
        
        # Get primary timeframe indicators
        tema_fast = features.get(f'tema_fast_{primary_tf}', features.get('tema_fast', pd.Series(index=dates)))
        tema_slow = features.get(f'tema_slow_{primary_tf}', features.get('tema_slow', pd.Series(index=dates)))
        adx = features.get(f'adx_{primary_tf}', features.get('adx', pd.Series(index=dates)))
        cmo = features.get(f'cmo_{primary_tf}', features.get('cmo', pd.Series(index=dates)))
        atr = features.get(f'atr_{primary_tf}', features.get('atr_normalized', pd.Series(index=dates)))
        close_price = features.get(f'close_{primary_tf}', features.get('close', pd.Series(index=dates)))
        
        # Get secondary timeframe indicators if dual timeframe mode
        if self.use_dual_timeframe and len(self.get_required_timeframes()) > 1:
            secondary_tf = self.get_required_timeframes()[1]
            tema_fast_sec = features.get(f'tema_fast_{secondary_tf}', tema_fast)
            tema_slow_sec = features.get(f'tema_slow_{secondary_tf}', tema_slow)
        else:
            tema_fast_sec = tema_fast
            tema_slow_sec = tema_slow
        
        # Generate signals based on TEMA rules
        for i in range(len(signals)):
            # Skip if missing data
            if pd.isna(tema_fast.iloc[i]) or pd.isna(adx.iloc[i]) or pd.isna(cmo.iloc[i]):
                continue
            
            # Primary timeframe trend
            primary_uptrend = tema_fast.iloc[i] > tema_slow.iloc[i]
            primary_downtrend = tema_fast.iloc[i] < tema_slow.iloc[i]
            
            # Secondary timeframe trend (if dual timeframe)
            if self.use_dual_timeframe:
                secondary_uptrend = tema_fast_sec.iloc[i] > tema_slow_sec.iloc[i]
                secondary_downtrend = tema_fast_sec.iloc[i] < tema_slow_sec.iloc[i]
            else:
                secondary_uptrend = True
                secondary_downtrend = True
            
            # Long signal conditions
            long_conditions = (
                primary_uptrend and
                secondary_uptrend and
                adx.iloc[i] > self.adx_threshold and
                cmo.iloc[i] > self.cmo_long_threshold
            )
            
            # Short signal conditions
            short_conditions = (
                primary_downtrend and
                secondary_downtrend and
                adx.iloc[i] > self.adx_threshold and
                cmo.iloc[i] < self.cmo_short_threshold
            )
            
            # Calculate ATR in price terms
            atr_price = atr.iloc[i] * close_price.iloc[i]
            
            if long_conditions:
                signals.iloc[i, signals.columns.get_loc('signal')] = 1
                # Entry at 1 ATR below previous close
                signals.iloc[i, signals.columns.get_loc('entry_price')] = (
                    close_price.iloc[i] - self.atr_entry_offset * atr_price
                )
                # Stop loss at 3 ATR below entry
                signals.iloc[i, signals.columns.get_loc('stop_loss')] = (
                    signals.iloc[i, signals.columns.get_loc('entry_price')] - 
                    self.atr_stop_loss * atr_price
                )
                # Take profit at 3 ATR above entry
                signals.iloc[i, signals.columns.get_loc('take_profit')] = (
                    signals.iloc[i, signals.columns.get_loc('entry_price')] + 
                    self.atr_take_profit * atr_price
                )
                
            elif short_conditions:
                signals.iloc[i, signals.columns.get_loc('signal')] = -1
                # Entry at 1 ATR above previous close
                signals.iloc[i, signals.columns.get_loc('entry_price')] = (
                    close_price.iloc[i] + self.atr_entry_offset * atr_price
                )
                # Stop loss at 3 ATR above entry
                signals.iloc[i, signals.columns.get_loc('stop_loss')] = (
                    signals.iloc[i, signals.columns.get_loc('entry_price')] + 
                    self.atr_stop_loss * atr_price
                )
                # Take profit at 3 ATR below entry
                signals.iloc[i, signals.columns.get_loc('take_profit')] = (
                    signals.iloc[i, signals.columns.get_loc('entry_price')] - 
                    self.atr_take_profit * atr_price
                )
        
        # Add position sizing
        signals['position_size'] = np.where(
            signals['signal'] != 0,
            self.config.get('position_size', 0.1),
            0
        )
        
        logger.info(f"Generated {(signals['signal'] != 0).sum()} TEMA signals")
        
        return signals
    
    def apply_risk_management(self, signals: pd.DataFrame, prices: pd.DataFrame,
                            features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply risk management rules.
        
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
        # The ATR-based stops are already set in generate_signals
        # Additional risk management could include:
        # - Time-based exits
        # - Volatility adjustments
        # - Correlation filters
        
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
        
        # TEMA indicators
        df['tema_fast'] = calculate_tema(df, period=self.tema_primary_fast)
        df['tema_slow'] = calculate_tema(df, period=self.tema_primary_slow)
        
        # ADX
        df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df, window=14)
        
        # CMO
        df['cmo'] = calculate_cmo(df, period=14)
        
        # ATR
        df['atr'] = calculate_atr(df, window=14)
        
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
        # Simple metrics calculation
        # In production, this would use the BacktestEngine
        
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
        timeframe = self.config.get('primary_timeframe', '1h')
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
        
        return {
            'total_return': total_return,
            'ann_return': ann_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'strategy': 'TEMA'
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
        
        # Add TEMA specific metrics
        metrics.update({
            'tema_primary_fast': self.tema_primary_fast,
            'tema_primary_slow': self.tema_primary_slow,
            'tema_secondary_fast': self.tema_secondary_fast,
            'tema_secondary_slow': self.tema_secondary_slow,
            'adx_threshold': self.adx_threshold,
            'cmo_long_threshold': self.cmo_long_threshold,
            'cmo_short_threshold': self.cmo_short_threshold,
            'atr_entry_offset': self.atr_entry_offset,
            'atr_stop_loss': self.atr_stop_loss,
            'atr_take_profit': self.atr_take_profit,
            'bars_to_enter': self.bars_to_enter,
            'use_dual_timeframe': self.use_dual_timeframe,
            'primary_timeframe': self.get_required_timeframes()[0] if self.get_required_timeframes() else '1h',
        })
        
        return metrics