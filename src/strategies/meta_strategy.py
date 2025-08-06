# src/strategies/meta_strategy.py

"""
Meta-strategy that dynamically selects between multiple trading strategies.

This strategy monitors the performance of multiple sub-strategies and 
dynamically switches between them based on recent performance or market regime.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional
from .base_strategy import BaseStrategy
from .strategy_registry import StrategyRegistry
from ..features.regime_detection import RegimeDetector

logger = logging.getLogger(__name__)


class MetaStrategy(BaseStrategy):
    """
    Meta-strategy that dynamically selects between multiple strategies.
    
    This strategy can operate in two modes:
    1. Performance-based: Selects the best performing strategy
    2. Regime-based: Selects strategy based on detected market regime
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the meta-strategy.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration with keys:
            - strategies: List of strategy names to use
            - selection_method: 'performance' or 'regime'
            - performance_window: Bars to track performance (default: 100)
            - switch_cooldown: Minimum bars between switches (default: 20)
            - regime_map: Dict mapping regimes to strategies (for regime mode)
        """
        super().__init__()
        self.config = config or {}
        
        # Strategy selection parameters
        self.selection_method = self.config.get('selection_method', 'performance')
        self.performance_window = self.config.get('performance_window', 100)
        self.switch_cooldown = self.config.get('switch_cooldown', 20)
        
        # Initialize strategy registry
        self.registry = StrategyRegistry()
        
        # Load configured strategies
        self.available_strategies = {}
        strategy_names = self.config.get('strategies', ['quod', 'tema', 'bb_rsi_adx'])
        
        for name in strategy_names:
            try:
                strategy = self.registry.create_strategy(name, self.config.get(f'{name}_config', {}))
                self.available_strategies[name] = strategy
                logger.info(f"Loaded strategy: {name}")
            except Exception as e:
                logger.error(f"Failed to load strategy {name}: {e}")
        
        if not self.available_strategies:
            raise ValueError("No strategies loaded successfully")
        
        # Performance tracking
        self.strategy_performance = {name: [] for name in self.available_strategies}
        self.current_strategy_name = list(self.available_strategies.keys())[0]
        self.current_strategy = self.available_strategies[self.current_strategy_name]
        self.bars_since_switch = 0
        
        # Regime detection (for regime-based selection)
        if self.selection_method == 'regime':
            self.regime_detector = RegimeDetector(
                method=self.config.get('regime_method', 'trend_volatility')
            )
            self.regime_map = self.config.get('regime_map', self._get_default_regime_map())
        
        # Initialize tracking
        self.selection_history = []
        
    def _get_default_regime_map(self) -> Dict[str, str]:
        """Get default mapping of regimes to strategies."""
        return {
            'strong_uptrend': 'tema',      # Trend following
            'uptrend': 'tema',
            'weak_uptrend': 'bb_rsi_adx',  # Momentum
            'volatile_neutral': 'bb_rsi_adx',
            'neutral': 'quod',             # Mean reversion
            'low_vol_neutral': 'quod',
            'weak_downtrend': 'bb_rsi_adx',
            'downtrend': 'tema',
            'strong_downtrend': 'tema'
        }
    
    def initialize(self, config: dict):
        """Initialize the strategy with configuration."""
        # Initialize all sub-strategies
        for name, strategy in self.available_strategies.items():
            strategy.initialize(config.get(f'{name}_config', {}))
        
        # Store initialization config
        self.config.update(config)
        
    def train(self, train_data: pd.DataFrame):
        """
        Train all sub-strategies.
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        """
        # Train each strategy (if it has a train method)
        for name, strategy in self.available_strategies.items():
            if hasattr(strategy, 'train') and callable(getattr(strategy, 'train')):
                logger.info(f"Training strategy: {name}")
                strategy.train(train_data)
            else:
                logger.info(f"Strategy {name} is rule-based, no training needed")
        
        # Detect regimes if using regime-based selection
        if self.selection_method == 'regime' and hasattr(self, 'regime_detector'):
            self.regime_detector.detect_regime(train_data)
    
    def generate_signals(self, features: pd.DataFrame, predictions: np.ndarray, 
                        dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Generate trading signals using the selected strategy.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix
        predictions : np.ndarray
            Model predictions (may be ignored)
        dates : pd.DatetimeIndex
            Dates for signals
            
        Returns:
        --------
        pd.DataFrame
            Trading signals
        """
        # Select appropriate strategy
        selected_strategy_name = self._select_strategy(features, dates)
        
        # Switch strategy if needed
        if selected_strategy_name != self.current_strategy_name:
            if self.bars_since_switch >= self.switch_cooldown:
                logger.info(f"Strategy switch triggered after {self.bars_since_switch} bars")
                logger.info(f"Switching from {self.current_strategy_name} to {selected_strategy_name}")
                self.current_strategy_name = selected_strategy_name
                self.current_strategy = self.available_strategies[selected_strategy_name]
                self.bars_since_switch = 0
                
                # Record switch
                self.selection_history.append({
                    'date': dates[-1] if len(dates) > 0 else pd.Timestamp.now(),
                    'strategy': selected_strategy_name,
                    'reason': self.selection_method
                })
            else:
                logger.debug(f"Strategy switch to {selected_strategy_name} blocked by cooldown "
                           f"({self.bars_since_switch}/{self.switch_cooldown} bars)")
        
        # Generate signals using current strategy
        signals = self.current_strategy.generate_signals(features, predictions, dates)
        
        # Add meta-strategy information
        signals['selected_strategy'] = self.current_strategy_name
        signals['bars_since_switch'] = self.bars_since_switch
        
        # Update counters
        self.bars_since_switch += len(dates)
        
        return signals
    
    def _select_strategy(self, features: pd.DataFrame, dates: pd.DatetimeIndex) -> str:
        """
        Select the appropriate strategy based on selection method.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Current features
        dates : pd.DatetimeIndex
            Current dates
            
        Returns:
        --------
        str
            Name of selected strategy
        """
        logger.debug(f"Selecting strategy using method: {self.selection_method}")
        
        if self.selection_method == 'performance':
            return self._select_by_performance()
        elif self.selection_method == 'regime':
            return self._select_by_regime(features, dates)
        else:
            # Default to current strategy
            return self.current_strategy_name
    
    def _select_by_performance(self) -> str:
        """
        Select strategy based on recent performance.
        
        Returns:
        --------
        str
            Name of best performing strategy
        """
        # Get performance stats from registry
        best_sharpe = -np.inf
        best_strategy = self.current_strategy_name
        strategy_stats = {}
        
        for name in self.available_strategies:
            stats = self.registry.get_performance_stats(name, self.performance_window)
            
            # Skip if insufficient data
            if stats.get('insufficient_data', True):
                logger.debug(f"Strategy {name}: Insufficient data for performance evaluation")
                continue
            
            sharpe = stats.get('sharpe_ratio', 0)
            strategy_stats[name] = sharpe
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_strategy = name
        
        # Log all strategy performances
        if strategy_stats:
            logger.info(f"Strategy performance comparison: {strategy_stats}")
            logger.info(f"Selected strategy: {best_strategy} with Sharpe: {best_sharpe:.2f}")
        else:
            logger.warning("No strategies have sufficient performance data")
        
        return best_strategy
    
    def _select_by_regime(self, features: pd.DataFrame, dates: pd.DatetimeIndex) -> str:
        """
        Select strategy based on detected market regime.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Current features
        dates : pd.DatetimeIndex
            Current dates
            
        Returns:
        --------
        str
            Name of selected strategy for current regime
        """
        if not hasattr(self, 'regime_detector'):
            return self.current_strategy_name
        
        try:
            # Get current regime
            current_regime = self.regime_detector.get_current_regime()
            regime_label = current_regime.get('regime_label', 'neutral')
            
            # Map regime to strategy
            selected_strategy = self.regime_map.get(regime_label, self.current_strategy_name)
            
            # Verify strategy exists
            if selected_strategy in self.available_strategies:
                return selected_strategy
            else:
                logger.warning(f"Strategy {selected_strategy} not available for regime {regime_label}")
                return self.current_strategy_name
                
        except Exception as e:
            logger.error(f"Error in regime-based selection: {e}")
            return self.current_strategy_name
    
    def update_performance(self, strategy_name: str, returns: List[float]):
        """
        Update performance tracking for a strategy.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
        returns : List[float]
            Recent returns to add
        """
        if strategy_name in self.strategy_performance:
            self.strategy_performance[strategy_name].extend(returns)
            
            # Keep only recent history
            max_history = self.performance_window * 2
            if len(self.strategy_performance[strategy_name]) > max_history:
                self.strategy_performance[strategy_name] = \
                    self.strategy_performance[strategy_name][-max_history:]
    
    def _run_strategy_backtest(self, strategy_name: str, strategy: BaseStrategy, 
                               data: pd.DataFrame, timeframe: str = 'daily') -> Dict[str, float]:
        """
        Run a backtest for a single strategy and return performance metrics.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
        strategy : BaseStrategy
            Strategy instance
        data : pd.DataFrame
            Price data for backtesting
        timeframe : str
            Trading timeframe
            
        Returns:
        --------
        Dict[str, float]
            Performance metrics including returns and sharpe ratio
        """
        try:
            # Generate features and signals
            X, y, dates = strategy.generate_features(data)
            if len(X) == 0:
                return {'returns': 0, 'sharpe_ratio': 0, 'num_trades': 0}
            
            predictions = np.zeros(len(X))  # Dummy predictions for momentum strategies
            signals = strategy.generate_signals(X, predictions, dates)
            
            if len(signals) == 0:
                return {'returns': 0, 'sharpe_ratio': 0, 'num_trades': 0}
            
            # Ensure signals doesn't have date as both index and column
            if 'date' in signals.columns and signals.index.name == 'date':
                signals = signals.reset_index(drop=True)
            
            # Run simplified backtest to get performance
            from src.backtesting.engine import BacktestEngine
            backtest_engine = BacktestEngine(
                initial_capital=100000,
                commission=0.001,
                slippage=0.001
            )
            
            # Make sure data doesn't have date as both index and column
            data_for_backtest = data.copy()
            if 'date' in data_for_backtest.columns and data_for_backtest.index.name == 'date':
                data_for_backtest = data_for_backtest.reset_index(drop=True)
            
            results = backtest_engine.run_backtest(
                signals, 
                {self.config.get('symbol', 'SPY'): data_for_backtest}, 
                timeframe
            )
            
            performance = results.get('performance', {})
            return {
                'returns': performance.get('total_return', 0),
                'sharpe_ratio': performance.get('sharpe_ratio', 0),
                'num_trades': performance.get('num_trades', 0)
            }
            
        except Exception as e:
            logger.error(f"Error running backtest for {strategy_name}: {e}")
            return {'returns': 0, 'sharpe_ratio': 0, 'num_trades': 0}
    
    def get_metrics(self) -> dict:
        """
        Get strategy metrics including selection history.
        
        Returns:
        --------
        dict
            Strategy metrics
        """
        metrics = super().get_metrics()
        
        # Add meta-strategy specific metrics
        metrics['meta_strategy'] = {
            'current_strategy': self.current_strategy_name,
            'available_strategies': list(self.available_strategies.keys()),
            'selection_method': self.selection_method,
            'switches_count': len(self.selection_history),
            'bars_since_switch': self.bars_since_switch
        }
        
        # Add selection history summary
        if self.selection_history:
            strategy_usage = {}
            for entry in self.selection_history:
                strategy = entry['strategy']
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
            
            metrics['meta_strategy']['strategy_usage'] = strategy_usage
        
        return metrics
    
    def generate_features(self, data: pd.DataFrame) -> tuple:
        """
        Generate features for the strategy.
        
        This is delegated to the current strategy.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
            
        Returns:
        --------
        tuple
            (X, y, dates)
        """
        return self.current_strategy.generate_features(data)
    
    def backtest(self, data: pd.DataFrame, train_data: pd.DataFrame = None,
                 test_data: pd.DataFrame = None, timeframe: str = 'daily') -> dict:
        """
        Run backtest for the meta-strategy with performance tracking.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Full price data
        train_data : pd.DataFrame, optional
            Training data (if None, uses 70% of data)
        test_data : pd.DataFrame, optional
            Testing data (if None, uses 30% of data)
        timeframe : str, default='daily'
            Trading timeframe
            
        Returns:
        --------
        dict
            Backtest results
        """
        # Split data if not provided
        if train_data is None or test_data is None:
            train_size = int(len(data) * 0.7)
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]
        
        # Train all strategies
        self.train(train_data)
        
        # Initialize performance tracking
        logger.info("Initializing strategy performance with warm-up period")
        
        # Use sufficient data for warm-up to fill performance window
        warmup_size = min(self.performance_window + 100, int(len(test_data) * 0.3))
        warmup_data = test_data.iloc[:warmup_size]
        remaining_test_data = test_data.iloc[warmup_size:]
        
        # Run warm-up backtests for all strategies
        logger.info(f"Running warm-up period with {warmup_size} bars")
        for name, strategy in self.available_strategies.items():
            metrics = self._run_strategy_backtest(name, strategy, warmup_data, timeframe)
            
            # Initialize performance tracking with warm-up results
            # Always track performance, even if sharpe is 0 or negative
            # Convert total return to daily returns approximation
            if warmup_size > 0:
                daily_return = metrics['returns'] / 100 / warmup_size  # Convert percentage to decimal daily
                warmup_returns = [daily_return] * warmup_size
                self.registry.track_performance(name, warmup_returns)
                logger.info(f"Warm-up {name}: Sharpe={metrics['sharpe_ratio']:.2f}, "
                           f"Return={metrics['returns']:.2f}%, Trades={metrics['num_trades']}")
        
        # Now run the actual backtest on remaining data
        from src.backtesting.engine import BacktestEngine
        
        backtest_engine = BacktestEngine(
            initial_capital=self.config.get('initial_capital', 100000),
            commission=self.config.get('commission', 0.001),
            slippage=self.config.get('slippage', 0.001)
        )
        
        # Generate signals for the remaining test data
        all_signals = []
        
        # Process entire remaining data as one chunk for simplicity
        if len(remaining_test_data) > 10:
            X, y, dates = self.generate_features(remaining_test_data)
            
            if len(X) > 0:
                predictions = np.zeros(len(X))
                signals = self.generate_signals(X, predictions, dates)
                all_signals.append(signals)
        
        # Combine all signals
        if all_signals:
            combined_signals = pd.concat(all_signals, ignore_index=True)
            
            # Run backtest with combined signals
            test_data_dict = {self.config.get('symbol', 'SPY'): test_data}
            results = backtest_engine.run_backtest(combined_signals, test_data_dict, timeframe)
        else:
            # Fallback to simple backtest
            logger.warning("No signals generated, falling back to simple backtest")
            
            if hasattr(self.current_strategy, 'backtest'):
                results = self.current_strategy.backtest(data, train_data, test_data, timeframe)
            else:
                # Empty results
                results = {
                    'trades': pd.DataFrame(),
                    'equity_curve': pd.DataFrame(),
                    'performance': {}
                }
        
        # Add meta-strategy specific metrics
        results['meta_strategy_metrics'] = self.get_metrics()
        results['performance']['strategy_used'] = 'meta_strategy'
        results['performance']['switches'] = len(self.selection_history)
        
        return results
    
    def _generate_meta_signals(self, strategy_signals: Dict[str, pd.DataFrame], 
                              test_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Generate meta-strategy signals by selecting best strategy.
        
        Parameters:
        -----------
        strategy_signals : Dict[str, pd.DataFrame]
            Signals from each strategy
        test_data : pd.DataFrame
            Test price data
        timeframe : str
            Trading timeframe
            
        Returns:
        --------
        pd.DataFrame
            Meta-strategy signals
        """
        # Start with first strategy's signals as template
        first_strategy = list(strategy_signals.keys())[0]
        meta_signals = strategy_signals[first_strategy].copy()
        
        # For simplified version, use first strategy throughout
        # In future, implement performance tracking and switching
        meta_signals['selected_strategy'] = self.current_strategy_name
        
        return meta_signals