"""
Base strategy interface for trading strategies.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from src.utils.threshold_manager import ThresholdManager

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    This interface ensures all strategies implement a common API,
    allowing them to be used interchangeably throughout the system.
    """

    def __init__(self):
        """Initialize the strategy with default values."""
        self.config = None
        self.threshold_manager = None
        
    def _get_threshold_manager(self):
        """
        Get the threshold manager, creating it if necessary.
        
        Returns:
        --------
        ThresholdManager
            Configured threshold manager
        """
        if self.threshold_manager is None:
            self.threshold_manager = ThresholdManager(self.config)
        return self.threshold_manager

    def _adjust_thresholds_if_needed(self, predictions):
        """
        Get appropriate thresholds based on configuration and predictions.
        
        This method is deprecated - use get_thresholds() instead.
        Kept for backward compatibility.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Model prediction probabilities
        
        Returns:
        --------
        tuple
            (adjusted_buy_threshold, adjusted_sell_threshold)
        """
        warnings.warn(
            "_adjust_thresholds_if_needed is deprecated. Use get_thresholds() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.get_thresholds(predictions)
    
    def get_thresholds(self, predictions=None):
        """
        Get appropriate buy and sell thresholds.
        
        Parameters:
        -----------
        predictions : np.ndarray, optional
            Model prediction probabilities
        
        Returns:
        --------
        tuple
            (buy_threshold, sell_threshold)
        """
        return self._get_threshold_manager().get_thresholds(predictions)

    def _prob_to_signal(self, prob, predictions=None):
        """
        Convert probability to trading signal.
        
        Parameters:
        -----------
        prob : float
            Model prediction probability
        predictions : np.ndarray, optional
            All model predictions (used for adaptive threshold calculation)
            
        Returns:
        --------
        int
            Trading signal (1 for buy, -1 for sell, 0 for hold)
        """
        return self._get_threshold_manager().prob_to_signal(prob, predictions)

    def _size_from_prob(self, prob, position_sizing='confidence'):
        """
        Calculate position size from probability.
        
        Parameters:
        -----------
        prob : float
            Model prediction probability
        position_sizing : str, optional
            Position sizing method ('fixed' or 'confidence')
            
        Returns:
        --------
        float
            Position size (0-1 scale)
        """
        return self._get_threshold_manager().get_position_size(prob, position_sizing)

    def get_threshold_configuration(self):
        """
        Get summary of current threshold configuration.
        
        Returns:
        --------
        dict
            Threshold configuration summary
        """
        return self._get_threshold_manager().get_configuration_summary()

    @abstractmethod
    def initialize(self, config):
        """
        Initialize strategy with configuration.
        
        Parameters:
        -----------
        config : dict
            Strategy configuration
        """
        self.config = config
        # Reset threshold manager to pick up new config
        self.threshold_manager = None

    @abstractmethod
    def generate_features(self, data):
        """
        Generate features for the strategy.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
            
        Returns:
        --------
        tuple
            (X, y, dates)
        """
        pass

    @abstractmethod
    def generate_signals(self, features, predictions, dates):
        """
        Generate trading signals based on features and predictions.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Feature matrix
        predictions : np.ndarray
            Model predictions
        dates : pd.DatetimeIndex
            Dates corresponding to predictions
            
        Returns:
        --------
        pd.DataFrame
            Trading signals
        """
        # Get thresholds for use in signal generation
        buy_threshold, sell_threshold = self.get_thresholds(predictions)
        
        # Store thresholds for reference
        self.current_buy_threshold = buy_threshold
        self.current_sell_threshold = sell_threshold

    @abstractmethod
    def backtest(self, data, train_data=None, test_data=None, timeframe='daily'):
        """
        Run backtest for the strategy.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
        train_data : pd.DataFrame, optional
            Training data (if None, uses a portion of data)
        test_data : pd.DataFrame, optional
            Testing data (if None, uses a portion of data)
        timeframe : str, default='daily'
            Trading timeframe ('daily', '5min', '5T')
            
        Returns:
        --------
        dict
            Backtest results
        """
        pass
    
    # New methods for strategy adapter pattern - non-abstract with default implementations
    
    def get_required_features(self) -> List[str]:
        """
        Get list of required feature names for this strategy.
        
        This allows strategies to declare what indicators they need,
        enabling the feature engineering system to prepare them.
        
        Returns:
        --------
        List[str]
            List of required feature names (e.g., ['rsi', 'macd', 'bb_upper'])
        """
        # Default: empty list (strategy will handle its own features)
        return []
    
    def get_required_timeframes(self) -> List[str]:
        """
        Get list of required timeframes for this strategy.
        
        Supports multi-timeframe analysis by declaring what timeframes
        are needed (e.g., ['1h', '4h'] for dual timeframe strategies).
        
        Returns:
        --------
        List[str]
            List of required timeframes in pandas frequency format
        """
        # Default: single daily timeframe
        return ['1D']
    
    def get_order_management_config(self) -> Dict[str, any]:
        """
        Get order management configuration for this strategy.
        
        Defines order types, persistence, and execution logic.
        
        Returns:
        --------
        Dict[str, any]
            Configuration dictionary with keys:
            - order_type: 'market' or 'limit'
            - limit_offset_atr: ATR multiplier for limit orders
            - order_persistence_bars: How many bars to keep orders active
            - allow_same_bar_exit: Whether to allow entry and exit on same bar
            - use_trailing_stop: Whether to use trailing stops
        """
        # Default: simple market orders
        return {
            'order_type': 'market',
            'limit_offset_atr': None,
            'order_persistence_bars': 1,
            'allow_same_bar_exit': True,
            'use_trailing_stop': False
        }
    
    def calculate_position_size(self, signal: int, confidence: float, 
                              capital: float, current_price: float,
                              volatility: Optional[float] = None) -> float:
        """
        Calculate position size for a signal.
        
        Enhanced position sizing that can incorporate volatility,
        Kelly criterion, or other sophisticated methods.
        
        Parameters:
        -----------
        signal : int
            Trading signal (1 for buy, -1 for sell, 0 for hold)
        confidence : float
            Model confidence/probability
        capital : float
            Available capital
        current_price : float
            Current asset price
        volatility : float, optional
            Current volatility (e.g., ATR)
            
        Returns:
        --------
        float
            Position size in units (shares/contracts)
        """
        if signal == 0:
            return 0.0
        
        # Default: Use existing logic from threshold manager
        position_pct = self._size_from_prob(confidence)
        
        # Convert percentage to units
        position_value = capital * position_pct
        position_units = position_value / current_price
        
        return position_units
    
    def apply_risk_management(self, signals: pd.DataFrame, 
                            prices: pd.DataFrame,
                            features: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply risk management rules to signals.
        
        Can modify signals based on stop loss, take profit,
        trailing stops, or other risk management rules.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            Trading signals with columns: date, signal, position_size
        prices : pd.DataFrame
            Price data (OHLCV)
        features : pd.DataFrame, optional
            Additional features that might be needed for risk management
            
        Returns:
        --------
        pd.DataFrame
            Modified signals with risk management applied
        """
        # Default: Return signals unchanged
        # Derived classes can implement stop loss, take profit, etc.
        return signals
    
    def validate_data_requirements(self, data: Dict[str, pd.DataFrame]) -> bool:
        """
        Validate that all required data is available.
        
        Parameters:
        -----------
        data : Dict[str, pd.DataFrame]
            Dictionary mapping timeframe to price data
            
        Returns:
        --------
        bool
            True if all requirements are met
        """
        required_timeframes = self.get_required_timeframes()
        
        for tf in required_timeframes:
            if tf not in data:
                return False
                
        return True
    
    def save(self, path: str) -> None:
        """
        Save strategy state to disk.
        
        For rule-based strategies, this saves the configuration.
        ML-based strategies should override to save their models.
        
        Parameters:
        -----------
        path : str
            Path to save the strategy state
        """
        import pickle
        import os
        
        # Default implementation: Save configuration
        save_data = {
            'strategy_type': self.__class__.__name__,
            'config': self.config if hasattr(self, 'config') else None,
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save as pickle file
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, path: str) -> None:
        """
        Load strategy state from disk.
        
        For rule-based strategies, this loads the configuration.
        ML-based strategies should override to load their models.
        
        Parameters:
        -----------
        path : str
            Path to load the strategy state from
        """
        import pickle
        
        # Default implementation: Load configuration
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Verify strategy type matches
        if save_data.get('strategy_type') != self.__class__.__name__:
            raise ValueError(
                f"Strategy type mismatch: expected {self.__class__.__name__}, "
                f"got {save_data.get('strategy_type')}"
            )
        
        # Load configuration
        if 'config' in save_data and save_data['config'] is not None:
            self.initialize(save_data['config'])
    
    def get_metrics(self) -> Dict[str, any]:
        """
        Get strategy performance metrics.
        
        Returns:
        --------
        Dict[str, any]
            Dictionary of performance metrics
        """
        # Default implementation: Return basic information
        metrics = {
            'strategy_type': self.__class__.__name__,
            'strategy_name': self.config.get('name', self.__class__.__name__) if hasattr(self, 'config') and self.config else self.__class__.__name__,
        }
        
        # Add configuration details if available
        if hasattr(self, 'config') and self.config:
            # Add relevant config parameters (excluding large objects)
            for key, value in self.config.items():
                if isinstance(value, (str, int, float, bool)):
                    metrics[f'config_{key}'] = value
        
        return metrics
