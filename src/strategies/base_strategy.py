"""
Base strategy interface for trading strategies.
"""

from abc import ABC, abstractmethod
import numpy as np
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
        print("Warning: _adjust_thresholds_if_needed is deprecated. Use get_thresholds() instead.")
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
    def backtest(self, data, train_data=None, test_data=None):
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
            
        Returns:
        --------
        dict
            Backtest results
        """
        pass
