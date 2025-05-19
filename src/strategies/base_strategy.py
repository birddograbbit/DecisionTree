"""
Base strategy interface for trading strategies.
"""

from abc import ABC, abstractmethod
import numpy as np
from src.utils.adaptive_thresholds import are_adaptive_thresholds_needed, calculate_adaptive_thresholds

# Global constants for signal thresholds
BUY_THRESHOLD = 0.65
SELL_THRESHOLD = 0.35

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    This interface ensures all strategies implement a common API,
    allowing them to be used interchangeably throughout the system.
    """

    def __init__(self):
        """Initialize the strategy with default values."""
        self.use_adaptive_thresholds = False
        self.buy_threshold = BUY_THRESHOLD
        self.sell_threshold = SELL_THRESHOLD
        self.config = None
        
    def _adjust_thresholds_if_needed(self, predictions):
        """
        Check if adaptive thresholds are needed and adjust accordingly.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Model prediction probabilities
        
        Returns:
        --------
        tuple
            (adjusted_buy_threshold, adjusted_sell_threshold)
        """
        # Check if custom thresholds are specified in config
        if self.config and 'buy_threshold' in self.config and 'sell_threshold' in self.config:
            return self.config['buy_threshold'], self.config['sell_threshold']
            
        # Check if we need adaptive thresholds
        if are_adaptive_thresholds_needed(predictions):
            self.use_adaptive_thresholds = True
            buy_percentile = self.config.get('buy_percentile', 80) if self.config else 80
            sell_percentile = self.config.get('sell_percentile', 20) if self.config else 20
            
            # Calculate adaptive thresholds
            adaptive_buy, adaptive_sell = calculate_adaptive_thresholds(
                predictions, 
                buy_percentile=buy_percentile,
                sell_percentile=sell_percentile
            )
            
            # Log the adaptive thresholds
            print(f"Using adaptive thresholds: buy={adaptive_buy:.4f}, sell={adaptive_sell:.4f}")
            print(f"Prediction range: min={np.min(predictions):.4f}, max={np.max(predictions):.4f}")
            
            return adaptive_buy, adaptive_sell
        
        # Use standard thresholds
        return self.buy_threshold, self.sell_threshold

    def _prob_to_signal(self, prob, thresholds=None):
        """
        Convert probability to trading signal.
        
        Parameters:
        -----------
        prob : float
            Model prediction probability
        thresholds : tuple, optional
            Custom (buy_threshold, sell_threshold) to use
            
        Returns:
        --------
        int
            Trading signal (1 for buy, -1 for sell, 0 for hold)
        """
        buy_threshold, sell_threshold = thresholds if thresholds else (self.buy_threshold, self.sell_threshold)
        
        if prob >= buy_threshold:
            return 1
        elif prob <= sell_threshold:
            return -1
        return 0

    def _size_from_prob(self, prob):
        """
        Calculate position size from probability.
        
        Parameters:
        -----------
        prob : float
            Model prediction probability
            
        Returns:
        --------
        float
            Position size (0-1 scale)
        """
        # Use square-root weighting for smoother sizing
        return (abs(prob - 0.5) * 2) ** 0.5

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
        
        # Check for custom thresholds in config
        if config and 'buy_threshold' in config:
            self.buy_threshold = config['buy_threshold']
        if config and 'sell_threshold' in config:
            self.sell_threshold = config['sell_threshold']

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
        # Check if we need adaptive thresholds before generating signals
        adaptive_buy, adaptive_sell = self._adjust_thresholds_if_needed(predictions)
        
        # Store adaptive thresholds for later use
        self.adaptive_buy_threshold = adaptive_buy
        self.adaptive_sell_threshold = adaptive_sell

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
