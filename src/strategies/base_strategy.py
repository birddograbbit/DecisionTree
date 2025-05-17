"""
Base strategy interface for trading strategies.
"""

from abc import ABC, abstractmethod

# Global constants for signal thresholds
BUY_THRESHOLD = 0.65
SELL_THRESHOLD = 0.35

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    
    This interface ensures all strategies implement a common API,
    allowing them to be used interchangeably throughout the system.
    """

    def _prob_to_signal(self, prob):
        """
        Convert probability to trading signal.
        
        Parameters:
        -----------
        prob : float
            Model prediction probability
            
        Returns:
        --------
        int
            Trading signal (1 for buy, -1 for sell, 0 for hold)
        """
        if prob >= BUY_THRESHOLD:
            return 1
        elif prob <= SELL_THRESHOLD:
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
        pass

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
        pass
    
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