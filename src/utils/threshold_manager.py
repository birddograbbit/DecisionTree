"""
Centralized threshold management for trading signals.

This module consolidates threshold logic that was previously duplicated
across BaseStrategy and SignalEngine classes, providing a single source
of truth for threshold determination.
"""

import numpy as np
from src.utils.adaptive_thresholds import are_adaptive_thresholds_needed, calculate_adaptive_thresholds

# Global threshold constants
DEFAULT_BUY_THRESHOLD = 0.65
DEFAULT_SELL_THRESHOLD = 0.35

class ThresholdManager:
    """
    Centralized manager for determining buy and sell thresholds.
    
    This class consolidates the threshold logic that was previously duplicated
    across multiple classes, providing consistent threshold determination
    based on configuration and prediction distributions.
    """
    
    def __init__(self, config=None):
        """
        Initialize the threshold manager.
        
        Parameters:
        -----------
        config : dict, optional
            Configuration dictionary containing threshold settings
        """
        self.config = config or {}
        self.use_adaptive_thresholds = self._determine_adaptive_usage()
        self.buy_threshold = self.config.get('buy_threshold', DEFAULT_BUY_THRESHOLD)
        self.sell_threshold = self.config.get('sell_threshold', DEFAULT_SELL_THRESHOLD)
        
    def _determine_adaptive_usage(self):
        """
        Determine whether to use adaptive thresholds based on configuration.
        
        Returns:
        --------
        str
            One of 'auto', 'always', 'never'
        """
        adaptive_setting = self.config.get('use_adaptive_thresholds', 'auto')
        if adaptive_setting in ['auto', 'always', 'never']:
            return adaptive_setting
        else:
            # Default fallback
            return 'auto'
    
    def get_thresholds(self, predictions=None):
        """
        Get appropriate buy and sell thresholds based on configuration and predictions.
        
        Parameters:
        -----------
        predictions : np.ndarray, optional
            Model prediction probabilities (required for adaptive thresholds)
            
        Returns:
        --------
        tuple
            (buy_threshold, sell_threshold)
        """
        # Check if custom thresholds are explicitly specified in config
        if 'buy_threshold' in self.config and 'sell_threshold' in self.config:
            return self.config['buy_threshold'], self.config['sell_threshold']
        
        # Determine if we should use adaptive thresholds
        should_use_adaptive = self._should_use_adaptive_thresholds(predictions)
        
        if should_use_adaptive and predictions is not None:
            return self._calculate_adaptive_thresholds(predictions)
        else:
            return self.buy_threshold, self.sell_threshold
    
    def _should_use_adaptive_thresholds(self, predictions):
        """
        Determine if adaptive thresholds should be used.
        
        Parameters:
        -----------
        predictions : np.ndarray or None
            Model prediction probabilities
            
        Returns:
        --------
        bool
            True if adaptive thresholds should be used
        """
        if predictions is None:
            return False
            
        if self.use_adaptive_thresholds == 'always':
            return True
        elif self.use_adaptive_thresholds == 'never':
            return False
        else:  # 'auto'
            return are_adaptive_thresholds_needed(predictions)
    
    def _calculate_adaptive_thresholds(self, predictions):
        """
        Calculate adaptive thresholds based on prediction distribution.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Model prediction probabilities
            
        Returns:
        --------
        tuple
            (adaptive_buy_threshold, adaptive_sell_threshold)
        """
        buy_percentile = self.config.get('buy_percentile', 80)
        sell_percentile = self.config.get('sell_percentile', 20)
        
        adaptive_buy, adaptive_sell = calculate_adaptive_thresholds(
            predictions, 
            buy_percentile=buy_percentile,
            sell_percentile=sell_percentile
        )
        
        # Log the adaptive thresholds for transparency
        print(f"Using adaptive thresholds: buy={adaptive_buy:.4f}, sell={adaptive_sell:.4f}")
        print(f"Prediction range: min={np.min(predictions):.4f}, max={np.max(predictions):.4f}")
        
        return adaptive_buy, adaptive_sell
    
    def prob_to_signal(self, prob, predictions=None):
        """
        Convert probability to trading signal using appropriate thresholds.
        
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
        buy_threshold, sell_threshold = self.get_thresholds(predictions)
        
        if prob >= buy_threshold:
            return 1
        elif prob <= sell_threshold:
            return -1
        return 0
    
    def get_position_size(self, prob, position_sizing='confidence'):
        """
        Calculate position size from probability.
        
        Parameters:
        -----------
        prob : float
            Model prediction probability
        position_sizing : str
            Position sizing method ('fixed' or 'confidence')
            
        Returns:
        --------
        float
            Position size (0-1 scale)
        """
        if position_sizing == 'fixed':
            # Fixed size for any non-zero signal
            signal = self.prob_to_signal(prob)
            return 1.0 if signal != 0 else 0.0
        else:  # confidence-based sizing
            # Use square-root weighting for smoother sizing
            return (abs(prob - 0.5) * 2) ** 0.5
    
    def get_configuration_summary(self):
        """
        Get a summary of current threshold configuration.
        
        Returns:
        --------
        dict
            Summary of threshold configuration
        """
        return {
            'use_adaptive_thresholds': self.use_adaptive_thresholds,
            'default_buy_threshold': self.buy_threshold,
            'default_sell_threshold': self.sell_threshold,
            'buy_percentile': self.config.get('buy_percentile', 80),
            'sell_percentile': self.config.get('sell_percentile', 20),
            'custom_buy_threshold': self.config.get('buy_threshold'),
            'custom_sell_threshold': self.config.get('sell_threshold')
        }
