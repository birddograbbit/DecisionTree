"""
Utility module for adaptive thresholds based on model probability distributions.
"""

import numpy as np

def analyze_probability_distribution(predictions):
    """
    Analyze the probability distribution of model predictions.
    
    Parameters:
    -----------
    predictions : np.ndarray
        Model prediction probabilities
        
    Returns:
    --------
    dict
        Statistics about the probability distribution
    """
    stats = {
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'mean': float(np.mean(predictions)),
        'median': float(np.median(predictions)),
        'std': float(np.std(predictions)),
        'p10': float(np.percentile(predictions, 10)),
        'p25': float(np.percentile(predictions, 25)),
        'p75': float(np.percentile(predictions, 75)),
        'p90': float(np.percentile(predictions, 90)),
        'range': float(np.max(predictions) - np.min(predictions))
    }
    
    return stats

def calculate_adaptive_thresholds(predictions, buy_percentile=80, sell_percentile=20):
    """
    Calculate adaptive buy and sell thresholds based on prediction distribution.
    
    Parameters:
    -----------
    predictions : np.ndarray
        Model prediction probabilities
    buy_percentile : int, default=80
        Percentile for buy threshold (0-100)
    sell_percentile : int, default=20
        Percentile for sell threshold (0-100)
        
    Returns:
    --------
    tuple
        (buy_threshold, sell_threshold)
    """
    stats = analyze_probability_distribution(predictions)
    
    # If the range is very small, we need to adapt more aggressively
    # Standard thresholds of 0.65 and 0.35 assume a range of 1.0
    range_ratio = min(1.0, stats['range'] / 0.3)
    
    if range_ratio < 0.5:
        # Very compressed range, use percentile-based thresholds
        buy_threshold = float(np.percentile(predictions, buy_percentile))
        sell_threshold = float(np.percentile(predictions, sell_percentile))
    else:
        # Use a hybrid approach, scaling standard thresholds based on the distribution
        midpoint = stats['median']
        half_range = stats['range'] / 2
        
        # Scale thresholds around the median
        buy_threshold = min(midpoint + (half_range * 0.7), stats['max'] * 0.95)
        sell_threshold = max(midpoint - (half_range * 0.7), stats['min'] * 1.05)
    
    return buy_threshold, sell_threshold

def are_adaptive_thresholds_needed(predictions, min_range=0.3):
    """
    Determine if adaptive thresholds are needed based on prediction range.
    
    Parameters:
    -----------
    predictions : np.ndarray
        Model prediction probabilities
    min_range : float, default=0.3
        Minimum range required to use standard thresholds
        
    Returns:
    --------
    bool
        True if adaptive thresholds are recommended
    """
    stats = analyze_probability_distribution(predictions)
    
    return stats['range'] < min_range
