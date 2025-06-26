"""
Example hybrid strategy implementation combining DecisionTree and Transformer models.

This module demonstrates how to combine the predictions from both model types
to create a more robust trading strategy.
"""

import numpy as np
import pandas as pd


class HybridMLStrategy:
    """
    Hybrid strategy that combines DecisionTree and Transformer predictions.
    
    This strategy uses:
    - Decision trees for regime detection and discrete signals
    - Transformers for temporal pattern recognition
    - Dynamic weighting based on market conditions
    """
    
    def __init__(self, dt_model, tf_model, regime_detector=None,
                 weight_config=None):
        """
        Initialize the hybrid strategy.
        
        Parameters:
        -----------
        dt_model : BaseModel
            Decision tree based model (RandomForest, XGBoost, etc.)
        tf_model : TransformerModelWrapper
            Transformer model wrapper
        regime_detector : RegimeDetector or None
            Market regime detector
        weight_config : dict or None
            Configuration for model weights by regime
        """
        self.dt_model = dt_model
        self.tf_model = tf_model
        self.regime_detector = regime_detector
        
        # Default weight configuration
        if weight_config is None:
            weight_config = {
                'trending': {'transformer': 0.7, 'decision_tree': 0.3},
                'ranging': {'transformer': 0.3, 'decision_tree': 0.7},
                'volatile': {'transformer': 0.5, 'decision_tree': 0.5},
                'default': {'transformer': 0.5, 'decision_tree': 0.5}
            }
        self.weight_config = weight_config
        
    def generate_signals(self, data):
        """
        Generate trading signals using hybrid approach.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with features
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with signals and metadata
        """
        # Get predictions from both models
        dt_predictions = self.dt_model.predict(data)
        tf_predictions = self.tf_model.predict(data)
        
        # Detect market regime if detector available
        if self.regime_detector is not None:
            regimes = self.regime_detector.detect_regime(data)
        else:
            regimes = pd.Series('default', index=data.index)
            
        # Combine predictions based on regime
        combined_signals = self._combine_predictions(
            dt_predictions, tf_predictions, regimes
        )
        
        # Create output dataframe
        signals_df = pd.DataFrame(index=data.index)
        signals_df['dt_prediction'] = dt_predictions
        signals_df['tf_prediction'] = tf_predictions
        signals_df['regime'] = regimes
        signals_df['combined_signal'] = combined_signals
        signals_df['position'] = self._generate_positions(combined_signals)
        
        return signals_df
    def predict(self, data):
        signals = self.generate_signals(data)
        return signals["combined_signal"].values

        
    def _combine_predictions(self, dt_pred, tf_pred, regimes):
        """
        Combine predictions based on market regime.
        
        Parameters:
        -----------
        dt_pred : np.ndarray
            Decision tree predictions
        tf_pred : np.ndarray
            Transformer predictions
        regimes : pd.Series
            Market regimes
            
        Returns:
        --------
        np.ndarray
            Combined predictions
        """
        combined = np.zeros_like(dt_pred)
        
        # Apply weights based on regime
        for regime in regimes.unique():
            mask = regimes == regime
            
            # Get weights for this regime
            if regime in self.weight_config:
                weights = self.weight_config[regime]
            else:
                weights = self.weight_config['default']
                
            # Combine predictions
            combined[mask] = (
                weights['decision_tree'] * dt_pred[mask] +
                weights['transformer'] * tf_pred[mask]
            )
            
        return combined
        
    def _generate_positions(self, signals, threshold=0.6):
        """
        Generate position sizes from signals.
        
        Parameters:
        -----------
        signals : np.ndarray
            Combined prediction signals (0-1)
        threshold : float
            Threshold for taking positions
            
        Returns:
        --------
        np.ndarray
            Position sizes (-1 to 1)
        """
        positions = np.zeros_like(signals)
        
        # Long positions
        long_mask = signals > threshold
        positions[long_mask] = (signals[long_mask] - threshold) / (1 - threshold)
        
        # Short positions
        short_mask = signals < (1 - threshold)
        positions[short_mask] = -(((1 - threshold) - signals[short_mask]) / 
                                  (1 - threshold))
        
        return positions
        
    def backtest(self, data, initial_capital=100000, commission=0.001):
        """
        Simple backtest of the hybrid strategy.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with prices
        initial_capital : float
            Starting capital
        commission : float
            Commission rate
            
        Returns:
        --------
        dict
            Backtest results
        """
        # Generate signals
        signals = self.generate_signals(data)
        
        # Calculate returns
        price_returns = data['close'].pct_change()
        
        # Calculate strategy returns
        strategy_returns = signals['position'].shift(1) * price_returns
        
        # Apply commission on position changes
        position_changes = signals['position'].diff().abs()
        commission_costs = position_changes * commission
        strategy_returns -= commission_costs
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # Calculate metrics
        total_return = cumulative_returns.iloc[-1] - 1
        sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        # Count trades
        n_trades = (signals['position'].diff() != 0).sum()
        
        results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'n_trades': n_trades,
            'cumulative_returns': cumulative_returns,
            'signals': signals
        }
        
        return results
        

def create_ensemble_predictions(models, data, weights=None):
    """
    Create ensemble predictions from multiple models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of models with names as keys
    data : pd.DataFrame
        Input data
    weights : dict or None
        Model weights. If None, uses equal weights
        
    Returns:
    --------
    np.ndarray
        Ensemble predictions
    """
    predictions = {}
    
    # Get predictions from each model
    for name, model in models.items():
        predictions[name] = model.predict(data)
        
    # Set default weights if not provided
    if weights is None:
        weights = {name: 1.0 / len(models) for name in models}
        
    # Combine predictions
    ensemble = np.zeros_like(list(predictions.values())[0])
    for name, pred in predictions.items():
        ensemble += weights.get(name, 0) * pred
        
    return ensemble


class AdaptiveHybridStrategy(HybridMLStrategy):
    """
    Advanced hybrid strategy with adaptive weighting.
    
    This strategy dynamically adjusts model weights based on
    recent performance.
    """
    
    def __init__(self, dt_model, tf_model, regime_detector=None,
                 lookback_window=20, adaptation_rate=0.1):
        """
        Initialize adaptive strategy.
        
        Parameters:
        -----------
        dt_model : BaseModel
            Decision tree model
        tf_model : TransformerModelWrapper
            Transformer model
        regime_detector : RegimeDetector or None
            Regime detector
        lookback_window : int
            Window for performance evaluation
        adaptation_rate : float
            Rate of weight adaptation
        """
        super().__init__(dt_model, tf_model, regime_detector)
        self.lookback_window = lookback_window
        self.adaptation_rate = adaptation_rate
        self.performance_history = {
            'decision_tree': [],
            'transformer': []
        }
        
    def update_weights(self, actual_returns, dt_pred, tf_pred):
        """
        Update model weights based on recent performance.
        
        Parameters:
        -----------
        actual_returns : pd.Series
            Actual market returns
        dt_pred : np.ndarray
            Decision tree predictions
        tf_pred : np.ndarray
            Transformer predictions
        """
        # Calculate model performance
        dt_returns = np.sign(dt_pred - 0.5) * actual_returns
        tf_returns = np.sign(tf_pred - 0.5) * actual_returns
        
        # Update performance history
        self.performance_history['decision_tree'].extend(dt_returns)
        self.performance_history['transformer'].extend(tf_returns)
        
        # Keep only recent history
        for key in self.performance_history:
            if len(self.performance_history[key]) > self.lookback_window:
                self.performance_history[key] = (
                    self.performance_history[key][-self.lookback_window:]
                )
                
        # Calculate average performance
        dt_avg = np.mean(self.performance_history['decision_tree'])
        tf_avg = np.mean(self.performance_history['transformer'])
        
        # Update weights
        if dt_avg > tf_avg:
            adjustment = self.adaptation_rate * (dt_avg - tf_avg)
            for regime in self.weight_config:
                if regime != 'default':
                    self.weight_config[regime]['decision_tree'] = min(
                        0.9, self.weight_config[regime]['decision_tree'] + adjustment
                    )
                    self.weight_config[regime]['transformer'] = (
                        1 - self.weight_config[regime]['decision_tree']
                    )
        else:
            adjustment = self.adaptation_rate * (tf_avg - dt_avg)
            for regime in self.weight_config:
                if regime != 'default':
                    self.weight_config[regime]['transformer'] = min(
                        0.9, self.weight_config[regime]['transformer'] + adjustment
                    )
                    self.weight_config[regime]['decision_tree'] = (
                        1 - self.weight_config[regime]['transformer']
                    )
