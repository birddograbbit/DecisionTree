"""
Trend following strategy using ensemble models.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy, BUY_THRESHOLD, SELL_THRESHOLD
from src.engines.model_engine import ModelEngine
from src.engines.signal_engine import SignalEngine
from src.features.feature_engineering import engineer_features
from src.backtesting.engine import BacktestEngine
from src.utils.adaptive_thresholds import are_adaptive_thresholds_needed, calculate_adaptive_thresholds
import config

class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy using ensemble models.
    
    This strategy uses technical indicators and price patterns to identify
    trends, and generates trading signals based on the predicted direction.
    """

    def initialize(self, config):
        """
        Initialize strategy with configuration.
        
        Parameters:
        -----------
        config : dict
            Strategy configuration with keys:
            - model_type (str): Type of model to use
            - model_params (dict): Model parameters
            - position_sizing (str, optional): Position sizing method for
              ``SignalEngine`` ('fixed' or 'confidence')
        """
        # Call parent initialize to setup thresholds
        super().initialize(config)
        
        # Store configuration
        self.config = config
        
        # Extract key parameters
        self.model_type = config.get('model_type', 'random_forest')
        self.model_params = config.get('model_params', {})
        
        # Initialize engines
        self.model_engine = ModelEngine(self.model_type, self.model_params)

        # Position sizing method for SignalEngine
        self.position_sizing = config.get('position_sizing', 'confidence')
        self.signal_engine = SignalEngine(position_sizing=self.position_sizing)
        
        # Store strategy state
        self.is_trained = False
        self.metrics = {}
        
        # Adaptive thresholds settings
        self.use_adaptive_thresholds = config.get('use_adaptive_thresholds', 'auto')  # 'auto', 'always', 'never'
        
        # Custom thresholds if specified
        if 'buy_threshold' in config and 'sell_threshold' in config:
            self.buy_threshold = config['buy_threshold']
            self.sell_threshold = config['sell_threshold']
            self.signal_engine.buy_threshold = config['buy_threshold']
            self.signal_engine.sell_threshold = config['sell_threshold']

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
        timeframe = self.config.get('timeframe', 'daily')
        lookback = config.LOOKBACK_PERIOD_5MIN if timeframe == '5min' else config.LOOKBACK_PERIOD
        return engineer_features(data, lookback_period=lookback, timeframe=timeframe)

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
        # Call parent method to check for adaptive thresholds
        super().generate_signals(features, predictions, dates)
        
        # Determine if we should use adaptive thresholds
        custom_thresholds = None
        if self.use_adaptive_thresholds == 'always' or (self.use_adaptive_thresholds == 'auto' and are_adaptive_thresholds_needed(predictions)):
            # Calculate adaptive thresholds
            buy_threshold, sell_threshold = calculate_adaptive_thresholds(
                predictions,
                buy_percentile=self.config.get('buy_percentile', 80),
                sell_percentile=self.config.get('sell_percentile', 20)
            )
            custom_thresholds = (buy_threshold, sell_threshold)
            
            # Log the adaptive thresholds
            stats = {
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'mean': float(np.mean(predictions)),
                'range': float(np.max(predictions) - np.min(predictions))
            }
            print(f"\nUsing adaptive thresholds: buy={buy_threshold:.4f}, sell={sell_threshold:.4f}")
            print(f"Prediction stats: min={stats['min']:.4f}, max={stats['max']:.4f}, mean={stats['mean']:.4f}, range={stats['range']:.4f}")
        
        # Generate raw signals
        signals = self.signal_engine.generate_signals(
            predictions, 
            dates, 
            symbol=self.config.get('symbol', 'SPY'),
            custom_thresholds=custom_thresholds
        )
        
        # Apply signal filters
        signals = self.signal_engine.apply_filters(
            signals,
            consecutive_buys=self.config.get('consecutive_buys', False),
            min_holding_days=self.config.get('min_holding_days', 1),
            max_holding_days=self.config.get('max_holding_days', None)
        )
        
        return signals

    def train(self, train_data):
        """
        Train the strategy's model on the given data.
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
            
        Returns:
        --------
        self
            For method chaining
        """
        # Generate features
        X_train, y_train, dates_train = self.generate_features(train_data)
        
        # Train model
        self.model_engine.train(X_train, y_train, 
                               cross_validation=self.config.get('cross_validation', True),
                               cv=self.config.get('cv_folds', 5),
                               perform_hpo=self.config.get('perform_hpo', False),
                               hpo_param_grid=self.config.get('hpo_param_grid', None),
                               hpo_cv=self.config.get('hpo_cv', 3),
                               hpo_scoring=self.config.get('hpo_scoring', 'roc_auc'))
        
        # Mark as trained
        self.is_trained = True
        
        return self

    def predict(self, test_data):
        """
        Generate predictions and signals for the test data.
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            Test data
            
        Returns:
        --------
        tuple
            (signals, predictions)
        """
        if not self.is_trained:
            raise ValueError("Strategy must be trained before prediction.")
        
        # Generate features
        X_test, y_test, dates_test = self.generate_features(test_data)
        
        # Generate predictions
        predictions = self.model_engine.predict(X_test)
        
        # --- Debugging: Print raw predictions ---
        print("\n--- Raw Model Predictions (Sample) ---")
        print(pd.Series(predictions).head())
        print(f"Min prediction: {np.min(predictions)}, Max prediction: {np.max(predictions)}, Mean prediction: {np.mean(predictions)}")
        # --- End Debugging ---

        # Check if we need adaptive thresholds
        custom_thresholds = None
        if self.use_adaptive_thresholds == 'always' or (self.use_adaptive_thresholds == 'auto' and are_adaptive_thresholds_needed(predictions)):
            # Calculate adaptive thresholds for raw signals display
            buy_threshold, sell_threshold = calculate_adaptive_thresholds(
                predictions,
                buy_percentile=self.config.get('buy_percentile', 80),
                sell_percentile=self.config.get('sell_percentile', 20)
            )
            custom_thresholds = (buy_threshold, sell_threshold)

        # Generate signals (before filtering in the main generate_signals method)
        # We call the signal_engine.generate_signals directly here for a raw look
        raw_signals_df = self.signal_engine.generate_signals(
            predictions,
            dates_test,
            symbol=self.config.get('symbol', 'SPY'),
            custom_thresholds=custom_thresholds
        )
        # --- Debugging: Print raw signals ---
        print("\n--- Raw Signals (Before main apply_filters in TrendFollowingStrategy.generate_signals) ---")
        print(raw_signals_df[raw_signals_df['signal'] != 0].head()) # Show only non-hold signals if any
        if raw_signals_df[raw_signals_df['signal'] != 0].empty:
            print("No non-hold signals generated at this stage.")
        print(f"Total raw signals: {len(raw_signals_df)}, Buy: {len(raw_signals_df[raw_signals_df['signal'] == 1])}, Sell: {len(raw_signals_df[raw_signals_df['signal'] == -1])}")
        # --- End Debugging ---

        # Generate signals (this will call signal_engine.generate_signals and then apply_filters)
        signals = self.generate_signals(X_test, predictions, dates_test)
        
        return signals, predictions

    def backtest(self, data, train_data=None, test_data=None, timeframe='daily'):
        """
        Run backtest for the strategy.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
        train_data : pd.DataFrame, optional
            Training data (if None, uses 70% of data)
        test_data : pd.DataFrame, optional
            Testing data (if None, uses 30% of data)
        timeframe : str, default='daily'
            Trading timeframe ('daily', '5min', '5T')
            
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
        
        # Train model
        self.train(train_data)
        
        # Generate signals
        signals, predictions = self.predict(test_data)
        
        # Create a dictionary with test data (for backtesting)
        symbol = self.config.get('symbol', 'SPY')
        test_data_dict = {symbol: test_data}
        
        # Run backtest
        backtest_engine = BacktestEngine(
            initial_capital=self.config.get('initial_capital', 100000.0),
            commission=self.config.get('commission', 0.001),
            slippage=self.config.get('slippage', 0.001)
        )
        
        results = backtest_engine.run_backtest(signals, test_data_dict, timeframe)
        
        # Store metrics
        self.metrics.update(results.get('performance', {}))
        
        return results
    
    def get_metrics(self):
        """
        Get strategy metrics.
        
        Returns:
        --------
        dict
            Strategy metrics including model and backtest metrics
        """
        # Combine model and strategy metrics
        metrics = {}
        metrics.update(self.model_engine.metrics)
        metrics.update(self.metrics)
        
        return metrics
    
    def save(self, path):
        """
        Save the strategy's model to disk.
        
        Parameters:
        -----------
        path : str
            Path to save model
        """
        self.model_engine.save(path)
    
    def load(self, path):
        """
        Load the strategy's model from disk.
        
        Parameters:
        -----------
        path : str
            Path to saved model
            
        Returns:
        --------
        self
            For method chaining
        """
        self.model_engine.load(path)
        self.is_trained = True
        return self
