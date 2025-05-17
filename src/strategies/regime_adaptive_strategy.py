"""
Regime-adaptive trading strategy.

This strategy extends the TrendFollowingStrategy to adapt 
to different market regimes detected by the RegimeDetector.
"""

import pandas as pd
import numpy as np
from .base_strategy import BaseStrategy
from .trend_following import TrendFollowingStrategy
from src.features.regime_detection import RegimeDetector
from src.engines.model_engine import ModelEngine
from src.engines.signal_engine import SignalEngine
from src.features.feature_engineering import engineer_features
from src.backtesting.engine import BacktestEngine


class RegimeAdaptiveStrategy(TrendFollowingStrategy):
    """
    Regime-adaptive trading strategy.
    
    This strategy extends the TrendFollowingStrategy to adapt trading parameters
    and behavior based on the detected market regime.
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
            - threshold (float): Signal threshold
            - position_sizing (str): Position sizing method
            - regime_detection (dict): Regime detection configuration
        """
        # Call the parent's initialize method
        super().initialize(config)
        
        # ---- Default probability thresholds ----
        self.buy_threshold = config.get("buy_threshold", 0.65)
        self.sell_threshold = config.get("sell_threshold", 0.35)
        
        # For backward compatibility with helper methods that still
        # expect a single attribute
        self.threshold = (self.buy_threshold, self.sell_threshold)
        
        # Extract regime detection configuration
        regime_config = config.get('regime_detection', {})
        regime_method = regime_config.get('method', 'trend_volatility')
        regime_params = regime_config.get('params', {})
        
        # Initialize regime detector
        self.regime_detector = RegimeDetector(method=regime_method, **regime_params)
        
        # Store regime-specific parameters
        self.regime_params = config.get('regime_params', {})
        
        # Flag to track if regimes have been detected
        self.regimes_detected = False
        self.known_regime_labels = [] # Initialize known_regime_labels

    def _detect_regimes(self, data):
        """
        Detect market regimes in the data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
            
        Returns:
        --------
        pd.DataFrame
            Data with regime information
        """
        # Detect regimes
        regime_data = self.regime_detector.detect_regime(data)
        
        # Set flag
        self.regimes_detected = True
        
        return regime_data

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
        # Detect regimes if not already done
        if not self.regimes_detected:
            regime_data = self._detect_regimes(data)
            
            # Merge regime information with original data
            # Only keep relevant regime columns to avoid duplication
            regime_cols = ['regime', 'regime_label']
            if 'trend' in regime_data.columns:
                regime_cols.append('trend')
            if 'vol_regime' in regime_data.columns:
                regime_cols.append('vol_regime')
            
            # For debugging, retrieve the regime data
            self.regime_data = regime_data[regime_cols]
        else:
            # Ensure regime data is available
            if not hasattr(self, 'regime_data'):
                regime_data = self._detect_regimes(data)
                regime_cols = ['regime', 'regime_label']
                if 'trend' in regime_data.columns:
                    regime_cols.append('trend')
                if 'vol_regime' in regime_data.columns:
                    regime_cols.append('vol_regime')
                self.regime_data = regime_data[regime_cols]
        
        # Generate base features
        X, y, dates = super().generate_features(data)
        
        # Add regime information as features
        # Match regime data to feature dates
        # self.regime_data should be derived from self.regime_detector.regime_history
        # which is populated based on the full dataset in the backtest method.
        if self.regime_detector.regime_history is None:
            # This is a fallback, ideally regime_history is always populated before this.
            # If detect_regime was never called on full data, this might lead to
            # self.known_regime_labels not being comprehensive.
            self.regime_detector.detect_regime(data) # Detect on current slice if necessary

        regime_features_for_slice = self.regime_detector.regime_history.reindex(dates)
        
        # Convert regime label to one-hot encoding
        if 'regime_label' in regime_features_for_slice.columns and self.known_regime_labels:
            # Ensure 'regime_label' is treated as categorical with ALL known categories
            regime_label_categorical = pd.Categorical(
                regime_features_for_slice['regime_label'],
                categories=self.known_regime_labels
            )
            regime_dummies = pd.get_dummies(regime_label_categorical, prefix='regime')
            if not X.empty and not regime_dummies.empty:
                X = X.join(regime_dummies)
            elif not regime_dummies.empty:
                X = regime_dummies

        elif 'regime_label' in regime_features_for_slice.columns: # Fallback if known_regime_labels is empty
            # This might lead to inconsistency if not all labels are present in this slice
            regime_dummies = pd.get_dummies(regime_features_for_slice['regime_label'], prefix='regime')
            if not X.empty and not regime_dummies.empty:
                X = X.join(regime_dummies)
            elif not regime_dummies.empty:
                X = regime_dummies

        # Add numerical regime indicators
        for col_name in ['regime', 'trend', 'vol_regime']:
            if col_name in regime_features_for_slice.columns:
                # Ensure alignment and fill NaNs that might result from reindexing
                # if a date in X is not in regime_data_for_slice (should not happen if dates match)
                # Fill with 0 for simplicity, or a more sophisticated method if needed.
                X[f'regime_{col_name}'] = regime_features_for_slice[col_name].reindex(X.index).fillna(0)
        
        # Safeguard: Align columns with those seen during training, especially for prediction
        # self.model_engine.feature_names is set during ModelEngine.train()
        if hasattr(self.model_engine, 'feature_names') and \
           self.model_engine.feature_names is not None and \
           not X.empty: # Ensure X is not empty
            # Check if self.model_engine.feature_names is a pandas Index or a list
            expected_cols = list(self.model_engine.feature_names) if isinstance(self.model_engine.feature_names, pd.Index) else self.model_engine.feature_names

            # Add missing columns with 0
            for col in expected_cols:
                if col not in X.columns:
                    X[col] = 0
            
            # Select and reorder columns to match training
            # Ensure all expected_cols are present before reindexing, otherwise it will introduce NaNs or errors.
            # Only reindex if all expected columns are now in X
            if all(col in X.columns for col in expected_cols):
                 X = X[expected_cols]
            else:
                # This case indicates a more fundamental issue if columns are still missing
                # For now, we'll proceed with available columns, but this might lead to errors downstream
                # or indicate a flaw in feature generation logic.
                pass # Or raise an error / log a warning

        return X, y, dates

    def _get_regime_parameters(self, regime_label):
        """
        Get parameters specific to the current regime.
        
        Parameters:
        -----------
        regime_label : str
            Current regime label
            
        Returns:
        --------
        dict
            Regime-specific parameters
        """
        # Default parameters (from base configuration)
        params = {
            "buy_threshold": self.buy_threshold,
            "sell_threshold": self.sell_threshold,
            "position_size_scale": 1.0,
            "lookback": self.config.get('lookback', 20),
            'position_size_pct': self.config.get('position_size_pct', 0.1),
            'max_holding_days': self.config.get('max_holding_days', None),
            'stop_loss_pct': self.config.get('stop_loss_pct', None),
            'take_profit_pct': self.config.get('take_profit_pct', None)
        }
        
        # Override with regime-specific parameters if available
        if regime_label in self.regime_params:
            regime_specific = self.regime_params.get(regime_label, {})
            params.update(regime_specific)
        
        return params

    def generate_signals(self, features, predictions, dates):
        """
        Generate trading signals based on features, predictions, and market regime.
        
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
        # Initialize signals as in the parent class
        base_signals = super().generate_signals(features, predictions, dates)
        
        # If regimes haven't been detected, return base signals
        if not self.regimes_detected or not hasattr(self, 'regime_data'):
            return base_signals
        
        # Create modified signals based on regimes
        signals = base_signals.copy()
        
        # Add regime information to signals
        regime_for_dates = self.regime_data.reindex(dates)
        signals['regime'] = regime_for_dates['regime'].values
        signals['regime_label'] = regime_for_dates['regime_label'].values
        
        # Adjust signal generation based on regime
        for date in signals.index:
            regime_label = signals.loc[date, 'regime_label']
            
            # Skip if regime label is missing
            if pd.isna(regime_label):
                continue
            
            original_signal = signals.loc[date, 'signal'] # Get signal from base_signals
            probability = signals.loc[date, 'probability']

            params = self._get_regime_parameters(regime_label)
            buy_threshold = params.get('buy_threshold', self.buy_threshold)
            sell_threshold = params.get('sell_threshold', self.sell_threshold)
            
            # Adjust signal based on threshold
            new_signal = original_signal # Default to original signal
            if probability > buy_threshold:
                new_signal = 1  # Buy
            elif probability < sell_threshold:
                new_signal = -1  # Sell
            else:
                new_signal = 0  # Hold
            
            signals.loc[date, 'signal'] = new_signal
            
            # Adjust position size based on regime
            position_size_pct = params.get('position_size_pct', 0.1)
            signals.loc[date, 'position_size_pct'] = position_size_pct
            
            # Add stop loss and take profit levels if specified
            stop_loss_pct = params.get('stop_loss_pct')
            take_profit_pct = params.get('take_profit_pct')
            
            if stop_loss_pct is not None:
                signals.loc[date, 'stop_loss_pct'] = stop_loss_pct
            
            if take_profit_pct is not None:
                signals.loc[date, 'take_profit_pct'] = take_profit_pct
        
        return signals

    def backtest(self, data, train_data=None, test_data=None):
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
            
        Returns:
        --------
        dict
            Backtest results
        """
        # Ensure regimes are detected for the entire dataset
        if not self.regimes_detected:
            # _detect_regimes calls self.regime_detector.detect_regime(data) internally
            # and returns the regime data, but we want to ensure self.regime_data
            # is consistently from the detector's history for the full period.
            self._detect_regimes(data) # This populates self.regime_detector.regime_history

            if self.regime_detector.regime_history is not None:
                self.known_regime_labels = sorted(self.regime_detector.regime_history['regime_label'].dropna().unique().tolist())
                
                # Ensure self.regime_data is based on the full history for later analysis
                relevant_cols = ['regime', 'regime_label']
                if 'trend' in self.regime_detector.regime_history.columns:
                    relevant_cols.append('trend')
                if 'vol_regime' in self.regime_detector.regime_history.columns:
                    relevant_cols.append('vol_regime')
                
                # Filter out columns that might not exist if regime_history is minimal
                existing_relevant_cols = [col for col in relevant_cols if col in self.regime_detector.regime_history.columns]
                if existing_relevant_cols:
                    self.regime_data = self.regime_detector.regime_history[existing_relevant_cols].copy()
                else:
                    # Fallback or error handling if essential columns are missing
                    # For now, create an empty DataFrame with expected index to avoid downstream errors,
                    # though this indicates an issue in regime detection if it happens.
                    self.regime_data = pd.DataFrame(index=data.index)


            self.regimes_detected = True # Ensure flag is set here as _detect_regimes sets its own internal flag
                                           # but RegimeAdaptiveStrategy uses this one.
        
        # Run backtest with the parent class
        results = super().backtest(data, train_data, test_data)
        
        # Add regime analysis to results
        self._add_regime_analysis(results, test_data)
        
        return results
    
    def _add_regime_analysis(self, results, test_data):
        """
        Add regime analysis to backtest results.
        
        Parameters:
        -----------
        results : dict
            Backtest results
        test_data : pd.DataFrame
            Test data used for backtesting
        """
        # Skip if no trades or regimes
        if 'trades' not in results or results['trades'].empty or not self.regimes_detected:
            return
        
        # Get trades
        trades = results['trades']
        
        # Make sure all required columns are in regime_data
        required_cols = ['regime_label']
        if not all(col in self.regime_data.columns for col in required_cols):
            return
        
        # Add regime information to trades
        trades['entry_regime'] = trades['entry_date'].map(
            lambda x: self.regime_data.loc[x, 'regime_label'] if x in self.regime_data.index else np.nan
        )
        
        trades['exit_regime'] = trades['exit_date'].map(
            lambda x: self.regime_data.loc[x, 'regime_label'] if x in self.regime_data.index else np.nan
        )
        
        # Calculate regime-specific performance
        regime_performance = trades.groupby('entry_regime').agg({
            'pnl': ['count', 'sum', 'mean'],
            'return': ['mean', 'std', 'min', 'max']
        })
        
        # Flatten columns
        regime_performance.columns = [f"{col[0]}_{col[1]}" for col in regime_performance.columns]
        
        # Calculate win rate
        regime_performance['win_rate'] = trades.groupby('entry_regime')['pnl'].apply(
            lambda x: (x > 0).mean()
        )
        
        # Calculate Sharpe ratio (annualized)
        regime_performance['sharpe'] = (
            regime_performance['return_mean'] * 252 / 
            (regime_performance['return_std'] * np.sqrt(252))
        )
        
        # Store regime performance in results
        results['regime_performance'] = regime_performance
        
        # Calculate time spent in each regime
        regime_counts = self.regime_data.loc[test_data.index, 'regime_label'].value_counts()
        regime_pcts = regime_counts / len(test_data)
        
        # Store regime time distribution
        results['regime_distribution'] = pd.DataFrame({
            'count': regime_counts,
            'percentage': regime_pcts
        })
        
    def get_metrics(self):
        """
        Get strategy metrics.
        
        Returns:
        --------
        dict
            Strategy metrics including model and backtest metrics
        """
        # Get base metrics from parent class
        metrics = super().get_metrics()
        
        # Add regime-specific metrics if available
        if hasattr(self, 'regime_detector') and self.regimes_detected:
            # Get current regime
            try:
                current_regime = self.regime_detector.get_current_regime()
                metrics['current_regime'] = current_regime['regime_label']
            except (ValueError, KeyError):
                pass
            
            # Add regime stats
            try:
                regime_stats = self.regime_detector.get_regime_stats()
                # Select key metrics only to avoid overwhelming output
                key_metrics = ['count', 'ann_return', 'sharpe', 'win_rate', 'time_pct']
                if not regime_stats.empty:
                    best_regime = regime_stats.sort_values('ann_return', ascending=False).iloc[0]
                    worst_regime = regime_stats.sort_values('ann_return').iloc[0]
                    
                    metrics['best_regime'] = {
                        'label': best_regime.name,
                        'ann_return': best_regime['ann_return'],
                        'sharpe': best_regime['sharpe'],
                        'win_rate': best_regime['positive_pct']
                    }
                    
                    metrics['worst_regime'] = {
                        'label': worst_regime.name,
                        'ann_return': worst_regime['ann_return'],
                        'sharpe': worst_regime['sharpe'],
                        'win_rate': worst_regime['positive_pct']
                    }
            except (ValueError, KeyError, AttributeError):
                pass
            
        return metrics