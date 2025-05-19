# src/strategies/regime_adaptive_strategy.py

"""
Regime-adaptive trading strategy.

This strategy extends the TrendFollowingStrategy to adapt 
to different market regimes detected by the RegimeDetector.
"""

import pandas as pd
import numpy as np
import logging
from .base_strategy import BaseStrategy
from .trend_following import TrendFollowingStrategy
from src.features.regime_detection import RegimeDetector
from src.engines.model_engine import ModelEngine
from src.engines.signal_engine import SignalEngine
from src.features.feature_engineering import engineer_features
from src.backtesting.engine import BacktestEngine
from src.models.model_factory import ModelFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            - regime_detection (dict): Regime detection configuration
        """
        # Call the parent's initialize method
        super().initialize(config)
        
        # ---- Default probability thresholds ----
        # Use global constants to maintain a single source of truth
        from .base_strategy import BUY_THRESHOLD, SELL_THRESHOLD
        self.buy_threshold = BUY_THRESHOLD
        self.sell_threshold = SELL_THRESHOLD
        
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
        
        # Add default regime parameters for every possible regime to avoid key errors
        self._add_default_regime_params()
        
        # Flag to use regime-specific hyperparameters
        self.use_regime_specific_params = config.get('use_regime_specific_params', False)
        
        # Store hyperparameter optimization settings
        self.use_optimized = config.get('use_optimized', False)
        
        # Initialize regime-specific models
        self.regime_models = {}
        
    def _add_default_regime_params(self):
        """
        Add default parameters for all possible regimes to avoid key errors.
        """
        default_regimes = [
            'strong_uptrend', 'uptrend', 'weak_uptrend',
            'volatile_neutral', 'neutral', 'low_vol_neutral',
            'weak_downtrend', 'downtrend', 'strong_downtrend'
        ]
        
        # Default parameter values
        default_params = {
            'position_size_pct': 0.05,    # Default position size
            'stop_loss_pct': 0.03,        # Default stop loss
            'take_profit_pct': 0.07,      # Default take profit
            'use_low_thresholds': False
        }
        
        # Add default parameters for any missing regimes
        for regime in default_regimes:
            if regime not in self.regime_params:
                self.regime_params[regime] = default_params.copy()

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
        try:
            # Detect regimes
            regime_data = self.regime_detector.detect_regime(data)
            
            # Set flag
            self.regimes_detected = True
            
            return regime_data
            
        except Exception as e:
            # Log error and create minimal regime data
            logger.error(f"Error in regime detection: {e}")
            result = pd.DataFrame(index=data.index)
            result['close'] = data['close']
            result['regime'] = 0
            result['regime_label'] = 'neutral'
            
            # Add minimal columns for trend_volatility method
            result['trend'] = 0
            result['vol_regime'] = 0
            result['volatility'] = 0.0
            result['vol_rank'] = 0.5
            
            # Set flag but with a warning
            self.regimes_detected = True
            self.regime_detector.regime_history = result
            
            return result

    def _initialize_regime_specific_models(self, train_data):
        """
        Initialize regime-specific models for each detected regime.
        
        Parameters:
        -----------
        train_data : pd.DataFrame
            Training data
        """
        if not self.use_regime_specific_params or not self.regimes_detected:
            return
        
        # Get regime labels
        if not hasattr(self, 'regime_data') or self.regime_data is None:
            logger.warning("No regime data available. Cannot initialize regime-specific models.")
            return
        
        try:
            # Get unique regime labels from training data
            regime_labels = self.regime_data.loc[train_data.index, 'regime_label'].dropna().unique()
            
            # Skip if no regimes detected
            if len(regime_labels) == 0:
                logger.warning("No regimes detected in training data. Using default model.")
                return
            
            # Use model_type and model_params from config
            model_type = self.config.get('model_type')
            model_params = self.config.get('model_params', {}).copy()
            
            # Set optimized flag
            model_params['use_optimized'] = self.use_optimized
            
            # Generate features
            X, y, dates = self.generate_features(train_data)
            
            # Track models for each regime
            for regime_label in regime_labels:
                # Get indices for this regime
                regime_indices = self.regime_data.loc[train_data.index, 'regime_label'] == regime_label
                
                # Skip if too few samples
                if regime_indices.sum() < 100:  # Minimum samples threshold
                    logger.warning(f"Too few samples ({regime_indices.sum()}) for regime {regime_label}. "
                                  f"Using default model.")
                    continue
                
                # Extract data for this regime
                X_regime = X.iloc[regime_indices.values]
                y_regime = y.iloc[regime_indices.values]
                
                # Create model with regime parameter
                model_params['regime'] = regime_label
                
                try:
                    regime_model = ModelFactory.create_model(model_type, **model_params)
                    self.regime_models[regime_label] = ModelEngine(model=regime_model)
                    
                    # Train model
                    self.regime_models[regime_label].train(X_regime, y_regime)
                    
                    logger.info(f"Trained regime-specific model for {regime_label} "
                               f"with {len(X_regime)} samples")
                except Exception as e:
                    logger.error(f"Error creating regime-specific model for {regime_label}: {e}")
                    # Skip this regime - we'll use the base model
            
            logger.info(f"Initialized {len(self.regime_models)} regime-specific models")
            
        except Exception as e:
            logger.error(f"Error initializing regime-specific models: {e}")

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
        try:
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
            
            # Ensure self.regime_data has values before proceeding
            if self.regime_data is None or len(self.regime_data) == 0:
                logger.warning("No regime data available. Using base features only.")
                return X, y, dates
                
            # Add regime information as features
            # Match regime data to feature dates
            # self.regime_data should be derived from self.regime_detector.regime_history
            # which is populated based on the full dataset in the backtest method.
            if self.regime_detector.regime_history is None:
                # This is a fallback, ideally regime_history is always populated before this.
                # If detect_regime was never called on full data, this might lead to
                # self.known_regime_labels not being comprehensive.
                logger.warning("Regime history not found. Detecting regimes on current data slice.")
                self.regime_detector.detect_regime(data) # Detect on current slice if necessary
                
            # Safely get regime features for this data slice
            try:
                # Reindex regime history to current feature dates
                if self.regime_detector.regime_history is not None:
                    regime_features_for_slice = self.regime_detector.regime_history.reindex(dates)
                    
                    # Fill NaN values that might result from reindexing
                    if 'regime_label' in regime_features_for_slice.columns:
                        regime_features_for_slice['regime_label'] = regime_features_for_slice['regime_label'].fillna('neutral')
                else:
                    # Create empty DataFrame with proper index if no regime history exists
                    logger.warning("No regime history available. Creating empty regime features.")
                    regime_features_for_slice = pd.DataFrame(index=dates)
                    regime_features_for_slice['regime_label'] = 'neutral'
                    
            except Exception as e:
                logger.error(f"Error reindexing regime features: {e}")
                # Create a minimal feature set
                regime_features_for_slice = pd.DataFrame(index=dates)
                regime_features_for_slice['regime_label'] = 'neutral'
            
            # Convert regime label to one-hot encoding
            if 'regime_label' in regime_features_for_slice.columns:
                # Ensure we have labels even if known_regime_labels is empty
                if not self.known_regime_labels:
                    # Use standard regime labels if the list is empty
                    self.known_regime_labels = [
                        'strong_uptrend', 'uptrend', 'weak_uptrend',
                        'volatile_neutral', 'neutral', 'low_vol_neutral',
                        'weak_downtrend', 'downtrend', 'strong_downtrend'
                    ]
                
                # Only include regime labels that actually appear in the data
                actual_labels = regime_features_for_slice['regime_label'].dropna().unique()
                
                # Use a safe approach to create dummy variables
                try:
                    # Create categorical variable with all known categories
                    regime_label_categorical = pd.Categorical(
                        regime_features_for_slice['regime_label'],
                        categories=self.known_regime_labels
                    )
                    regime_dummies = pd.get_dummies(regime_label_categorical, prefix='regime')
                except Exception as e:
                    logger.error(f"Error creating regime dummies with categorical: {e}")
                    # Fallback to basic get_dummies without categories
                    regime_dummies = pd.get_dummies(regime_features_for_slice['regime_label'], prefix='regime')
                
                # Safely add the regime dummies to features
                if not X.empty and not regime_dummies.empty:
                    # Ensure indexes match
                    regime_dummies.index = X.index
                    X = X.join(regime_dummies)
                elif not regime_dummies.empty:
                    X = regime_dummies
            
            # Add numerical regime indicators
            for col_name in ['regime', 'trend', 'vol_regime']:
                if col_name in regime_features_for_slice.columns:
                    # Ensure alignment and fill NaNs that might result from reindexing
                    # Fill with 0 for simplicity, or a more sophisticated method if needed.
                    col_values = regime_features_for_slice[col_name].reindex(X.index).fillna(0)
                    X[f'regime_{col_name}'] = col_values
            
            # Safeguard: Align columns with those seen during training, especially for prediction
            if hasattr(self.model_engine, 'feature_names') and \
               self.model_engine.feature_names is not None and \
               not X.empty:  # Ensure X is not empty
                try:
                    # Check if self.model_engine.feature_names is a pandas Index or a list
                    expected_cols = list(self.model_engine.feature_names) if isinstance(self.model_engine.feature_names, pd.Index) else self.model_engine.feature_names
                    
                    # Add missing columns with 0
                    for col in expected_cols:
                        if col not in X.columns:
                            X[col] = 0
                    
                    # Select and reorder columns to match training - only if all expected columns are present
                    if all(col in X.columns for col in expected_cols):
                        X = X[expected_cols]
                    else:
                        # Just use what we have instead of failing
                        logger.warning("Some expected features are missing. Using available features only.")
                        missing_cols = [col for col in expected_cols if col not in X.columns]
                        logger.debug(f"Missing columns: {missing_cols}")
                except Exception as e:
                    logger.error(f"Error aligning feature columns: {e}")
            
            return X, y, dates
            
        except Exception as e:
            logger.error(f"Error in generate_features: {e}")
            # Return the base features from parent class as fallback
            return super().generate_features(data)

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
        
        # Default to 'neutral' if regime_label is None or not in regime_params
        if pd.isna(regime_label) or regime_label not in self.regime_params:
            regime_label = 'neutral'
            
        # Override with regime-specific parameters if available
        if regime_label in self.regime_params:
            regime_specific = self.regime_params.get(regime_label, {}).copy()

            # Apply lower thresholds if requested and explicit values aren't provided
            use_low = regime_specific.pop('use_low_thresholds', False)
            params.update(regime_specific)
            if use_low:
                params['buy_threshold'] = params.get('buy_threshold', 0.55)
                params['sell_threshold'] = params.get('sell_threshold', 0.45)

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
        try:
            # Initialize signals based on parent method, but without sending
            # predictions (we'll use regime-specific models if available)
            # Create a signals dataframe first
            signals = pd.DataFrame(index=dates)
            signals['date'] = dates
            signals['symbol'] = self.config.get('symbol', 'SPY')
            signals['signal'] = 0  # Initialize with no signal
            signals['probability'] = 0.5  # Default probability
            signals['position_size'] = 0.0  # Default position size
            
            # If regimes haven't been detected or regime_data is missing, use base signals
            if not self.regimes_detected or not hasattr(self, 'regime_data'):
                logger.warning("No regimes detected. Using base signals.")
                return super().generate_signals(features, predictions, dates)
            
            # Add regime information to signals
            try:
                # Check if regime_data matches dates needed
                if self.regime_data is not None and len(self.regime_data) > 0:
                    # Match regime data to signal dates
                    regime_for_dates = self.regime_data.reindex(dates)
                    
                    # Fill NaN values to avoid downstream issues
                    regime_for_dates['regime'] = regime_for_dates['regime'].fillna(0)
                    regime_for_dates['regime_label'] = regime_for_dates['regime_label'].fillna('neutral')
                    
                    # Add regime information
                    signals['regime'] = regime_for_dates['regime'].values
                    signals['regime_label'] = regime_for_dates['regime_label'].values
                else:
                    # Add default values if regime_data is invalid
                    signals['regime'] = 0
                    signals['regime_label'] = 'neutral'
            except Exception as e:
                logger.error(f"Error adding regime data to signals: {e}")
                # Add default values if adding regime data fails
                signals['regime'] = 0
                signals['regime_label'] = 'neutral'
            
            # Use regime-specific models if available
            if self.use_regime_specific_params and self.regime_models:
                # Process each regime separately
                for regime_label, regime_rows in signals.groupby('regime_label'):
                    # Skip if no rows for this regime
                    if len(regime_rows) == 0:
                        continue
                    
                    # Get indices for this regime
                    regime_indices = signals.index.isin(regime_rows.index)
                    
                    # Use regime-specific model if available
                    if regime_label in self.regime_models:
                        logger.info(f"Using regime-specific model for {regime_label}")
                        # Get features for this regime
                        regime_features = features.loc[regime_indices]
                        
                        # Get predictions from regime-specific model
                        regime_predictions = self.regime_models[regime_label].predict(regime_features)
                        
                        # Update signals with regime-specific predictions
                        signals.loc[regime_indices, 'probability'] = regime_predictions
                    else:
                        # Use base model predictions
                        signals.loc[regime_indices, 'probability'] = predictions[regime_indices]
            else:
                # Use base model predictions for all regimes
                signals['probability'] = predictions
            
            # Apply signal generation based on regime and probability
            for date in signals.index:
                try:
                    regime_label = signals.loc[date, 'regime_label']
                    probability = signals.loc[date, 'probability']
                    
                    # Skip if regime label is missing
                    if pd.isna(regime_label):
                        continue
                    
                    # Get regime-specific parameters
                    params = self._get_regime_parameters(regime_label)
                    buy_threshold = params.get('buy_threshold', self.buy_threshold)
                    sell_threshold = params.get('sell_threshold', self.sell_threshold)
                    
                    # Determine signal
                    if probability > buy_threshold:
                        new_signal = 1  # Buy
                    elif probability < sell_threshold:
                        new_signal = -1  # Sell
                    else:
                        new_signal = 0  # Hold
                    
                    signals.loc[date, 'signal'] = new_signal
                    
                    # Calculate position size based on probability distance from threshold
                    if new_signal == 1:
                        # Buy signal: scale by distance from buy threshold
                        confidence = (probability - buy_threshold) / (1 - buy_threshold)
                        signals.loc[date, 'position_size'] = max(0.0, min(1.0, confidence))
                    elif new_signal == -1:
                        # Sell signal: scale by distance from sell threshold
                        confidence = (sell_threshold - probability) / sell_threshold
                        signals.loc[date, 'position_size'] = max(0.0, min(1.0, confidence))
                    
                    # Adjust position size based on regime
                    position_size_pct = params.get('position_size_pct', 0.1)
                    signals.loc[date, 'position_size_pct'] = position_size_pct
                    signals.loc[date, 'position_size'] *= position_size_pct
                    
                    # Add stop loss and take profit levels if specified
                    stop_loss_pct = params.get('stop_loss_pct')
                    take_profit_pct = params.get('take_profit_pct')
                    
                    if stop_loss_pct is not None:
                        signals.loc[date, 'stop_loss_pct'] = stop_loss_pct
                    
                    if take_profit_pct is not None:
                        signals.loc[date, 'take_profit_pct'] = take_profit_pct
                        
                except Exception as e:
                    logger.error(f"Error adjusting signal for date {date}: {e}")
                    # Use default values if signal adjustment fails
            
            # Apply additional signal filtering and processing from SignalEngine
            if hasattr(self, 'signal_engine') and self.signal_engine is not None:
                signals = self.signal_engine.apply_filters(signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in generate_signals: {e}")
            # Return base signals if regime adjustment fails
            return super().generate_signals(features, predictions, dates)

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
        try:
            # Ensure regimes are detected for the entire dataset
            if not self.regimes_detected:
                logger.info("Detecting regimes for the full dataset")
                try:
                    # _detect_regimes calls self.regime_detector.detect_regime(data) internally
                    # and returns the regime data, but we want to ensure self.regime_data
                    # is consistently from the detector's history for the full period.
                    self._detect_regimes(data) # This populates self.regime_detector.regime_history
                    
                    if self.regime_detector.regime_history is not None:
                        # Get unique regime labels
                        regime_labels = self.regime_detector.regime_history['regime_label'].dropna().unique().tolist()
                        self.known_regime_labels = sorted(regime_labels)
                        
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
                            logger.warning("Essential regime columns missing. Creating minimal regime_data.")
                            self.regime_data = pd.DataFrame(index=data.index)
                            self.regime_data['regime'] = 0
                            self.regime_data['regime_label'] = 'neutral'
                except Exception as e:
                    logger.error(f"Error in regime detection during backtest: {e}")
                    # Create minimal regime data if detection fails
                    self.regime_data = pd.DataFrame(index=data.index)
                    self.regime_data['regime'] = 0
                    self.regime_data['regime_label'] = 'neutral'
                    
                self.regimes_detected = True # Ensure flag is set, even if detection had issues
                
            # Initialize regime-specific models if using regime-specific hyperparameters
            if self.use_regime_specific_params and train_data is not None:
                self._initialize_regime_specific_models(train_data)
            
            # Run backtest (modified to use regime-specific models)
            # Split data if not provided
            if train_data is None or test_data is None:
                train_size = int(len(data) * 0.7)
                train_data = data.iloc[:train_size]
                test_data = data.iloc[train_size:]
                
            # Generate features and train model
            train_features, train_target, train_dates = self.generate_features(train_data)
            
            # Train base model (still needed even with regime-specific models)
            self.model_engine.train(train_features, train_target)
            
            # Generate features for test data
            test_features, test_target, test_dates = self.generate_features(test_data)
            
            # Generate predictions (base model)
            predictions = self.model_engine.predict(test_features)
            
            # Generate signals (will use regime-specific models if available)
            signals = self.generate_signals(test_features, predictions, test_dates)
            
            # Run backtest
            backtest_results = self.backtest_engine.run_backtest(
                data=test_data,
                signals=signals,
                initial_capital=100000,
                commission=0.001,
                slippage=0.001
            )
            
            # Add regime analysis to results
            self._add_regime_analysis(backtest_results, test_data)
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            # If error occurs, try to run the parent's backtest without regime adaptation
            logger.info("Falling back to base TrendFollowingStrategy backtest")
            return super().backtest(data, train_data, test_data)
    
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
        try:
            # Skip if no trades or regimes
            if 'trades' not in results or results['trades'].empty or not self.regimes_detected:
                return
            
            # Make sure we have regime_data to work with
            if not hasattr(self, 'regime_data') or self.regime_data is None or self.regime_data.empty:
                logger.warning("No regime data available for analysis")
                return
            
            # Make sure all required columns are in regime_data
            required_cols = ['regime_label']
            if not all(col in self.regime_data.columns for col in required_cols):
                logger.warning(f"Missing required columns in regime_data: {required_cols}")
                return
            
            # Get trades
            trades = results['trades']
            
            # Add regime information to trades
            trades['entry_regime'] = trades['entry_date'].map(
                lambda x: self.regime_data.loc[x, 'regime_label'] if x in self.regime_data.index else 'unknown'
            )
            
            trades['exit_regime'] = trades['exit_date'].map(
                lambda x: self.regime_data.loc[x, 'regime_label'] if x in self.regime_data.index else 'unknown'
            )
            
            # Fill NaN values
            trades['entry_regime'] = trades['entry_regime'].fillna('unknown')
            trades['exit_regime'] = trades['exit_regime'].fillna('unknown')
            
            # Calculate regime-specific performance with error handling
            try:
                # Calculate statistics only if we have trade data for at least one regime
                if trades['entry_regime'].nunique() > 0:
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
                    
                    # Calculate Sharpe ratio (annualized) with protection against division by zero
                    regime_performance['sharpe'] = np.where(
                        regime_performance['return_std'] > 0,
                        regime_performance['return_mean'] * 252 / (regime_performance['return_std'] * np.sqrt(252)),
                        0  # Default value when std is 0
                    )
                    
                    # Store regime performance in results
                    results['regime_performance'] = regime_performance
                else:
                    logger.warning("No regime trade data for performance analysis")
            except Exception as e:
                logger.error(f"Error calculating regime performance: {e}")
            
            # Calculate time spent in each regime
            try:
                if test_data is not None and len(test_data) > 0:
                    # Get regime labels for test period
                    test_regimes = self.regime_data.loc[test_data.index, 'regime_label'].fillna('unknown')
                    
                    # Calculate distribution
                    regime_counts = test_regimes.value_counts()
                    regime_pcts = regime_counts / len(test_data)
                    
                    # Store regime time distribution
                    results['regime_distribution'] = pd.DataFrame({
                        'count': regime_counts,
                        'percentage': regime_pcts
                    })
            except Exception as e:
                logger.error(f"Error calculating regime distribution: {e}")
                
        except Exception as e:
            logger.error(f"Error in regime analysis: {e}")
            
    def get_metrics(self):
        """
        Get strategy metrics.
        
        Returns:
        --------
        dict
            Strategy metrics including model and backtest metrics
        """
        try:
            # Get base metrics from parent class
            metrics = super().get_metrics()
            
            # Add regime-specific metrics if available
            if hasattr(self, 'regime_detector') and self.regimes_detected:
                # Get current regime
                try:
                    current_regime = self.regime_detector.get_current_regime()
                    metrics['current_regime'] = current_regime['regime_label']
                except Exception as e:
                    logger.error(f"Error getting current regime: {e}")
                
                # Add regime stats
                try:
                    regime_stats = self.regime_detector.get_regime_stats()
                    # Select key metrics only to avoid overwhelming output
                    if not regime_stats.empty:
                        # Look for best and worst regimes by return
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
                except Exception as e:
                    logger.error(f"Error calculating regime stats: {e}")
            
            # Add metrics on regime-specific models if available
            if self.use_regime_specific_params and self.regime_models:
                metrics['regime_models'] = {
                    'count': len(self.regime_models),
                    'regimes_with_models': list(self.regime_models.keys())
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            # Return empty metrics dict if error occurs
            return {}
