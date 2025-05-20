#!/usr/bin/env python
"""
Fixed Regime Adaptive Strategy - Patched version

This script applies fixes to the RegimeAdaptiveStrategy class before running the strategy runner.
It addresses:
1. The missing backtest_engine attribute
2. The date ambiguity in the DataFrame (having it as both index and column)
3. Correct parameter handling for the BacktestEngine.run_backtest method
"""

import os
import sys
import pandas as pd
import logging
from src.strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.backtesting.engine import BacktestEngine

def apply_fixes():
    """
    Apply all necessary fixes to the RegimeAdaptiveStrategy class
    """
    print("Applying fixes to RegimeAdaptiveStrategy...")
    
    # Store original methods for reference
    original_initialize = RegimeAdaptiveStrategy.initialize
    original_backtest = RegimeAdaptiveStrategy.backtest
    original_predict = RegimeAdaptiveStrategy.predict
    original_generate_signals = RegimeAdaptiveStrategy.generate_signals
    
    # 1. Fix the initialize method to include backtest_engine
    def fixed_initialize(self, config):
        # Call the original initialize method
        original_initialize(self, config)
        
        # Initialize backtest_engine (add this to fix the missing attribute)
        self.backtest_engine = BacktestEngine(
            initial_capital=self.config.get('initial_capital', 100000.0),
            commission=self.config.get('commission', 0.001),
            slippage=self.config.get('slippage', 0.001)
        )
        print("Fixed: backtest_engine initialized")
    
    # 2. Fix the generate_signals method to prevent date ambiguity
    def fixed_generate_signals(self, features, predictions, dates):
        # Call the original generate_signals method
        signals = original_generate_signals(self, features, predictions, dates)
        
        # Fix date ambiguity if both in index and column
        if isinstance(signals, pd.DataFrame):
            if signals.index.name == 'date' and 'date' in signals.columns:
                signals = signals.reset_index(drop=True)
                print("Fixed: date ambiguity in signals DataFrame")
        
        return signals
    
    # 3. Fix the predict method to ensure signals have proper format
    def fixed_predict(self, test_data):
        # Call the original predict method
        signals, predictions = original_predict(self, test_data)
        
        # Fix date ambiguity if both in index and column
        if isinstance(signals, pd.DataFrame):
            if signals.index.name == 'date' and 'date' in signals.columns:
                signals = signals.reset_index(drop=True)
                print("Fixed: date ambiguity in signals DataFrame (predict)")
        
        return signals, predictions
    
    # 4. Fix the backtest method to handle parameters correctly
    def fixed_backtest(self, data, train_data=None, test_data=None):
        try:
            # Make sure backtest_engine is initialized
            if not hasattr(self, 'backtest_engine'):
                self.backtest_engine = BacktestEngine(
                    initial_capital=self.config.get('initial_capital', 100000.0),
                    commission=self.config.get('commission', 0.001),
                    slippage=self.config.get('slippage', 0.001)
                )
                print("Fixed: created missing backtest_engine")
            
            # Ensure regimes are detected for the entire dataset
            if not self.regimes_detected:
                logging.info("Detecting regimes for the full dataset")
                try:
                    self._detect_regimes(data)
                    
                    if self.regime_detector.regime_history is not None:
                        # Get unique regime labels
                        regime_labels = self.regime_detector.regime_history['regime_label'].dropna().unique().tolist()
                        self.known_regime_labels = sorted(regime_labels)
                        
                        # Set regime_data
                        relevant_cols = ['regime', 'regime_label']
                        if 'trend' in self.regime_detector.regime_history.columns:
                            relevant_cols.append('trend')
                        if 'vol_regime' in self.regime_detector.regime_history.columns:
                            relevant_cols.append('vol_regime')
                        
                        # Filter to existing columns
                        existing_relevant_cols = [col for col in relevant_cols if col in self.regime_detector.regime_history.columns]
                        if existing_relevant_cols:
                            self.regime_data = self.regime_detector.regime_history[existing_relevant_cols].copy()
                        else:
                            # Create minimal regime_data
                            self.regime_data = pd.DataFrame(index=data.index)
                            self.regime_data['regime'] = 0
                            self.regime_data['regime_label'] = 'neutral'
                except Exception as e:
                    logging.error(f"Error in regime detection during backtest: {e}")
                    # Create minimal regime data
                    self.regime_data = pd.DataFrame(index=data.index)
                    self.regime_data['regime'] = 0
                    self.regime_data['regime_label'] = 'neutral'
                    
                self.regimes_detected = True
                
            # Initialize regime-specific models if requested
            if self.use_regime_specific_params and train_data is not None:
                self._initialize_regime_specific_models(train_data)
            
            # Split data if not provided
            if train_data is None or test_data is None:
                train_size = int(len(data) * 0.7)
                train_data = data.iloc[:train_size]
                test_data = data.iloc[train_size:]
                
            # Generate features and train model
            train_features, train_target, train_dates = self.generate_features(train_data)
            
            # Train base model
            self.model_engine.train(train_features, train_target)
            
            # Generate features for test data
            test_features, test_target, test_dates = self.generate_features(test_data)
            
            # Generate predictions
            predictions = self.model_engine.predict(test_features)
            
            # Generate signals
            signals = self.generate_signals(test_features, predictions, test_dates)
            
            # Fix date ambiguity - CRITICAL FIX!
            if isinstance(signals, pd.DataFrame):
                if signals.index.name == 'date' and 'date' in signals.columns:
                    signals = signals.reset_index(drop=True)
                    print("Fixed: date ambiguity in signals DataFrame (backtest)")
                elif 'date' not in signals.columns:
                    # If date is missing, make sure it's not used for sorting
                    if hasattr(self.backtest_engine, 'run_backtest'):
                        # Monkey patch the run_backtest method temporarily
                        original_run_backtest = self.backtest_engine.run_backtest
                        
                        def safe_run_backtest(signals_df, data_dict):
                            # Add a check for the date column
                            if isinstance(signals_df, pd.DataFrame) and 'date' not in signals_df.columns and signals_df.index.name == 'date':
                                # Add the date column from the index
                                signals_df = signals_df.reset_index()
                            return original_run_backtest(signals_df, data_dict)
                        
                        # Apply the patch
                        self.backtest_engine.run_backtest = safe_run_backtest
                        print("Fixed: patched run_backtest to handle missing date column")
            
            # Create a dictionary with test data
            symbol = self.config.get('symbol', 'SPY')
            test_data_dict = {symbol: test_data}
            
            # Run backtest
            print("Running backtest with fixed parameters...")
            backtest_results = self.backtest_engine.run_backtest(signals, test_data_dict)
            
            # Add regime analysis to results
            self._add_regime_analysis(backtest_results, test_data)
            
            # Restore original run_backtest if patched
            if hasattr(self, 'backtest_engine') and hasattr(self.backtest_engine, 'original_run_backtest'):
                self.backtest_engine.run_backtest = self.backtest_engine.original_run_backtest
            
            return backtest_results
            
        except Exception as e:
            logging.error(f"Error in backtest: {e}")
            print(f"Falling back to fixed base backtest. Error was: {e}")
            
            # Ensure backtest_engine is initialized
            if not hasattr(self, 'backtest_engine'):
                self.backtest_engine = BacktestEngine(
                    initial_capital=self.config.get('initial_capital', 100000.0),
                    commission=self.config.get('commission', 0.001),
                    slippage=self.config.get('slippage', 0.001)
                )
            
            # Split data if not provided
            if train_data is None or test_data is None:
                train_size = int(len(data) * 0.7)
                train_data = data.iloc[:train_size]
                test_data = data.iloc[train_size:]
            
            # Train model
            from src.strategies.trend_following import TrendFollowingStrategy
            TrendFollowingStrategy.train(self, train_data)
            
            # Generate signals
            signals, predictions = self.predict(test_data)
            
            # Fix date ambiguity
            if isinstance(signals, pd.DataFrame):
                if signals.index.name == 'date' and 'date' in signals.columns:
                    print("Fixed: date ambiguity in fallback signals DataFrame")
                    signals = signals.reset_index(drop=True)
                elif 'date' not in signals.columns:
                    # If date is missing but is the index name, reset the index
                    if signals.index.name == 'date':
                        signals = signals.reset_index()
                        print("Fixed: added date column from index in fallback signals")
            
            # Create test data dict
            symbol = self.config.get('symbol', 'SPY')
            test_data_dict = {symbol: test_data}
            
            # Run backtest
            results = self.backtest_engine.run_backtest(signals, test_data_dict)
            
            # Store metrics
            self.metrics.update(results.get('performance', {}))
            
            return results
    
    # Apply the fixes
    RegimeAdaptiveStrategy.initialize = fixed_initialize
    RegimeAdaptiveStrategy.backtest = fixed_backtest
    RegimeAdaptiveStrategy.predict = fixed_predict
    RegimeAdaptiveStrategy.generate_signals = fixed_generate_signals
    
    print("All fixes applied to RegimeAdaptiveStrategy")

# Apply fixes when the module is imported
apply_fixes()

# If run directly, also run the strategy_runner
if __name__ == "__main__":
    print("\nRunning strategy_runner with fixed RegimeAdaptiveStrategy...")
    import strategy_runner
    strategy_runner.main()
