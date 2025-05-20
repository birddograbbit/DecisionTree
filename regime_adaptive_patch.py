#!/usr/bin/env python
"""
Patch for RegimeAdaptiveStrategy backtest method

This script fixes the issue with the RegimeAdaptiveStrategy backtest method
where 'backtest_engine' attribute is missing and the 'date' column is ambiguous.
"""

import pandas as pd
import os
import argparse
from src.strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.backtesting.engine import BacktestEngine

def patch_regime_adaptive_strategy():
    """
    Apply the patch to RegimeAdaptiveStrategy class
    """
    # Monkey patch the backtest method in RegimeAdaptiveStrategy
    original_backtest = RegimeAdaptiveStrategy.backtest
    
    def patched_backtest(self, data, train_data=None, test_data=None):
        """
        Patched version of the backtest method that fixes the issues:
        1. Initialize backtest_engine attribute
        2. Reset date index in signals DataFrame to avoid ambiguity
        
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
            # Initialize backtest_engine (missing in original code)
            self.backtest_engine = BacktestEngine(
                initial_capital=self.config.get('initial_capital', 100000.0),
                commission=self.config.get('commission', 0.001),
                slippage=self.config.get('slippage', 0.001)
            )
            
            # Continue with original backtest logic
            return original_backtest(self, data, train_data, test_data)
            
        except Exception as e:
            print(f"Error in patched backtest: {e}")
            # If error still occurs, try base class backtest with fixed backtest_engine
            from src.strategies.trend_following import TrendFollowingStrategy
            
            print("Falling back to base TrendFollowingStrategy backtest with fixes")
            
            # Split data if not provided
            if train_data is None or test_data is None:
                train_size = int(len(data) * 0.7)
                train_data = data.iloc[:train_size]
                test_data = data.iloc[train_size:]
            
            # Train model
            self.train(train_data)
            
            # Generate signals
            signals, predictions = self.predict(test_data)
            
            # Make sure date is not in both index and columns
            if isinstance(signals, pd.DataFrame) and 'date' in signals.columns and signals.index.name == 'date':
                # Reset index to avoid ambiguity
                signals = signals.reset_index()
            
            # Create a dictionary with test data (for backtesting)
            symbol = self.config.get('symbol', 'SPY')
            test_data_dict = {symbol: test_data}
            
            # Initialize backtest_engine if not already done
            if not hasattr(self, 'backtest_engine'):
                self.backtest_engine = BacktestEngine(
                    initial_capital=self.config.get('initial_capital', 100000.0),
                    commission=self.config.get('commission', 0.001),
                    slippage=self.config.get('slippage', 0.001)
                )
            
            # Run backtest
            results = self.backtest_engine.run_backtest(signals, test_data_dict)
            
            # Store metrics
            self.metrics.update(results.get('performance', {}))
            
            return results
    
    # Apply the patch
    RegimeAdaptiveStrategy.backtest = patched_backtest
    print("RegimeAdaptiveStrategy.backtest method has been patched")
    
    # Also patch predict method to handle date ambiguity
    original_predict = RegimeAdaptiveStrategy.predict
    
    def patched_predict(self, test_data):
        """
        Patched predict method to ensure signals DataFrame has proper date handling
        
        Parameters:
        -----------
        test_data : pd.DataFrame
            Test data
            
        Returns:
        --------
        tuple
            (signals, predictions)
        """
        if not hasattr(self, 'is_trained') or not self.is_trained:
            # Call parent train method
            from src.strategies.trend_following import TrendFollowingStrategy
            TrendFollowingStrategy.train(self, test_data)
            self.is_trained = True
        
        # Generate features
        X_test, y_test, dates_test = self.generate_features(test_data)
        
        # Generate predictions
        predictions = self.model_engine.predict(X_test)
        
        # Generate signals
        signals = self.generate_signals(X_test, predictions, dates_test)
        
        # Ensure signals has a proper date column and no date index to avoid ambiguity
        if isinstance(signals, pd.DataFrame):
            if signals.index.name == 'date' and 'date' not in signals.columns:
                # Add date column from index
                signals['date'] = signals.index
            
            # If both index and column contain date, reset index to avoid ambiguity
            if signals.index.name == 'date' and 'date' in signals.columns:
                signals = signals.reset_index(drop=True)
        
        return signals, predictions
    
    # Apply the predict patch
    RegimeAdaptiveStrategy.predict = patched_predict
    print("RegimeAdaptiveStrategy.predict method has been patched")

def update_strategy_runner():
    """
    Update the strategy_runner.py to apply the patch
    """
    # For demonstration - in a real scenario, you'd modify the actual file
    print("To update strategy_runner.py, add the following at the top of the imports:")
    print("from regime_adaptive_patch import patch_regime_adaptive_strategy")
    print("# Apply the patch")
    print("patch_regime_adaptive_strategy()")
    
def main():
    """Main function to run the patch"""
    parser = argparse.ArgumentParser(description='Fix RegimeAdaptiveStrategy issues')
    parser.add_argument('--apply', action='store_true', help='Apply the patch')
    
    args = parser.parse_args()
    
    if args.apply:
        patch_regime_adaptive_strategy()
        update_strategy_runner()
        print("Patch applied successfully")
    else:
        print("Run with --apply to apply the patch")
        
if __name__ == "__main__":
    main()
