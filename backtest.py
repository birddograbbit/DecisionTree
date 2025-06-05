# backtest.py
"""
DEPRECATED: Legacy backtesting module.

⚠️  WARNING: This module is deprecated as of v0.2 ⚠️

This module contains legacy backtesting functions that are no longer
maintained. The modern trading system uses the engine-based architecture
in strategy_runner.py instead.

Migration Path:
1. Use strategy_runner.py for all new backtesting workflows
2. The modern approach provides:
   - Engine-based architecture
   - Strategy class framework
   - Hyperparameter optimization integration
   - Feature auditing capabilities
   - Regime-adaptive strategies

Legacy Functions Provided (with deprecation warnings):
- run_backtest(): Use strategy_runner.py --mode single instead
- run_walkforward_backtest(): Use strategy_runner.py with regime-adaptive strategies

For documentation on the modern approach, see:
- Decision_Tree_Classifier_Strategy.md
- strategy_runner.py --help
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from datetime import datetime

from src.data.preprocessing import load_ibkr_data, preprocess_data
from src.features.feature_engineering import prepare_train_test_data
from src.models.decision_tree import train_decision_tree, evaluate_model, load_model
from src.backtesting.signal_generation import generate_signals, apply_signal_rules
from src.backtesting.engine import BacktestEngine
from src.backtesting.performance import calculate_performance_metrics, plot_performance, compare_to_benchmark
import config


def _deprecation_warning(function_name, replacement):
    """Issue a deprecation warning for legacy functions."""
    warnings.warn(
        f"{function_name} is deprecated and will be removed in a future version. "
        f"Use {replacement} instead. See strategy_runner.py for the modern approach.",
        DeprecationWarning,
        stacklevel=3
    )


def run_backtest(model_path, data_path, output_dir='results', commission=0.0005, slippage=0.0001):
    """
    DEPRECATED: Run a backtest using a trained model and historical data.
    
    ⚠️  WARNING: This function is deprecated. Use strategy_runner.py instead.
    
    Migration example:
    Old: python backtest.py --model model.pkl --data data/raw --output results
    New: python strategy_runner.py --data data/raw --mode single --model xgboost --output results
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model file
    data_path : str
        Path to the historical data file
    output_dir : str, default='results'
        Directory to save results
    commission : float, default=0.0005
        Commission rate per trade
    slippage : float, default=0.0001
        Slippage rate per trade
        
    Returns:
    --------
    dict
        Backtest results
    """
    _deprecation_warning(
        "run_backtest()", 
        "strategy_runner.py --mode single"
    )
    
    print("🔄 DEPRECATED FUNCTION: Redirecting to modern backtesting approach...")
    print("📖 For full functionality, please use strategy_runner.py directly")
    print("📋 Example: python strategy_runner.py --data data/raw --mode single --model xgboost")
    
    # Provide basic legacy functionality for backward compatibility
    # This is a simplified version that maintains some compatibility
    # but encourages users to migrate to the modern approach
    
    print(f"Running legacy backtest with model: {model_path}")
    print(f"Using data: {data_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load model and scaler
        model, scaler = load_model(model_path)
        
        if model is None:
            print(f"Failed to load model from {model_path}")
            return None
        
        # Load data
        if os.path.isdir(data_path):
            # If data_path is a directory, look for SPY data files
            train_file = os.path.join(data_path, 'historical_data_STOCK_SPY_1_day2000-2009.csv')
            test_file = os.path.join(data_path, 'historical_data_STOCK_SPY_1_day2010-2025.csv')
            
            if not os.path.exists(train_file) or not os.path.exists(test_file):
                print(f"Data files not found in {data_path}")
                return None
            
            # Load and combine data
            df = load_ibkr_data(train_file, test_file)
        else:
            # If data_path is a file, load it directly
            df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        
        if df is None or df.empty:
            print("Failed to load data")
            return None
        
        # Preprocess data
        df = preprocess_data(df)
        
        print(f"Data loaded and preprocessed. Shape: {df.shape}")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Split data into training and testing periods
        train_end_date = '2009-12-31'
        test_data = df[df.index > train_end_date]
        
        print(f"Test data shape: {test_data.shape}")
        print(f"Test date range: {test_data.index.min()} to {test_data.index.max()}")
        
        # Generate features for test data
        print("Generating features for test data...")
        _, X_test, _, y_test, _, dates_test, _ = prepare_train_test_data(test_data, None)
        
        print(f"Test features shape: {X_test.shape}")
        
        # Generate signals
        print("Generating trading signals...")
        signals = generate_signals(model, X_test, dates_test)
        
        # Apply signal rules
        signals = apply_signal_rules(signals, consecutive_buys=False)
        
        print(f"Generated {len(signals)} signals")
        print(f"Buy signals: {(signals['signal'] == 1).sum()}")
        print(f"Sell signals: {(signals['signal'] == -1).sum()}")
        print(f"Hold signals: {(signals['signal'] == 0).sum()}")
        
        # Run backtest
        print("Running backtest...")
        backtest = BacktestEngine(initial_capital=config.INITIAL_CAPITAL, 
                                  commission=commission, 
                                  slippage=slippage)
        backtest_results = backtest.run_backtest(signals, test_data)
        
        # Print performance summary
        performance = backtest_results['performance']
        
        print("\nPerformance Summary:")
        print(f"Total Return: {performance['total_return']:.2%}")
        print(f"Annualized Return: {performance['ann_return']:.2%}")
        print(f"Annualized Volatility: {performance['ann_volatility']:.2%}")
        print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {performance['max_drawdown']:.2%}")
        print(f"CAGR/Max DD Ratio: {performance['cagr_dd_ratio']:.2f}")
        print(f"Win Rate: {performance['win_rate']:.2%}")
        print(f"Number of Trades: {performance['num_trades']}")
        
        # Compare to benchmark (SPY buy and hold)
        print("\nComparing to benchmark (SPY buy and hold)...")
        comparison = compare_to_benchmark(backtest_results, df[df.index > train_end_date])
        
        print(f"Strategy Return: {comparison['strategy_return']:.2%}, Benchmark Return: {comparison['benchmark_return']:.2%}")
        print(f"Outperformance: {comparison['outperformance']:.2%}")
        
        # Save basic results
        print(f"\nSaving legacy backtest results to {output_dir}")
        
        # Save performance metrics
        with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as f:
            f.write("Performance Summary (Legacy Backtest):\n")
            f.write("⚠️  Generated by deprecated backtest.py - migrate to strategy_runner.py\n\n")
            for key, value in performance.items():
                if isinstance(value, (int, float)):
                    if key.endswith('_rate') or key in ['total_return', 'ann_return', 'ann_volatility', 'max_drawdown']:
                        f.write(f"{key}: {value:.2%}\n")
                    else:
                        f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        # Save trades
        if not backtest_results['trades'].empty:
            backtest_results['trades'].to_csv(os.path.join(output_dir, 'trades.csv'))
        
        # Save equity curve
        backtest_results['equity_curve'].to_csv(os.path.join(output_dir, 'equity_curve.csv'))
        
        # Save signals
        signals.to_csv(os.path.join(output_dir, 'signals.csv'))
        
        print(f"Legacy backtest completed. Results saved to {output_dir}")
        print("\n🔄 To access full functionality, please migrate to strategy_runner.py")
        
        return backtest_results
        
    except Exception as e:
        print(f"Error in legacy backtest: {e}")
        print("🔄 For better error handling and full functionality, please use strategy_runner.py")
        return None


def run_walkforward_backtest(data_path, output_dir='results_walkforward',
                             train_size=252*5, test_size=126, step_size=63,
                             max_depth=5, min_samples_split=5):
    """
    DEPRECATED: Run a walkforward backtest.
    
    ⚠️  WARNING: This function is deprecated. Use strategy_runner.py with regime-adaptive strategies instead.
    
    Migration example:
    Old: python backtest.py --walkforward --data data/raw
    New: python strategy_runner.py --data data/raw --strategy regime_adaptive --model random_forest
    
    Parameters:
    -----------
    data_path : str
        Path to the historical data file
    output_dir : str, default='results_walkforward'
        Directory to save results
    train_size : int, default=252*5
        Size of training window in days
    test_size : int, default=126
        Size of testing window in days
    step_size : int, default=63
        Step size in days for moving the window forward
    max_depth : int, default=5
        Maximum depth of the decision tree
    min_samples_split : int, default=5
        Minimum samples required to split an internal node
        
    Returns:
    --------
    dict
        Walkforward backtest results
    """
    _deprecation_warning(
        "run_walkforward_backtest()", 
        "strategy_runner.py --strategy regime_adaptive"
    )
    
    print("🔄 DEPRECATED FUNCTION: Walk-forward functionality is better handled by regime-adaptive strategies")
    print("📖 For modern walk-forward equivalent, use strategy_runner.py with regime-adaptive strategies")
    print("📋 Example: python strategy_runner.py --data data/raw --strategy regime_adaptive --model random_forest")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save a migration note
    with open(os.path.join(output_dir, 'MIGRATION_NOTE.txt'), 'w') as f:
        f.write("Walk-Forward Backtesting Migration\n")
        f.write("===================================\n\n")
        f.write("This legacy walk-forward backtest function has been deprecated.\n\n")
        f.write("The modern approach uses regime-adaptive strategies which provide:\n")
        f.write("- Dynamic adaptation to market conditions\n")
        f.write("- Automatic retraining based on regime detection\n")
        f.write("- Better integration with hyperparameter optimization\n")
        f.write("- Professional engine-based architecture\n\n")
        f.write("To migrate your workflow:\n")
        f.write("python strategy_runner.py --data data/raw --strategy regime_adaptive --model random_forest\n\n")
        f.write("For more information, see Decision_Tree_Classifier_Strategy.md\n")
    
    print(f"Migration note saved to {output_dir}/MIGRATION_NOTE.txt")
    print("🔄 Please migrate to the modern strategy_runner.py approach for full functionality")
    
    return {
        'status': 'deprecated',
        'message': 'Use strategy_runner.py with regime-adaptive strategies instead',
        'migration_example': 'python strategy_runner.py --data data/raw --strategy regime_adaptive --model random_forest'
    }


# Legacy command-line interface (deprecated)
if __name__ == "__main__":
    import argparse
    
    print("⚠️  WARNING: backtest.py command-line interface is deprecated")
    print("🔄 Please use strategy_runner.py instead for full functionality")
    print("📋 Example: python strategy_runner.py --data data/raw --mode single --model xgboost")
    print()
    
    parser = argparse.ArgumentParser(description='DEPRECATED: Use strategy_runner.py instead')
    parser.add_argument('--model', type=str, default='data/models/SPY_decision_tree.pkl',
                        help='Path to the trained model file')
    parser.add_argument('--data', type=str, default='data/raw',
                        help='Path to the historical data directory or file')
    parser.add_argument('--output', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--walkforward', action='store_true',
                        help='Run walkforward backtest instead of standard backtest')
    parser.add_argument('--train-size', type=int, default=252*5,
                        help='Size of training window in days for walkforward backtest')
    parser.add_argument('--test-size', type=int, default=126,
                        help='Size of testing window in days for walkforward backtest')
    parser.add_argument('--step-size', type=int, default=63,
                        help='Step size in days for moving the window forward in walkforward backtest')
    parser.add_argument('--max-depth', type=int, default=5,
                        help='Maximum depth of the decision tree for walkforward backtest')
    parser.add_argument('--min-samples-split', type=int, default=5,
                        help='Minimum samples required to split an internal node for walkforward backtest')
    args = parser.parse_args()
    
    if args.walkforward:
        # Run walkforward backtest
        results = run_walkforward_backtest(
            data_path=args.data,
            output_dir=args.output,
            train_size=args.train_size,
            test_size=args.test_size,
            step_size=args.step_size,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split
        )
    else:
        # Run standard backtest
        results = run_backtest(
            model_path=args.model,
            data_path=args.data,
            output_dir=args.output
        )
