# backtest.py
"""
Script to run backtesting of the decision tree trading strategy.
"""

import os
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


def run_backtest(model_path, data_path, output_dir='results', threshold=0.65, commission=0.0005, slippage=0.0001):
    """
    Run a backtest using a trained model and historical data.
    
    Parameters:
    -----------
    model_path : str
        Path to the trained model file
    data_path : str
        Path to the historical data file
    output_dir : str, default='results'
        Directory to save results
    threshold : float, default=0.65
        Probability threshold for generating signals
    commission : float, default=0.0005
        Commission rate per trade
    slippage : float, default=0.0001
        Slippage rate per trade
        
    Returns:
    --------
    dict
        Backtest results
    """
    print(f"Running backtest with model: {model_path}")
    print(f"Using data: {data_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
    signals = generate_signals(model, X_test, dates_test, threshold=threshold)
    
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
    print(f"Strategy CAGR: {comparison['strategy_cagr']:.2%}, Benchmark CAGR: {comparison['benchmark_cagr']:.2%}")
    print(f"Strategy Max DD: {comparison['strategy_max_dd']:.2%}, Benchmark Max DD: {comparison['benchmark_max_dd']:.2%}")
    print(f"Strategy CAGR/DD: {comparison['strategy_cagr_dd']:.2f}, Benchmark CAGR/DD: {comparison['benchmark_cagr_dd']:.2f}")
    print(f"Alpha: {comparison['alpha']:.2%}")
    print(f"Beta: {comparison['beta']:.2f}")
    
    # Plot performance
    print("\nGenerating performance visualizations...")
    figures = plot_performance(backtest_results, df[df.index > train_end_date], save_dir=output_dir)
    
    # Save backtest results
    print(f"\nSaving backtest results to {output_dir}")
    
    # Save performance metrics
    with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as f:
        f.write("Performance Summary:\n")
        for key, value in performance.items():
            if isinstance(value, (int, float)):
                if key.endswith('_rate') or key in ['total_return', 'ann_return', 'ann_volatility', 'max_drawdown']:
                    f.write(f"{key}: {value:.2%}\n")
                else:
                    f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\nBenchmark Comparison:\n")
        for key, value in comparison.items():
            if isinstance(value, (int, float)):
                if key.endswith('_return') or key.endswith('_dd') or key in ['alpha', 'outperformance']:
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
    
    print(f"Backtest completed. Results saved to {output_dir}")
    
    return backtest_results


def run_walkforward_backtest(data_path, output_dir='results_walkforward', 
                             train_size=252*5, test_size=126, step_size=63,
                             threshold=0.65, max_depth=5, min_samples_split=5):
    """
    Run a walkforward backtest.
    
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
    threshold : float, default=0.65
        Probability threshold for generating signals
    max_depth : int, default=5
        Maximum depth of the decision tree
    min_samples_split : int, default=5
        Minimum samples required to split an internal node
        
    Returns:
    --------
    dict
        Walkforward backtest results
    """
    from src.backtesting.engine import WalkforwardBacktester
    
    print(f"Running walkforward backtest...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # Define model factory function
    def model_factory(X_train, y_train):
        return train_decision_tree(X_train, y_train, max_depth=max_depth, min_samples_split=min_samples_split)
    
    # Define feature engineering function (adapt to match your implementation)
    def feature_engineer(data):
        # Get the correct indices for X_test_scaled, y_test, and dates_test
        results = prepare_train_test_data(data, None)
        return results[1], results[3], results[5]  # X_test_scaled, y_test, dates_test
    
    # Create walkforward backtester
    backtester = WalkforwardBacktester(
        data=df,
        model_factory=model_factory,
        feature_engineer=feature_engineer,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        initial_capital=config.INITIAL_CAPITAL,
        commission=config.COMMISSION_RATE,
        slippage=config.SLIPPAGE_RATE
    )
    
    # Run walkforward backtest
    results = backtester.run(signal_threshold=threshold)
    
    # Print performance summary
    performance = results['performance']
    
    print("\nOverall Performance Summary:")
    print(f"Total Return: {performance['total_return']:.2%}")
    print(f"Annualized Return: {performance['ann_return']:.2%}")
    print(f"Annualized Volatility: {performance['ann_volatility']:.2%}")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Maximum Drawdown: {performance['max_drawdown']:.2%}")
    print(f"CAGR/Max DD Ratio: {performance['cagr_dd_ratio']:.2f}")
    print(f"Win Rate: {performance['win_rate']:.2%}")
    print(f"Number of Trades: {performance['num_trades']}")
    
    # Print window results
    print("\nWindow Results:")
    for i, window in enumerate(results['window_results']):
        print(f"Window {i+1}:")
        print(f"  Train: {window['train_start']} to {window['train_end']}")
        print(f"  Test: {window['test_start']} to {window['test_end']}")
        print(f"  Total Return: {window['performance']['total_return']:.2%}")
        print(f"  Win Rate: {window['performance']['win_rate']:.2%}")
        print(f"  Number of Trades: {window['performance']['num_trades']}")
    
    # Plot overall equity curve
    plt.figure(figsize=(12, 6))
    
    # Use matplotlib.dates for proper date handling
    import matplotlib.dates as mdates
    from matplotlib.ticker import AutoLocator
    
    # Convert dates to numeric for plotting
    date_nums = mdates.date2num(results['equity_curve'].index.to_pydatetime())
    plt.plot(date_nums, results['equity_curve']['equity'])
    
    # Format x-axis with dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(AutoLocator())
    plt.xticks(rotation=45)
    
    plt.title('Walkforward Backtest Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'walkforward_equity_curve.png'))
    
    # Save results
    print(f"\nSaving walkforward backtest results to {output_dir}")
    
    # Save performance metrics
    with open(os.path.join(output_dir, 'walkforward_performance.txt'), 'w') as f:
        f.write("Overall Performance Summary:\n")
        for key, value in performance.items():
            if isinstance(value, (int, float)):
                if key.endswith('_rate') or key in ['total_return', 'ann_return', 'ann_volatility', 'max_drawdown']:
                    f.write(f"{key}: {value:.2%}\n")
                else:
                    f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        f.write("\nWindow Results:\n")
        for i, window in enumerate(results['window_results']):
            f.write(f"Window {i+1}:\n")
            f.write(f"  Train: {window['train_start']} to {window['train_end']}\n")
            f.write(f"  Test: {window['test_start']} to {window['test_end']}\n")
            for key, value in window['performance'].items():
                if isinstance(value, (int, float)):
                    if key.endswith('_rate') or key in ['total_return', 'ann_return', 'ann_volatility', 'max_drawdown']:
                        f.write(f"  {key}: {value:.2%}\n")
                    else:
                        f.write(f"  {key}: {value:.4f}\n")
    
    # Save equity curve
    results['equity_curve'].to_csv(os.path.join(output_dir, 'walkforward_equity_curve.csv'))
    
    # Save trades
    if not results['trades'].empty:
        results['trades'].to_csv(os.path.join(output_dir, 'walkforward_trades.csv'))
    
    print(f"Walkforward backtest completed. Results saved to {output_dir}")
    
    return results


# backtest.py (continued)
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run a backtest of the decision tree trading strategy')
    parser.add_argument('--model', type=str, default='data/models/SPY_decision_tree.pkl',
                        help='Path to the trained model file')
    parser.add_argument('--data', type=str, default='data/raw',
                        help='Path to the historical data directory or file')
    parser.add_argument('--output', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--threshold', type=float, default=0.65,
                        help='Probability threshold for generating signals')
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
            threshold=args.threshold,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split
        )
    else:
        # Run standard backtest
        results = run_backtest(
            model_path=args.model,
            data_path=args.data,
            output_dir=args.output,
            threshold=args.threshold
        )