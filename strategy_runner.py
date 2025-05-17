#!/usr/bin/env python
"""
Strategy runner for decision tree-based trading strategies.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from src.data.data_acquisition import load_historical_data
from src.features.feature_engineering import create_features
from src.features.regime_detection import RegimeDetector, plot_regime_shifts
from src.models.model_factory import ModelFactory
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.backtesting.engine import BacktestingEngine
from src.backtesting.performance import calculate_performance_metrics

def setup_output_directory(output_dir):
    """
    Create output directory if it doesn't exist.
    
    Parameters:
    -----------
    output_dir : str
        Path to output directory
        
    Returns:
    --------
    str
        Absolute path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    return os.path.abspath(output_dir)

def run_single_strategy(data_path, model_type='random_forest', output_dir='results',
                       train_end_date=None, symbol='SPY', strategy_type='trend_following',
                       calibrate=False):
    """
    Run a single strategy with specified parameters.
    
    Parameters:
    -----------
    data_path : str
        Path to the historical data file or directory
    model_type : str, default='random_forest'
        Type of model to use ('decision_tree', 'random_forest', 'xgboost', 'stacking')
    output_dir : str, default='results'
        Directory to save results
    train_end_date : str or None, default=None
        End date for training data (e.g., '2020-12-31')
        If None, uses 70% of data for training
    symbol : str, default='SPY'
        Trading symbol
    strategy_type : str, default='trend_following'
        Type of strategy to use ('trend_following', 'regime_adaptive')
    calibrate : bool, default=False
        Whether to use probability calibration for tree-based models
        
    Returns:
    --------
    dict
        Dictionary with backtest results and performance metrics
    """
    # Load and prepare data
    print(f"Loading data from {data_path} for {symbol}")
    df = load_historical_data(data_path, symbol)
    
    # Calculate features
    print("Calculating features...")
    df = create_features(df)
    
    # Prepare training/testing split
    if train_end_date is not None:
        train_end_idx = df.index.get_loc(train_end_date, method='nearest')
    else:
        train_end_idx = int(len(df) * 0.7)
    
    df_train = df.iloc[:train_end_idx].copy()
    df_test = df.iloc[train_end_idx:].copy()
    
    print(f"Training data: {df_train.index[0]} to {df_train.index[-1]} ({len(df_train)} samples)")
    print(f"Testing data: {df_test.index[0]} to {df_test.index[-1]} ({len(df_test)} samples)")
    
    # Prepare model
    print(f"Creating {model_type} model...")
    
    # Handle special case for stacking model
    if model_type == 'stacking':
        # Create default stacking configuration
        base_models = []
        
        # Add Decision Tree and Random Forest as base models
        base_models.append({'model_type': 'decision_tree', 'model_params': {'max_depth': 5, 'calibrate': calibrate}})
        base_models.append({'model_type': 'random_forest', 'model_params': {'n_estimators': 100, 'max_depth': 5, 'calibrate': calibrate}})
        
        # Add XGBoost if available
        if 'xgboost' in ModelFactory.get_available_models():
            base_models.append({'model_type': 'xgboost', 'model_params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}})
        
        # Create model
        model = ModelFactory.create_model(model_type, base_models=base_models)
    else:
        # Create standard model
        model = ModelFactory.create_model(model_type, calibrate=calibrate)
    
    # Prepare strategy
    print(f"Creating {strategy_type} strategy...")
    if strategy_type == 'trend_following':
        strategy = TrendFollowingStrategy(model=model, probability_threshold=0.55)
    elif strategy_type == 'regime_adaptive':
        # For regime-adaptive, we need to detect regimes first
        print("Detecting market regimes...")
        regime_detector = RegimeDetector()
        regimes = regime_detector.detect_regimes(df, method='hmm')
        
        # Visualize regime shifts
        regime_fig = plot_regime_shifts(df, regimes)
        
        # Create strategies for different regimes
        regime_models = {}
        for regime in np.unique(regimes):
            if model_type == 'stacking':
                # Create specialized stacking model for each regime
                regime_model = ModelFactory.create_model(model_type, base_models=base_models)
            else:
                # Create standard model for each regime
                regime_model = ModelFactory.create_model(model_type, calibrate=calibrate)
            regime_models[regime] = regime_model
        
        # Create adaptive strategy
        strategy = RegimeAdaptiveStrategy(
            regime_models=regime_models,
            regimes=regimes,
            probability_thresholds={0: 0.55, 1: 0.6, 2: 0.52}  # Example thresholds per regime
        )
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")
    
    # Set up backtesting engine
    backtester = BacktestingEngine(strategy)
    
    # Run backtest
    print("Running backtest...")
    results = backtester.run_backtest(
        df_train, df_test,
        features_cols=[col for col in df.columns if 'feature_' in col or 'indicator_' in col],
        target_col='target',
        price_col='close',
        date_col='date',
        report=True
    )
    
    # Calculate performance metrics
    metrics = calculate_performance_metrics(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{symbol}_{model_type}_{strategy_type}_{timestamp}"
    
    # Create output directory
    output_path = setup_output_directory(output_dir)
    
    # Save performance metrics
    metrics_file = os.path.join(output_path, f"{base_filename}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save backtest results
    results_file = os.path.join(output_path, f"{base_filename}_results.csv")
    results.to_csv(results_file)
    
    # Save backtesting visualization
    fig = backtester.plot_results(title=f"{symbol} - {model_type} - {strategy_type}")
    fig_file = os.path.join(output_path, f"{base_filename}_backtest.png")
    fig.savefig(fig_file)
    
    # Save regime visualization if applicable
    if strategy_type == 'regime_adaptive' and 'regime_fig' in locals():
        regime_fig_file = os.path.join(output_path, f"{base_filename}_regimes.png")
        regime_fig.savefig(regime_fig_file)
    
    print(f"Results saved to {output_path}")
    
    # Display key metrics
    print("\nPerformance Metrics:")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"CAGR: {metrics['cagr']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"CAGR/Max DD: {metrics['cagr_maxdd_ratio']:.2f}")
    
    return {
        'results': results,
        'metrics': metrics,
        'model': model,
        'strategy': strategy
    }

def run_strategy_comparison(data_path, output_dir='results_comparison', 
                           train_end_date=None, symbol='SPY'):
    """
    Run comparison of different strategies/models.
    
    Parameters:
    -----------
    data_path : str
        Path to the historical data file or directory
    output_dir : str, default='results_comparison'
        Directory to save results
    train_end_date : str or None, default=None
        End date for training data (e.g., '2020-12-31')
        If None, uses 70% of data for training
    symbol : str, default='SPY'
        Trading symbol
        
    Returns:
    --------
    dict
        Comparison results
    """
    # Create output directory
    output_path = setup_output_directory(output_dir)
    
    # Models to compare
    models_to_compare = [
        {'name': 'Decision Tree', 'type': 'decision_tree', 'strategy': 'trend_following', 'calibrate': False},
        {'name': 'Decision Tree (Calibrated)', 'type': 'decision_tree', 'strategy': 'trend_following', 'calibrate': True},
        {'name': 'Random Forest', 'type': 'random_forest', 'strategy': 'trend_following', 'calibrate': False},
        {'name': 'Random Forest (Calibrated)', 'type': 'random_forest', 'strategy': 'trend_following', 'calibrate': True},
        {'name': 'XGBoost', 'type': 'xgboost', 'strategy': 'trend_following', 'calibrate': False},
        {'name': 'Stacking Ensemble', 'type': 'stacking', 'strategy': 'trend_following', 'calibrate': False},
        {'name': 'Regime-Adaptive RF', 'type': 'random_forest', 'strategy': 'regime_adaptive', 'calibrate': False},
        {'name': 'Regime-Adaptive RF (Calibrated)', 'type': 'random_forest', 'strategy': 'regime_adaptive', 'calibrate': True},
    ]
    
    # Check which models are available
    available_models = ModelFactory.get_available_models()
    models_to_compare = [m for m in models_to_compare if m['type'] in available_models]
    
    # Run each model
    results = {}
    metrics_summary = []
    
    for model_info in models_to_compare:
        print(f"\n{'='*50}")
        print(f"Running {model_info['name']}...")
        print(f"{'='*50}")
        
        try:
            model_results = run_single_strategy(
                data_path,
                model_type=model_info['type'],
                output_dir=os.path.join(output_path, model_info['name'].replace(' ', '_')),
                train_end_date=train_end_date,
                symbol=symbol,
                strategy_type=model_info['strategy'],
                calibrate=model_info['calibrate']
            )
            
            results[model_info['name']] = model_results
            
            # Add to metrics summary
            metrics = model_results['metrics']
            metrics_summary.append({
                'Model': model_info['name'],
                'Total Return': f"{metrics['total_return']:.2%}",
                'CAGR': f"{metrics['cagr']:.2%}",
                'Sharpe': f"{metrics['sharpe_ratio']:.2f}",
                'Max DD': f"{metrics['max_drawdown']:.2%}",
                'Win Rate': f"{metrics['win_rate']:.2%}",
                'CAGR/Max DD': f"{metrics['cagr_maxdd_ratio']:.2f}",
                'Trades': len(model_results['results'][model_results['results']['position'] != 0])
            })
            
        except Exception as e:
            print(f"Error running {model_info['name']}: {str(e)}")
    
    # Create comparison table
    if metrics_summary:
        comparison_df = pd.DataFrame(metrics_summary)
        
        # Save comparison table
        comparison_file = os.path.join(output_path, f"{symbol}_comparison.csv")
        comparison_df.to_csv(comparison_file, index=False)
        
        # Display comparison table
        print("\nStrategy Comparison:")
        print(comparison_df.to_string(index=False))
    else:
        print("No results to compare.")
    
    return results

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run trading strategies')
    
    parser.add_argument('--data', type=str, required=True,
                        help='Path to historical data file or directory')
    
    parser.add_argument('--mode', type=str, choices=['single', 'compare'], default='single',
                        help='Mode: single strategy or comparison (default: single)')
    
    parser.add_argument('--model', type=str, choices=['decision_tree', 'random_forest', 'xgboost', 'stacking'],
                        default='random_forest',
                        help='Model type for single mode (default: random_forest)')
    
    parser.add_argument('--strategy', type=str, choices=['trend_following', 'regime_adaptive'],
                        default='trend_following',
                        help='Strategy type (default: trend_following)')
    
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory (default: results)')
    
    parser.add_argument('--symbol', type=str, default='SPY',
                        help='Trading symbol (default: SPY)')
    
    parser.add_argument('--train-end', type=str, default=None,
                        help='End date for training data (e.g., 2020-12-31)')
    
    parser.add_argument('--calibrate', action='store_true',
                        help='Use probability calibration for tree-based models')
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_arguments()
    
    if args.mode == 'single':
        run_single_strategy(
            data_path=args.data,
            model_type=args.model,
            output_dir=args.output,
            train_end_date=args.train_end,
            symbol=args.symbol,
            strategy_type=args.strategy,
            calibrate=args.calibrate
        )
    elif args.mode == 'compare':
        run_strategy_comparison(
            data_path=args.data,
            output_dir=args.output,
            train_end_date=args.train_end,
            symbol=args.symbol
        )
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()