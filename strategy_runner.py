#!/usr/bin/env python
"""
Strategy Runner - Script for running and comparing different trading strategies.

This script provides functionality to:
1. Run a single strategy with specific parameters
2. Compare multiple strategies or configurations
3. Visualize and save results
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.data.preprocessing import preprocess_data
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.models.model_factory import ModelFactory
from src.models.hyperparameter_manager import HyperparameterManager
from strategy_configs import STRATEGY_CONFIGS

def load_data(data_path, symbol='SPY'):
    """
    Load and preprocess historical price data.
    
    Parameters:
    -----------
    data_path : str
        Path to the data file or directory
    symbol : str, default='SPY'
        Symbol for the data
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed price data
    """
    # Check if the path points to a file or directory
    if os.path.isfile(data_path):
        # Load data from single file
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        # Look for CSV files in the directory
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and symbol in f]
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found for {symbol} in {data_path}")
        
        # Load and concatenate all matching files
        dfs = []
        for file in csv_files:
            file_path = os.path.join(data_path, file)
            dfs.append(pd.read_csv(file_path, index_col=0, parse_dates=True))
        
        df = pd.concat(dfs)
        
        # Sort by date
        df = df.sort_index()
    
    # Preprocess data
    df = preprocess_data(df)
    
    print(f"Data loaded and preprocessed. Shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    return df

def run_strategy_comparison(data_path, output_dir='results_comparison', 
                           train_end_date=None, symbol='SPY', use_optimized_params=False):
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
    use_optimized_params : bool, default=False
        Whether to use optimized hyperparameters for models
        
    Returns:
    --------
    dict
        Comparison results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(data_path, symbol=symbol)
    
    # Define training and testing periods
    if train_end_date is not None:
        train_end_date = pd.to_datetime(train_end_date)
        train_data = df[df.index <= train_end_date]
        test_data = df[df.index > train_end_date]
    else:
        # Use 70% of data for training
        train_size = int(len(df) * 0.7)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
    
    print(f"Training data: {len(train_data)} rows ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"Testing data: {len(test_data)} rows ({test_data.index[0]} to {test_data.index[-1]})")
    
    # Get strategy configurations
    strategy_configs = get_strategy_configs(use_optimized_params)
    
    # Set symbol for all configs
    for config in strategy_configs:
        config['symbol'] = symbol
        if use_optimized_params:
            config['use_optimized'] = True
    
    # Run strategies
    results = {}
    equity_curves = []
    metrics_list = []

    for config in strategy_configs:
        print(f"\nRunning strategy: {config['name']}")

        # Initialize strategy based on name
        if 'Regime Adaptive' in config['name']:
            strategy = RegimeAdaptiveStrategy()
        else:
            strategy = TrendFollowingStrategy()
            
        strategy.initialize(config)

        # Run backtest
        backtest_results = strategy.backtest(df, train_data, test_data)

        # Store results
        results[config['name']] = backtest_results
        
        # Check if equity curve exists
        if 'equity_curve' in backtest_results and 'equity' in backtest_results['equity_curve']:
            equity_curves.append((config['name'], backtest_results['equity_curve']['equity']))

        # Get all metrics
        metrics = {
            'name': config['name'],
            'model_type': config['model_type']
        }
        metrics.update(strategy.get_metrics())
        metrics_list.append(metrics)

        # Print performance summary
        performance = backtest_results['performance']
        print(f"Total Return: {performance.get('total_return', 0):.2%}")
        print(f"CAGR: {performance.get('ann_return', 0):.2%}")
        print(f"Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
        print(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
        print(f"CAGR/Max DD: {performance.get('cagr_dd_ratio', 0):.2f}")
        print(f"Win Rate: {performance.get('win_rate', 0):.2%}")
        print(f"Number of Trades: {performance.get('num_trades', 0)}")

    # Plot equity curves if we have any
    if equity_curves:
        plt.figure(figsize=(12, 8))

        # Add buy and hold equity curve for reference
        buy_hold = (test_data['close'] / test_data['close'].iloc[0])
        plt.plot(buy_hold.index, buy_hold, label='Buy & Hold', linestyle='--')

        # Plot strategy equity curves
        for name, equity in equity_curves:
            # Normalize to start at 1.0
            normalized_equity = equity / equity.iloc[0]
            plt.plot(equity.index, normalized_equity, label=name)

        plt.title('Equity Curves Comparison')
        plt.xlabel('Date')
        plt.ylabel('Equity (normalized)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'equity_curves_comparison.png'))

    # Create metrics DataFrame
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.to_csv(os.path.join(output_dir, 'strategy_comparison.csv'), index=False)
    
    # Save detailed results for each strategy
    for name, result in results.items():
        result_dir = os.path.join(output_dir, name.replace(' ', '_').lower())
        os.makedirs(result_dir, exist_ok=True)

        # Save equity curve
        if 'equity_curve' in result:
            result['equity_curve'].to_csv(os.path.join(result_dir, 'equity_curve.csv'))

        # Save trades
        if 'trades' in result and not result['trades'].empty:
            result['trades'].to_csv(os.path.join(result_dir, 'trades.csv'))

        # Save performance metrics
        with open(os.path.join(result_dir, 'performance.txt'), 'w') as f:
            f.write(f"Performance Summary for {name}:\n")
            for key, value in result['performance'].items():
                if isinstance(value, (int, float)):
                    if key.endswith('_rate') or key in ['total_return', 'ann_return', 'ann_volatility', 'max_drawdown']:
                        f.write(f"{key}: {value:.2%}\n")
                    else:
                        f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        # If this is a regime adaptive strategy, save regime performance
        if 'Regime Adaptive' in name and 'regime_performance' in result:
            regime_perf = result['regime_performance']
            if not regime_perf.empty:
                # Save regime performance to CSV
                regime_perf.to_csv(os.path.join(result_dir, 'regime_performance.csv'))
                
                # Plot regime performance
                plt.figure(figsize=(12, 8))
                gs = plt.GridSpec(2, 2)
                
                # Plot mean return by regime
                ax1 = plt.subplot(gs[0, 0])
                regime_perf['return_mean'].plot(kind='bar', ax=ax1, color='skyblue')
                ax1.set_title('Mean Return by Regime')
                ax1.set_ylabel('Return')
                ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax1.grid(True, alpha=0.3)
                
                # Plot win rate by regime
                ax2 = plt.subplot(gs[0, 1])
                if 'win_rate' in regime_perf.columns:
                    regime_perf['win_rate'].plot(kind='bar', ax=ax2, color='green')
                else:
                    # Try to compute win rate from pnl data
                    win_rates = result['trades'].groupby('entry_regime')['pnl'].apply(
                        lambda x: (x > 0).mean()
                    )
                    win_rates.plot(kind='bar', ax=ax2, color='green')
                
                ax2.set_title('Win Rate by Regime')
                ax2.set_ylabel('Win Rate')
                ax2.axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
                ax2.grid(True, alpha=0.3)
                
                # Plot trade count by regime
                ax3 = plt.subplot(gs[1, 0])
                regime_perf['pnl_count'].plot(kind='bar', ax=ax3, color='orange')
                ax3.set_title('Number of Trades by Regime')
                ax3.set_ylabel('Count')
                ax3.grid(True, alpha=0.3)
                
                # Plot Sharpe ratio by regime if available
                ax4 = plt.subplot(gs[1, 1])
                if 'sharpe' in regime_perf.columns:
                    regime_perf['sharpe'].plot(kind='bar', ax=ax4, color='purple')
                elif 'return_mean' in regime_perf.columns and 'return_std' in regime_perf.columns:
                    # Calculate Sharpe ratio
                    sharpe = regime_perf['return_mean'] * 252 / (regime_perf['return_std'] * np.sqrt(252))
                    sharpe.plot(kind='bar', ax=ax4, color='purple')
                
                ax4.set_title('Sharpe Ratio by Regime')
                ax4.set_ylabel('Sharpe Ratio')
                ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(result_dir, 'regime_performance.png'))
                plt.close()

    print(f"\nComparison completed. Results saved to {output_dir}")

    return results

def get_strategy_configs(use_optimized_params=False):
    """
    Get strategy configurations, optionally using optimized hyperparameters.
    
    Parameters:
    -----------
    use_optimized_params : bool
        Whether to use optimized hyperparameters
        
    Returns:
    --------
    list
        List of strategy configurations
    """
    # Use predefined strategy configurations from strategy_configs.py
    strategy_configs = [
        STRATEGY_CONFIGS['decision_tree'].copy(),
        STRATEGY_CONFIGS['decision_tree_calibrated'].copy(),
        STRATEGY_CONFIGS['random_forest'].copy(),
        STRATEGY_CONFIGS['random_forest_calibrated'].copy(),
        STRATEGY_CONFIGS['xgboost_fixed'].copy(),
        STRATEGY_CONFIGS['xgboost_confidence'].copy(),
        STRATEGY_CONFIGS['stacking'].copy(),
        STRATEGY_CONFIGS['regime_adaptive_rf'].copy()
    ]
    
    # If using optimized parameters, set flag in config
    if use_optimized_params:
        for config in strategy_configs:
            config['use_optimized'] = True
            
            # For regime-adaptive strategies, enable regime-specific hyperparameters
            if 'Regime Adaptive' in config['name']:
                config['use_regime_specific_params'] = True
    
    return strategy_configs

def run_single_strategy(data_path, model_type='random_forest', output_dir='results',
                       train_end_date=None, symbol='SPY', strategy_type='trend_following',
                       calibrate=False, use_optimized_params=False):
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
        Type of strategy to use ('trend_following' or 'regime_adaptive')
    calibrate : bool, default=False
        Whether to use probability calibration for Decision Tree and Random Forest models
    use_optimized_params : bool, default=False
        Whether to use optimized hyperparameters for models
        
    Returns:
    --------
    dict
        Backtest results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(data_path, symbol=symbol)
    
    # Define training and testing periods
    if train_end_date is not None:
        train_end_date = pd.to_datetime(train_end_date)
        train_data = df[df.index <= train_end_date]
        test_data = df[df.index > train_end_date]
    else:
        # Use 70% of data for training
        train_size = int(len(df) * 0.7)
        train_data = df.iloc[:train_size]
        test_data = df.iloc[train_size:]
    
    # Print training and testing data info
    print(f"Training data: {len(train_data)} rows ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"Testing data: {len(test_data)} rows ({test_data.index[0]} to {test_data.index[-1]})")
    
    # Select appropriate configuration from STRATEGY_CONFIGS
    config_key = None
    
    if model_type == 'decision_tree':
        config_key = 'decision_tree_calibrated' if calibrate else 'decision_tree'
    elif model_type == 'random_forest':
        config_key = 'random_forest_calibrated' if calibrate else 'random_forest'
    elif model_type == 'xgboost':
        config_key = 'xgboost_confidence'  # Default to confidence-based position sizing
    elif model_type == 'stacking':
        config_key = 'stacking'
    
    if strategy_type == 'regime_adaptive' and model_type == 'random_forest':
        config_key = 'regime_adaptive_rf'
    
    # Get configuration
    if config_key and config_key in STRATEGY_CONFIGS:
        config = STRATEGY_CONFIGS[config_key].copy()
    else:
        # Fallback to basic configuration
        config = {
            'name': model_type.title(),
            'model_type': model_type,
            'model_params': ModelFactory.get_default_params(model_type),
            'use_adaptive_thresholds': 'auto'
        }
        
        # Add calibration flag for tree-based models
        if model_type in ['decision_tree', 'random_forest'] and calibrate:
            config['model_params']['calibrate'] = True
            config['use_calibration'] = True
            config['use_adaptive_thresholds'] = 'always'
    
    # Set symbol
    config['symbol'] = symbol
    
    # Set optimized parameters flag
    if use_optimized_params:
        config['use_optimized'] = True
        
        # For regime-adaptive strategies, enable regime-specific hyperparameters
        if strategy_type == 'regime_adaptive':
            config['use_regime_specific_params'] = True
    
    # Initialize and run strategy
    if strategy_type == 'regime_adaptive':
        strategy = RegimeAdaptiveStrategy()
        config['name'] = f"Regime Adaptive {model_type.title()}"
        
        # Add regime detection if not present
        if 'regime_detection' not in config:
            config['regime_detection'] = {
                'method': 'trend_volatility',
                'params': {
                    'fast_window': 20,
                    'slow_window': 50,
                    'vol_window': 20,
                    'vol_threshold': 0.75
                }
            }
            
        # Add regime parameters if not present
        if 'regime_params' not in config:
            config['regime_params'] = {
                'strong_uptrend': {'position_size_pct': 0.15, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.15},
                'uptrend': {'position_size_pct': 0.1, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.1},
                'weak_uptrend': {'position_size_pct': 0.05, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.07},
                'volatile_neutral': {'position_size_pct': 0.03, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.05},
                'neutral': {'position_size_pct': 0.05, 'stop_loss_pct': 0.03, 'take_profit_pct': 0.07},
                'low_vol_neutral': {'position_size_pct': 0.08, 'stop_loss_pct': 0.04, 'take_profit_pct': 0.08},
                'weak_downtrend': {'position_size_pct': 0.03, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.05},
                'downtrend': {'position_size_pct': 0.02, 'stop_loss_pct': 0.02, 'take_profit_pct': 0.05},
                'strong_downtrend': {'position_size_pct': 0.01, 'stop_loss_pct': 0.01, 'take_profit_pct': 0.03}
            }
    else:
        strategy = TrendFollowingStrategy()
    
    strategy.initialize(config)
    
    # Run backtest
    results = strategy.backtest(df, train_data, test_data)
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_type}_model.pkl")
    strategy.save(model_path)
    
    # Save equity curve
    if 'equity_curve' in results:
        results['equity_curve'].to_csv(os.path.join(output_dir, 'equity_curve.csv'))
    
    # Save trades
    if 'trades' in results and not results['trades'].empty:
        results['trades'].to_csv(os.path.join(output_dir, 'trades.csv'))
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    
    # Check if equity curve exists
    if 'equity_curve' in results and 'equity' in results['equity_curve']:
        # Strategy equity curve
        equity = results['equity_curve']['equity']
        plt.plot(equity.index, equity / equity.iloc[0], label=config['name'])
        
        # Buy and hold reference
        buy_hold = (test_data['close'] / test_data['close'].iloc[0])
        plt.plot(buy_hold.index, buy_hold, label='Buy & Hold', linestyle='--')
        
        plt.title(f'{config["name"]} Strategy vs Buy & Hold')
        plt.xlabel('Date')
        plt.ylabel('Equity (normalized)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
    
    # Print performance summary
    performance = results['performance']
    print("\nPerformance Summary:")
    print(f"Total Return: {performance.get('total_return', 0):.2%}")
    print(f"CAGR: {performance.get('ann_return', 0):.2%}")
    print(f"Max Drawdown: {performance.get('max_drawdown', 0):.2%}")
    print(f"Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}")
    print(f"CAGR/Max DD: {performance.get('cagr_dd_ratio', 0):.2f}")
    print(f"Win Rate: {performance.get('win_rate', 0):.2%}")
    print(f"Number of Trades: {performance.get('num_trades', 0)}")
    
    # If this is a regime adaptive strategy, save regime performance
    if strategy_type == 'regime_adaptive' and 'regime_performance' in results:
        regime_perf = results['regime_performance']
        if not regime_perf.empty:
            # Save regime performance to CSV
            regime_perf.to_csv(os.path.join(output_dir, 'regime_performance.csv'))
            
            # Plot regime performance
            plt.figure(figsize=(12, 8))
            gs = plt.GridSpec(2, 2)
            
            # Plot mean return by regime
            ax1 = plt.subplot(gs[0, 0])
            regime_perf['return_mean'].plot(kind='bar', ax=ax1, color='skyblue')
            ax1.set_title('Mean Return by Regime')
            ax1.set_ylabel('Return')
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.grid(True, alpha=0.3)
            
            # Plot win rate by regime
            ax2 = plt.subplot(gs[0, 1])
            regime_perf['win_rate'].plot(kind='bar', ax=ax2, color='green')
            ax2.set_title('Win Rate by Regime')
            ax2.set_ylabel('Win Rate')
            ax2.axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            # Plot trade count by regime
            ax3 = plt.subplot(gs[1, 0])
            regime_perf['pnl_count'].plot(kind='bar', ax=ax3, color='orange')
            ax3.set_title('Number of Trades by Regime')
            ax3.set_ylabel('Count')
            ax3.grid(True, alpha=0.3)
            
            # Plot Sharpe ratio by regime
            ax4 = plt.subplot(gs[1, 1])
            regime_perf['sharpe'].plot(kind='bar', ax=ax4, color='purple')
            ax4.set_title('Sharpe Ratio by Regime')
            ax4.set_ylabel('Sharpe Ratio')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'regime_performance.png'))
            plt.close()
    
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
                        help='Directory to save results (default: results)')
    
    parser.add_argument('--train-end', type=str, default=None,
                        help='End date for training data (format: YYYY-MM-DD)')
    
    parser.add_argument('--symbol', type=str, default='SPY',
                        help='Trading symbol (default: SPY)')
    
    parser.add_argument('--calibrate', action='store_true',
                        help='Use probability calibration for Decision Tree and Random Forest models')
    
    parser.add_argument('--use-optimized', action='store_true',
                        help='Use optimized hyperparameters for models')
    
    parser.add_argument('--adaptive-thresholds', type=str, 
                        choices=['auto', 'always', 'never'], default='auto',
                        help='Adaptive threshold behavior (default: auto)')
    
    return parser.parse_args()

def main():
    """Main function to parse arguments and run the script."""
    args = parse_arguments()
    
    # Run in specified mode
    if args.mode == 'single':
        run_single_strategy(
            data_path=args.data,
            model_type=args.model,
            output_dir=args.output,
            train_end_date=args.train_end,
            symbol=args.symbol,
            strategy_type=args.strategy,
            calibrate=args.calibrate,
            use_optimized_params=args.use_optimized
        )
    else:  # compare mode
        run_strategy_comparison(
            data_path=args.data,
            output_dir=args.output,
            train_end_date=args.train_end,
            symbol=args.symbol,
            use_optimized_params=args.use_optimized
        )

if __name__ == "__main__":
    main()
