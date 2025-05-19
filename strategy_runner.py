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
    
    # Define strategy configurations to compare
    strategy_configs = [
        {
            'name': 'Decision Tree',
            'model_type': 'decision_tree',
            'model_params': {'max_depth': 5, 'min_samples_split': 5, 'calibrate': False},
            'symbol': symbol
        },
        {
            'name': 'Decision Tree (Calibrated)',
            'model_type': 'decision_tree',
            'model_params': {'max_depth': 5, 'min_samples_split': 5, 'calibrate': True},
            'symbol': symbol
        },
        {
            'name': 'Random Forest',
            'model_type': 'random_forest',
            'model_params': {'n_estimators': 100, 'max_depth': 5, 'calibrate': False},
            'symbol': symbol
        },
        {
            'name': 'Random Forest (Calibrated)',
            'model_type': 'random_forest',
            'model_params': {'n_estimators': 100, 'max_depth': 5, 'calibrate': True},
            'symbol': symbol
        }
    ]
    
    # Add XGBoost configuration if available
    if 'xgboost' in ModelFactory.get_available_models():
        strategy_configs.append({
            'name': 'XGBoost',
            'model_type': 'xgboost',
            'model_params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
            'symbol': symbol,
            'position_sizing': 'fixed'
        })
        
        strategy_configs.append({
            'name': 'XGBoost (Confidence)',
            'model_type': 'xgboost',
            'model_params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
            'symbol': symbol
        })
    
    # Add Stacking model configuration
    base_models = []
    
    # Add Decision Tree and Random Forest as base models
    base_models.append({'model_type': 'decision_tree', 'model_params': {'max_depth': 5, 'calibrate': True}})
    base_models.append({'model_type': 'random_forest', 'model_params': {'n_estimators': 100, 'max_depth': 5, 'calibrate': True}})
    
    # Add XGBoost if available
    if 'xgboost' in ModelFactory.get_available_models():
        base_models.append({'model_type': 'xgboost', 'model_params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}})
    
        strategy_configs.append({
            'name': 'Stacking Ensemble',
            'model_type': 'stacking',
            'model_params': {
                'base_models': base_models,
                'meta_model': {'model_type': 'random_forest', 'model_params': {'n_estimators': 100, 'max_depth': 3}},
                'cv': 5,
                'use_features': False
        },
        'symbol': symbol
    })
    
    # Add Regime Adaptive Strategy
    # Define regime detection configuration
    regime_detection_config = {
        'method': 'trend_volatility',
        'params': {
            'fast_window': 20,
            'slow_window': 50,
            'vol_window': 20,
            'vol_threshold': 0.75
        }
    }
    
    # Define regime-specific parameters
    regime_params = {
        'strong_uptrend': {
            'position_size_pct': 0.15,  # Larger position size
            'stop_loss_pct': 0.05,      # Standard stop loss
            'take_profit_pct': 0.15     # Generous take profit
        },
        'uptrend': {
            'position_size_pct': 0.1,   # Standard position size
            'stop_loss_pct': 0.05,      # Standard stop loss
            'take_profit_pct': 0.1      # Standard take profit
        },
        'weak_uptrend': {
            'position_size_pct': 0.05,  # Smaller position size
            'stop_loss_pct': 0.03,      # Tighter stop loss
            'take_profit_pct': 0.07     # More conservative take profit
        },
        'volatile_neutral': {
            'position_size_pct': 0.03,  # Minimal position size
            'stop_loss_pct': 0.02,      # Very tight stop loss
            'take_profit_pct': 0.05     # Modest take profit
        },
        'neutral': {
            'position_size_pct': 0.05,  # Smaller position size
            'stop_loss_pct': 0.03,      # Tighter stop loss
            'take_profit_pct': 0.07     # More conservative take profit
        },
        'low_vol_neutral': {
            'position_size_pct': 0.08,  # Moderate position size
            'stop_loss_pct': 0.04,      # Moderate stop loss
            'take_profit_pct': 0.08     # Moderate take profit
        },
        'weak_downtrend': {
            'position_size_pct': 0.03,  # Minimal position size
            'stop_loss_pct': 0.02,      # Very tight stop loss
            'take_profit_pct': 0.05     # Modest take profit
        },
        'downtrend': {
            'position_size_pct': 0.02,  # Minimal position size
            'stop_loss_pct': 0.02,      # Very tight stop loss
            'take_profit_pct': 0.05     # Modest take profit
        },
        'strong_downtrend': {
            'position_size_pct': 0.01,  # Smallest position size
            'stop_loss_pct': 0.01,      # Tightest stop loss
            'take_profit_pct': 0.03     # Small take profit
        }
    }
    
    # Create a regime adaptive config for random forest
    rf_regime_config = strategy_configs[1].copy()  # Copy the random forest config
    rf_regime_config['name'] = 'Regime Adaptive RF'
    rf_regime_config['regime_detection'] = regime_detection_config
    rf_regime_config['regime_params'] = regime_params
    
    # Add to strategy configs
    strategy_configs.append(rf_regime_config)
    
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

    # Plot equity curves
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
        Type of strategy to use ('trend_following' or 'regime_adaptive')
    calibrate : bool, default=False
        Whether to use probability calibration for Decision Tree and Random Forest models
        
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
        
        model_params = {
            'base_models': base_models,
            'meta_model': {'model_type': 'random_forest', 'model_params': {'n_estimators': 100, 'max_depth': 3, 'calibrate': calibrate}},
            'cv': 5,
            'use_features': False
        }
    else:
        # Use default parameters for other model types
        model_params = ModelFactory.get_default_params(model_type)
        
        # Add calibrate flag for tree-based models
        if model_type in ['decision_tree', 'random_forest']:
            model_params['calibrate'] = calibrate
    
    # Configure strategy
    config = {
        'name': model_type.title(),
        'model_type': model_type,
        'model_params': model_params,
        'symbol': symbol
    }
    
    # Add regime detection configuration for regime adaptive strategy
    if strategy_type == 'regime_adaptive':
        config['name'] = f"Regime Adaptive {model_type.title()}"
        
        # Define regime detection configuration
        config['regime_detection'] = {
            'method': 'trend_volatility',
            'params': {
                'fast_window': 20,
                'slow_window': 50,
                'vol_window': 20,
                'vol_threshold': 0.75
            }
        }
        
        # Define regime-specific parameters
        config['regime_params'] = {
            'strong_uptrend': {
                'position_size_pct': 0.15,  # Larger position size
                'stop_loss_pct': 0.05,      # Standard stop loss
                'take_profit_pct': 0.15     # Generous take profit
            },
            'uptrend': {
                'position_size_pct': 0.1,   # Standard position size
                'stop_loss_pct': 0.05,      # Standard stop loss
                'take_profit_pct': 0.1      # Standard take profit
            },
            'weak_uptrend': {
                'position_size_pct': 0.05,  # Smaller position size
                'stop_loss_pct': 0.03,      # Tighter stop loss
                'take_profit_pct': 0.07     # More conservative take profit
            },
            'volatile_neutral': {
                'position_size_pct': 0.03,  # Minimal position size
                'stop_loss_pct': 0.02,      # Very tight stop loss
                'take_profit_pct': 0.05     # Modest take profit
            },
            'neutral': {
                'position_size_pct': 0.05,  # Smaller position size
                'stop_loss_pct': 0.03,      # Tighter stop loss
                'take_profit_pct': 0.07     # More conservative take profit
            },
            'low_vol_neutral': {
                'position_size_pct': 0.08,  # Moderate position size
                'stop_loss_pct': 0.04,      # Moderate stop loss
                'take_profit_pct': 0.08     # Moderate take profit
            },
            'weak_downtrend': {
                'position_size_pct': 0.03,  # Minimal position size
                'stop_loss_pct': 0.02,      # Very tight stop loss
                'take_profit_pct': 0.05     # Modest take profit
            },
            'downtrend': {
                'position_size_pct': 0.02,  # Minimal position size
                'stop_loss_pct': 0.02,      # Very tight stop loss
                'take_profit_pct': 0.05     # Modest take profit
            },
            'strong_downtrend': {
                'position_size_pct': 0.01,  # Smallest position size
                'stop_loss_pct': 0.01,      # Tightest stop loss
                'take_profit_pct': 0.03     # Small take profit
            }
        }
    
    # Initialize and run strategy
    if strategy_type == 'regime_adaptive':
        strategy = RegimeAdaptiveStrategy()
    else:
        strategy = TrendFollowingStrategy()
        
    strategy.initialize(config)
    
    # Run backtest
    results = strategy.backtest(df, train_data, test_data)
    
    # Save model
    model_path = os.path.join(output_dir, f"{model_type}_model.pkl")
    strategy.save(model_path)
    
    # Save equity curve
    results['equity_curve'].to_csv(os.path.join(output_dir, 'equity_curve.csv'))
    
    # Save trades
    if 'trades' in results and not results['trades'].empty:
        results['trades'].to_csv(os.path.join(output_dir, 'trades.csv'))
    
    # Plot equity curve
    plt.figure(figsize=(12, 6))
    
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
            calibrate=args.calibrate
        )
    else:  # compare mode
        run_strategy_comparison(
            data_path=args.data,
            output_dir=args.output,
            train_end_date=args.train_end,
            symbol=args.symbol
        )

if __name__ == "__main__":
    main()

