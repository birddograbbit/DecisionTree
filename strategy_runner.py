#!/usr/bin/env python
"""
Strategy Runner - Script for running and comparing different trading strategies.

This script provides functionality to:
1. Run a single strategy with specific parameters
2. Compare multiple strategies or configurations
3. Visualize and save results
4. Perform feature auditing and pruning
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from src.data.preprocessing import preprocess_data, load_5min_data
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.strategies.strategy_registry import StrategyRegistry
from src.models.model_factory import ModelFactory
from src.models.hyperparameter_manager import HyperparameterManager
from src.features.feature_engineering import (
    prepare_train_test_data,
    audit_features,
    prune_features,
    check_collinearity,
    add_technical_indicators,
    engineer_features
)
from strategy_configs import STRATEGY_CONFIGS
import config

def load_data(data_path, symbol='SPY', timeframe='daily'):
    """
    Load and preprocess historical price data.
    
    Parameters:
    -----------
    data_path : str
        Path to the data file or directory
    symbol : str, default='SPY'
        Symbol for the data
    timeframe : str, default='daily'
        Data timeframe - 'daily' or '5min'
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed price data
    """
    # Handle 5-minute data specifically
    if timeframe == '5min':
        # Use specific 5-minute data files
        train_file = os.path.join(data_path, f'historical_data_STOCK_{symbol}_5_mins_2023-2024.csv')
        test_file = os.path.join(data_path, f'historical_data_STOCK_{symbol}_5_mins_2025.csv')
        
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            raise FileNotFoundError(f"5-minute data files not found for {symbol}")
        
        # Use the dedicated 5-minute data loading function
        df = load_5min_data(train_file, test_file)
        # Don't preprocess again as load_5min_data already handles it
        return df
    
    # Original logic for daily data
    if os.path.isfile(data_path):
        # Load data from single file
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    else:
        # Look for CSV files in the directory
        csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and symbol in f and '5_mins' not in f]
        
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

def run_feature_audit(data_path, output_dir, model_type='random_forest', 
                     top_n_features=None, audit_only=False, symbol='SPY', timeframe='daily'):
    """
    Run feature importance audit on the data.
    
    Parameters:
    -----------
    data_path : str
        Path to the historical data file or directory
    output_dir : str
        Directory to save audit results
    model_type : str, default='random_forest'
        Model type to use for feature importance evaluation
    top_n_features : int, optional
        Number of top features to keep (default from config)
    audit_only : bool, default=False
        If True, only perform audit without affecting main workflow
    symbol : str, default='SPY'
        Trading symbol
        
    Returns:
    --------
    tuple
        (top_features_list, feature_audit_results) if audit_only,
        otherwise (X_train_pruned, X_test_pruned, y_train, y_test, audit_results)
    """
    # Create audit output directory
    audit_dir = os.path.join(output_dir, 'feature_audit')
    os.makedirs(audit_dir, exist_ok=True)
    
    print(f"\n=== Running Feature Audit with {model_type} ===")
    
    # Load and prepare data
    df = load_data(data_path, symbol=symbol, timeframe=timeframe)
    
    # Use config default if not specified
    if top_n_features is None:
        top_n_features = config.TOP_N_FEATURES
    
    # Determine lookback period based on timeframe
    lookback_period = config.LOOKBACK_PERIOD_5MIN if timeframe == '5min' else config.LOOKBACK_PERIOD
    print(f"Using lookback period: {lookback_period} {'bars' if timeframe == '5min' else 'days'}")
    
    # Add technical indicators with appropriate lookback
    df_features = add_technical_indicators(df, lookback_period)
    
    # Engineer features with appropriate lookback
    X, y, dates = engineer_features(df_features, lookback_period, timeframe=timeframe)
    
    # Split data for audit (using 70/30 split)
    train_size = int(len(X) * 0.7)
    X_train = X.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_train = y.iloc[:train_size]
    y_test = y.iloc[train_size:]
    dates_train = dates[:train_size]
    dates_test = dates[train_size:]
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    X_train = X_train_scaled
    X_test = X_test_scaled
    
    print(f"Original feature count: {X_train.shape[1]}")
    print(f"Features: {list(X_train.columns)}")
    
    # Check for collinearity
    print("\n--- Checking for Feature Collinearity ---")
    correlated_pairs = check_collinearity(X_train, threshold=config.COLLINEARITY_THRESHOLD)
    if correlated_pairs:
        print(f"Found {len(correlated_pairs)} highly correlated feature pairs:")
        for feat1, feat2, corr in correlated_pairs:
            print(f"  {feat1} and {feat2}: {corr:.3f}")
        
        # Save collinearity results
        with open(os.path.join(audit_dir, 'collinearity_analysis.txt'), 'w') as f:
            f.write("Highly Correlated Feature Pairs:\n")
            f.write(f"Threshold: {config.COLLINEARITY_THRESHOLD}\n\n")
            for feat1, feat2, corr in correlated_pairs:
                f.write(f"{feat1} and {feat2}: {corr:.3f}\n")
    else:
        print("No highly correlated features found.")
    
    # Train model for feature importance analysis
    print(f"\n--- Training {model_type} for Feature Importance ---")
    model = ModelFactory.create_model(model_type)
    model.train(X_train, y_train)
    
    # Perform feature audit
    print(f"--- Running Permutation Importance Analysis ---")
    train_imp, test_imp, top_features = audit_features(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        n_repeats=config.FEATURE_AUDIT_N_REPEATS,
        n_top_features=top_n_features,
        random_state=config.RANDOM_STATE
    )
    
    print(f"\nTop {len(top_features)} features selected:")
    for i, feature in enumerate(top_features, 1):
        importance = test_imp[test_imp['feature'] == feature]['importance_mean'].iloc[0]
        print(f"  {i:2d}. {feature}: {importance:.4f}")
    
    # Save audit results
    train_imp.to_csv(os.path.join(audit_dir, 'train_importance.csv'), index=False)
    test_imp.to_csv(os.path.join(audit_dir, 'test_importance.csv'), index=False)
    
    with open(os.path.join(audit_dir, 'top_features.txt'), 'w') as f:
        for feature in top_features:
            f.write(f"{feature}\n")
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    
    # Plot top features from test set
    top_importance = test_imp.head(top_n_features)
    plt.barh(range(len(top_importance)), top_importance['importance_mean'], 
             yerr=top_importance['importance_std'], color='skyblue', alpha=0.7)
    plt.yticks(range(len(top_importance)), top_importance['feature'])
    plt.xlabel('Permutation Importance (Mean ± Std)')
    plt.title(f'Top {top_n_features} Feature Importance ({model_type})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(audit_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save audit summary
    with open(os.path.join(audit_dir, 'audit_summary.txt'), 'w') as f:
        f.write(f"Feature Audit Summary\n")
        f.write(f"====================\n\n")
        f.write(f"Model Used: {model_type}\n")
        f.write(f"Original Features: {X_train.shape[1]}\n")
        f.write(f"Selected Features: {len(top_features)}\n")
        f.write(f"Reduction: {(1 - len(top_features) / X_train.shape[1]) * 100:.1f}%\n\n")
        f.write(f"Top {len(top_features)} Features:\n")
        for i, feature in enumerate(top_features, 1):
            importance = test_imp[test_imp['feature'] == feature]['importance_mean'].iloc[0]
            f.write(f"{i:2d}. {feature}: {importance:.4f}\n")
    
    audit_results = {
        'train_importance': train_imp,
        'test_importance': test_imp,
        'top_features': top_features,
        'collinearity_pairs': correlated_pairs,
        'original_feature_count': X_train.shape[1],
        'selected_feature_count': len(top_features)
    }
    
    if audit_only:
        print(f"\nFeature audit completed. Results saved to {audit_dir}")
        return top_features, audit_results
    
    # Prune features for main workflow
    print(f"\n--- Pruning Features to Top {len(top_features)} ---")
    X_train_pruned, X_test_pruned = prune_features(X_train, X_test, top_features)
    
    print(f"Features after pruning: {X_train_pruned.shape[1]}")
    print(f"Pruned features: {list(X_train_pruned.columns)}")
    
    return X_train_pruned, X_test_pruned, y_train, y_test, audit_results

def run_strategy_comparison(data_path, output_dir='results_comparison', 
                           train_end_date=None, symbol='SPY', use_optimized_params=False,
                           run_feature_audit_flag=False, audit_model='random_forest', 
                           top_n_features=None, include_momentum=False, timeframe='daily'):
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
    run_feature_audit_flag : bool, default=False
        Whether to perform feature auditing before running strategies
    audit_model : str, default='random_forest'
        Model type to use for feature importance evaluation
    top_n_features : int, optional
        Number of top features to keep after auditing
    include_momentum : bool, default=False
        Whether to include momentum strategies in comparison
        
    Returns:
    --------
    dict
        Comparison results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run feature audit if requested
    if run_feature_audit_flag:
        print("=== Feature Audit Phase ===")
        top_features, audit_results = run_feature_audit(
            data_path, output_dir, audit_model, top_n_features, audit_only=True, symbol=symbol, timeframe=timeframe
        )
        print(f"Feature audit completed. Selected {len(top_features)} features.")
        
        # Save audit results for reference
        with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f:
            for feature in top_features:
                f.write(f"{feature}\n")
    
    # Load data
    df = load_data(data_path, symbol=symbol, timeframe=timeframe)
    
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
    strategy_configs = get_strategy_configs(use_optimized_params, include_momentum=include_momentum, timeframe=timeframe)
    
    # Set symbol for all configs and feature audit settings
    for config in strategy_configs:
        config['symbol'] = symbol
        config['timeframe'] = timeframe
        if use_optimized_params:
            config['use_optimized'] = True
        if run_feature_audit_flag:
            config['use_feature_pruning'] = True
            config['audit_model'] = audit_model
            config['top_n_features'] = top_n_features or config.TOP_N_FEATURES
    
    # Run strategies
    results = {}
    equity_curves = []
    metrics_list = []

    for config in strategy_configs:
        print(f"\n=== Running strategy: {config['name']} ===")

        # Determine strategy type from config name
        if config.get('model_type') == 'multi_timeframe' or 'multi_timeframe' in config['name'].lower():
            strategy_type = 'multi_timeframe'
        elif 'meta_strategy' in config['name'].lower():
            strategy_type = 'meta_strategy'
            # Dynamically import and register meta-strategy if needed
            if 'meta_strategy' not in StrategyRegistry.list_strategies():
                from src.strategies.meta_strategy import MetaStrategy
                StrategyRegistry.register_strategy('meta_strategy', MetaStrategy)
        elif 'Regime Adaptive' in config['name']:
            strategy_type = 'regime_adaptive'
        elif 'BB-RSI-ADX' in config['name']:
            strategy_type = 'bb_rsi_adx'
        elif 'TEMA' in config['name']:
            strategy_type = 'tema'
        elif 'Quod' in config['name']:
            strategy_type = 'quod'
        else:
            strategy_type = 'trend_following'
            
        # Use StrategyRegistry to get strategy instance
        strategy = StrategyRegistry.get_strategy(strategy_type, config)

        # Run backtest
        backtest_results = strategy.backtest(df, train_data, test_data, timeframe)

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

def get_strategy_configs(use_optimized_params=False, include_momentum=False, timeframe='daily'):
    """
    Get strategy configurations, optionally using optimized hyperparameters.
    
    Parameters:
    -----------
    use_optimized_params : bool
        Whether to use optimized hyperparameters
    include_momentum : bool
        Whether to include momentum strategies
    timeframe : str
        Data timeframe - 'daily' or '5min'
        
    Returns:
    --------
    list
        List of strategy configurations
    """
    # Use predefined strategy configurations from strategy_configs.py
    if timeframe == '5min':
        strategy_configs = [
            STRATEGY_CONFIGS['decision_tree_5min'].copy(),
            STRATEGY_CONFIGS['random_forest_5min'].copy(),
            STRATEGY_CONFIGS['xgboost_5min'].copy(),
            STRATEGY_CONFIGS['transformer_5min'].copy(),
        ]
    else:
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
    
    # Add momentum strategies if requested
    if include_momentum:
        if timeframe == '5min':
            # Use 5-minute optimized configurations
            strategy_configs.extend([
                STRATEGY_CONFIGS['bb_rsi_adx_5min'].copy(),
                STRATEGY_CONFIGS['tema_5min'].copy(),
                STRATEGY_CONFIGS['quod'].copy()  # Quod already configured for 5T
            ])
        else:
            # Use standard configurations
            strategy_configs.extend([
                STRATEGY_CONFIGS['bb_rsi_adx'].copy(),
                STRATEGY_CONFIGS['tema'].copy(),
                STRATEGY_CONFIGS['quod'].copy()
            ])
    
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
                       calibrate=False, use_optimized_params=False,
                       run_feature_audit_flag=False, audit_model='random_forest',
                       top_n_features=None, timeframe='daily',
                       performance_window=None, switch_cooldown=None):
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
    run_feature_audit_flag : bool, default=False
        Whether to perform feature auditing before running strategy
    audit_model : str, default='random_forest'
        Model type to use for feature importance evaluation
    top_n_features : int, optional
        Number of top features to keep after auditing
    performance_window : int, optional
        Lookback window (in bars) for meta-strategy performance tracking
    switch_cooldown : int, optional
        Minimum bars between meta-strategy switches

    Returns:
    --------
    dict
        Backtest results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run feature audit if requested
    if run_feature_audit_flag:
        print("=== Feature Audit Phase ===")
        top_features, audit_results = run_feature_audit(
            data_path, output_dir, audit_model, top_n_features, audit_only=True, symbol=symbol, timeframe=timeframe
        )
        print(f"Feature audit completed. Selected {len(top_features)} features.")
    
    # Load data
    df = load_data(data_path, symbol=symbol, timeframe=timeframe)
    
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
    
    # Check if model_type is a momentum strategy or meta-strategy
    momentum_strategies = ['bb_rsi_adx', 'tema', 'quod', 'meta_strategy']
    
    if model_type in momentum_strategies:
        # For momentum strategies, use the model type as the strategy type
        strategy_type = model_type
        config_key = None  # Will create custom config below
    elif model_type == 'hybrid_momentum':
        strategy_type = 'hybrid_momentum'
        config_key = 'hybrid_xgb_tema_5min'
    else:
        # Select appropriate configuration from STRATEGY_CONFIGS for ML models
        config_key = None
        
        if model_type == 'decision_tree':
            if timeframe == '5min':
                config_key = 'decision_tree_5min'
            else:
                config_key = 'decision_tree_calibrated' if calibrate else 'decision_tree'
        elif model_type == 'random_forest':
            if timeframe == '5min':
                config_key = 'random_forest_5min'
            else:
                config_key = 'random_forest_calibrated' if calibrate else 'random_forest'
        elif model_type == 'xgboost':
            config_key = 'xgboost_5min' if timeframe == '5min' else 'xgboost_confidence'
        elif model_type == 'stacking':
            config_key = 'stacking'
        elif model_type in ['transformer', 'hybrid']:
            config_key = 'transformer_5min' if (model_type == 'transformer' and timeframe == '5min') else None
        
        if strategy_type == 'regime_adaptive' and model_type == 'random_forest':
            config_key = 'regime_adaptive_rf'
    
    # Get configuration
    if config_key and config_key in STRATEGY_CONFIGS:
        config = STRATEGY_CONFIGS[config_key].copy()
    elif model_type in momentum_strategies:
        # Create configuration for momentum strategies
        config = {
            'name': model_type.upper().replace('_', '-'),
            'symbol': symbol,
            'position_size': 0.1,
            'primary_timeframe': '1h'  # Default timeframe
        }
        
        # Add strategy-specific default parameters
        if model_type == 'bb_rsi_adx':
            config.update({
                'bb_period': 20,
                'rsi_period': 14,
                'adx_primary_threshold': 20,
                'adx_secondary_threshold': 40
            })
        elif model_type == 'tema':
            config.update({
                'tema_primary_fast': 10,
                'tema_primary_slow': 80,
                'adx_threshold': 40,
                'use_dual_timeframe': True
            })
        elif model_type == 'quod':
            config.update({
                'use_stoch_reversal': True,
                'use_stoch_pullback': True,
                'use_d60_trend_exit': True,
                'primary_timeframe': '5T'  # 5-minute default for Quod
            })
        elif model_type == 'meta_strategy':
            config = STRATEGY_CONFIGS.get('meta_strategy', {}).copy()
            if performance_window is not None:
                config['performance_window'] = performance_window
            if switch_cooldown is not None:
                config['switch_cooldown'] = switch_cooldown
            config.setdefault('strategies', ['quod', 'tema', 'bb_rsi_adx'])
            # Dynamically import and register meta-strategy if needed
            if 'meta_strategy' not in StrategyRegistry.list_strategies():
                from src.strategies.meta_strategy import MetaStrategy
                StrategyRegistry.register_strategy('meta_strategy', MetaStrategy)
    else:
        # Fallback to basic configuration
        config = {
            'name': model_type.title(),
            'model_type': model_type,
            'model_params': {} if model_type == 'meta_strategy' else ModelFactory.get_default_params(model_type),
            'use_adaptive_thresholds': 'auto'
        }
        
        # Add calibration flag for tree-based models
        if model_type in ['decision_tree', 'random_forest'] and calibrate:
            config['model_params']['calibrate'] = True
            config['use_calibration'] = True
            config['use_adaptive_thresholds'] = 'always'
    
    # Set symbol and feature audit settings
    config['symbol'] = symbol
    config['timeframe'] = timeframe
    
    # Set optimized parameters flag
    if use_optimized_params:
        config['use_optimized'] = True
        
        # For regime-adaptive strategies, enable regime-specific hyperparameters
        if strategy_type == 'regime_adaptive':
            config['use_regime_specific_params'] = True
    
    # Set feature audit settings
    if run_feature_audit_flag:
        config['use_feature_pruning'] = True
        config['audit_model'] = audit_model
        config['top_n_features'] = top_n_features or config.TOP_N_FEATURES
    
    # Initialize and run strategy
    if strategy_type == 'regime_adaptive':
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
    
    # Use StrategyRegistry to get strategy instance
    strategy = StrategyRegistry.get_strategy(strategy_type, config)
    
    # Run backtest
    results = strategy.backtest(df, train_data, test_data, timeframe)
    
    # Save model/strategy
    # For momentum strategies, this saves configuration only
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
    parser = argparse.ArgumentParser(description='Run trading strategies with optional feature auditing')
    
    parser.add_argument('--data', type=str, required=True,
                        help='Path to historical data file or directory')
    
    parser.add_argument('--mode', type=str, choices=['single', 'compare', 'audit'], default='single',
                        help='Mode: single strategy, comparison, or feature audit only (default: single)')
    
    parser.add_argument('--model', type=str,
                        choices=['decision_tree', 'random_forest', 'xgboost', 'stacking', 'transformer', 'hybrid',
                                 'bb_rsi_adx', 'tema', 'quod', 'meta_strategy', 'hybrid_momentum'],
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
    
    # Feature auditing arguments
    parser.add_argument('--feature-audit', action='store_true',
                        help='Perform feature importance audit before running strategies')
    
    parser.add_argument('--audit-model', type=str, 
                        choices=['decision_tree', 'random_forest', 'xgboost'], 
                        default='random_forest',
                        help='Model type for feature importance evaluation (default: random_forest)')
    
    parser.add_argument('--top-features', type=int, default=None,
                        help=f'Number of top features to keep (default: {config.TOP_N_FEATURES})')
    
    parser.add_argument('--include-momentum', action='store_true',
                        help='Include momentum strategies (BB-RSI-ADX, TEMA, Quod) in comparison mode')
    
    parser.add_argument('--timeframe',
                        choices=['daily', '5min'],
                        default='daily',
                        help='Data timeframe to use (default: daily)')

    parser.add_argument('--performance-window', type=int, default=None,
                        help='Performance window for meta-strategy (bars)')
    parser.add_argument('--switch-cooldown', type=int, default=None,
                        help='Switch cooldown for meta-strategy (bars)')
    
    return parser.parse_args()

def main():
    """Main function to parse arguments and run the script."""
    args = parse_arguments()
    
    # Run in specified mode
    if args.mode == 'audit':
        # Run feature audit only
        run_feature_audit(
            data_path=args.data,
            output_dir=args.output,
            model_type=args.audit_model,
            top_n_features=args.top_features,
            audit_only=True,
            symbol=args.symbol,
            timeframe=args.timeframe
        )
    elif args.mode == 'single':
        run_single_strategy(
            data_path=args.data,
            model_type=args.model,
            output_dir=args.output,
            train_end_date=args.train_end,
            symbol=args.symbol,
            strategy_type=args.strategy,
            calibrate=args.calibrate,
            use_optimized_params=args.use_optimized,
            run_feature_audit_flag=args.feature_audit,
            audit_model=args.audit_model,
            top_n_features=args.top_features,
            timeframe=args.timeframe,
            performance_window=args.performance_window,
            switch_cooldown=args.switch_cooldown
        )
    else:  # compare mode
        run_strategy_comparison(
            data_path=args.data,
            output_dir=args.output,
            train_end_date=args.train_end,
            symbol=args.symbol,
            use_optimized_params=args.use_optimized,
            run_feature_audit_flag=args.feature_audit,
            audit_model=args.audit_model,
            top_n_features=args.top_features,
            include_momentum=args.include_momentum,
            timeframe=args.timeframe
        )

if __name__ == "__main__":
    main()
