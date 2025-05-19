#!/usr/bin/env python
"""
Hyperparameter Optimization CLI

This script provides command-line functionality for optimizing hyperparameters for
different model types using the HyperparameterManager. It supports:

1. Optimizing hyperparameters for different model types
2. Saving optimized hyperparameters for later use
3. Generating regime-specific hyperparameters
4. Running optimization in batch mode for multiple models

Part of Phase 1.5 - Hyperparameter Optimization Integration
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime

from src.data.preprocessing import preprocess_data
from src.features.regime_detection import RegimeDetector
from src.models.hyperparameter_manager import HyperparameterManager
from src.models.model_factory import ModelFactory, OPTUNA_AVAILABLE
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('hyperparameter_optimization')

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

def prepare_features_and_target(df, lookback_period=None):
    """
    Prepare features and target for model training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data
    lookback_period : int or None
        Number of days to look back for feature creation
        
    Returns:
    --------
    tuple
        X, y DataFrames for model training
    """
    # Use default from config if not specified
    if lookback_period is None:
        lookback_period = config.LOOKBACK_PERIOD
    
    # Calculate returns
    df['return'] = df['close'].pct_change()
    
    # Calculate target: up or down movement over next N days
    future_return = df['close'].pct_change(5).shift(-5)
    df['target'] = (future_return > 0).astype(int)
    
    # Calculate features
    features = []
    
    # Price-based features
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma50'] = df['close'].rolling(50).mean()
    df['ma200'] = df['close'].rolling(200).mean()
    
    features.extend(['ma20', 'ma50', 'ma200'])
    
    # Moving average ratios
    df['ma_ratio_20_50'] = df['ma20'] / df['ma50']
    df['ma_ratio_20_200'] = df['ma20'] / df['ma200']
    df['ma_ratio_50_200'] = df['ma50'] / df['ma200']
    
    features.extend(['ma_ratio_20_50', 'ma_ratio_20_200', 'ma_ratio_50_200'])
    
    # Volatility indicators
    df['std20'] = df['return'].rolling(20).std()
    df['std50'] = df['return'].rolling(50).std()
    
    features.extend(['std20', 'std50'])
    
    # Return-based features
    for period in [5, 10, 20, 50]:
        col_name = f'return_{period}d'
        df[col_name] = df['close'].pct_change(period)
        features.append(col_name)
    
    # Volume-based features if volume column exists
    if 'volume' in df.columns:
        df['vol_change'] = df['volume'].pct_change()
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma20']
        
        features.extend(['vol_change', 'vol_ma20', 'vol_ratio'])
    
    # Remove rows with NaN values
    df = df.dropna()
    
    # Extract features and target
    X = df[features]
    y = df['target']
    
    print(f"Features prepared. X shape: {X.shape}, y shape: {y.shape}")
    print(f"Features: {features}")
    print(f"Target distribution: {y.value_counts(normalize=True)}")
    
    return X, y

def detect_regimes(df, method='trend_volatility'):
    """
    Detect market regimes in the price data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data
    method : str
        Regime detection method
        
    Returns:
    --------
    dict
        Dictionary mapping data indices to regime labels
    """
    # Create regime detector
    detector = RegimeDetector(method=method)
    
    # Detect regimes
    regime_data = detector.detect_regime(df)
    
    # Get regimes
    regimes = {i: regime for i, regime in enumerate(regime_data['regime'])}
    
    # Print regime statistics
    unique_regimes = set(regimes.values())
    logger.info(f"Detected {len(unique_regimes)} unique regimes: {unique_regimes}")
    
    for regime in unique_regimes:
        count = list(regimes.values()).count(regime)
        logger.info(f"Regime '{regime}': {count} samples ({count/len(regimes)*100:.1f}%)")
    
    return regimes

def optimize_hyperparameters(args):
    """
    Optimize hyperparameters for the specified model.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    if not OPTUNA_AVAILABLE:
        logger.error("Optuna is not installed. Cannot optimize hyperparameters.")
        return
    
    # Load data
    df = load_data(args.data, args.symbol)
    
    # Prepare features and target
    X, y = prepare_features_and_target(df, args.lookback)
    
    # Create hyperparameter manager
    hyperparam_manager = HyperparameterManager(args.output)
    
    # Check if we should do regime-specific optimization
    if args.regime_specific:
        # Detect regimes
        regimes = detect_regimes(df, args.regime_method)
        
        # Perform regime-specific optimization for each model type
        if args.model == 'all':
            model_types = ['decision_tree', 'random_forest']
            if XGBOOST_AVAILABLE:
                model_types.append('xgboost')
        else:
            model_types = [args.model]
        
        for model_type in model_types:
            logger.info(f"Starting regime-specific optimization for {model_type}")
            regime_models = hyperparam_manager.get_regime_specific_models(
                X=X,
                y=y,
                model_type=model_type,
                regimes=regimes,
                force_optimization=True,
                n_trials=args.trials
            )
            
            logger.info(f"Regime-specific optimization completed for {model_type}")
            
            # Print optimized parameters for each regime
            for regime, model in regime_models.items():
                logger.info(f"Optimized parameters for {model_type} in {regime} regime:")
                params = hyperparam_manager.get_best_params(model_type, regime)
                for key, value in params.items():
                    logger.info(f"  {key}: {value}")
    else:
        # Perform regular optimization for the specified model
        if args.model == 'all':
            model_types = ['decision_tree', 'random_forest']
            if XGBOOST_AVAILABLE:
                model_types.append('xgboost')
        else:
            model_types = [args.model]
        
        for model_type in model_types:
            logger.info(f"Starting optimization for {model_type}")
            
            best_params = hyperparam_manager.optimize_hyperparameters(
                model_type=model_type,
                X=X,
                y=y,
                n_trials=args.trials
            )
            
            logger.info(f"Optimization completed for {model_type}")
            logger.info(f"Best parameters for {model_type}:")
            for key, value in best_params.items():
                logger.info(f"  {key}: {value}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter Optimization CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--data', type=str, required=True,
                        help='Path to historical data file or directory')
    
    parser.add_argument('--model', type=str, choices=['decision_tree', 'random_forest', 'xgboost', 'all'],
                        default='all',
                        help='Model type to optimize (default: all)')
    
    parser.add_argument('--output', type=str, default='data/hyperparameters',
                        help='Directory to save optimized hyperparameters')
    
    parser.add_argument('--trials', type=int, default=None,
                        help=f'Number of optimization trials (default: {config.OPTUNA_TRIALS})')
    
    parser.add_argument('--lookback', type=int, default=None,
                        help=f'Lookback period for feature creation (default: {config.LOOKBACK_PERIOD})')
    
    parser.add_argument('--symbol', type=str, default='SPY',
                        help='Trading symbol (default: SPY)')
    
    parser.add_argument('--regime-specific', action='store_true',
                        help='Perform regime-specific optimization')
    
    parser.add_argument('--regime-method', type=str, 
                        choices=['trend_volatility', 'ma_crossover', 'volatility', 'statistical'],
                        default='trend_volatility',
                        help='Regime detection method (default: trend_volatility)')
    
    return parser.parse_args()

def main():
    """Main function to parse arguments and run optimization."""
    args = parse_arguments()
    optimize_hyperparameters(args)

if __name__ == "__main__":
    main()
