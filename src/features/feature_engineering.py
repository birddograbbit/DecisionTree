# src/features/feature_engineering.py

"""
Module for feature engineering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from src.features.indicators import *

def add_technical_indicators(df, lookback_period=10):
    """
    Add technical indicators to price data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data with OHLCV columns
    lookback_period : int
        Lookback period for indicators (default: 10)
        
    Returns:
    --------
    pd.DataFrame
        Price data with added technical indicators
    """
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Price-based indicators
    result['returns'] = result['close'].pct_change(1)
    result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
    
    # Moving averages
    result['sma'] = result['close'].rolling(window=lookback_period).mean()
    result['ema'] = result['close'].ewm(span=lookback_period).mean()
    
    # Volatility
    result['std'] = result['returns'].rolling(window=lookback_period).std()
    
    # Momentum
    result['rsi'] = calculate_rsi(result, window=lookback_period)
    result['macd'], result['macd_signal'] = calculate_macd(result)
    
    # Volume indicators
    result['vwap'] = calculate_vwap(result)
    result['obv'] = calculate_obv(result)
    
    # Price patterns
    result['upper_band'], result['middle_band'], result['lower_band'] = calculate_bollinger_bands(result)
    
    # Additional indicators
    result['atr'] = calculate_atr(result)
    result['stoch_k'], result['stoch_d'] = calculate_stochastic(result)
    
    # Calculate SMA ratio (close / SMA)
    result['sma_ratio'] = result['close'] / result['sma']
    
    # Calculate price position within Bollinger Bands
    bb_range = result['upper_band'] - result['lower_band']
    result['bb_position'] = (result['close'] - result['lower_band']) / bb_range
    
    # Price momentum (close vs. previous periods)
    result['price_momentum_2d'] = result['close'] / result['close'].shift(2) - 1
    result['price_momentum_5d'] = result['close'] / result['close'].shift(5) - 1
    result['price_momentum_10d'] = result['close'] / result['close'].shift(10) - 1

    # Volume momentum (volume vs. previous periods)
    result['volume_momentum_1d'] = result['volume'] / result['volume'].shift(1) - 1
    result['volume_momentum_5d'] = result['volume'] / result['volume'].shift(5) - 1
    
    # New momentum-volatility hybrid features
    # ADX (Average Directional Index)
    result['adx'], result['plus_di'], result['minus_di'] = calculate_adx(result, window=lookback_period)
    
    # ADX Momentum
    result['adx_momentum'] = calculate_adx_momentum(result, adx_window=lookback_period, roc_window=5)
    
    # ATR Z-Score
    result['atr_zscore'] = calculate_atr_zscore(result, atr_window=lookback_period, zscore_window=60)
    
    # Drop NaN values introduced by indicators
    result = result.dropna()
    
    return result

def engineer_features(df, lookback_period=10):
    """
    Create feature matrix for model training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data with OHLCV columns
    lookback_period : int
        Number of days to look back for feature creation (default: 10)
        
    Returns:
    --------
    tuple
        (X: feature matrix, y: target values, dates: corresponding dates)
    """
    # Add technical indicators
    df_indicators = add_technical_indicators(df, lookback_period)
    
    # Make a copy to avoid warnings
    df_features = df_indicators.copy()
    
    # Ensure we're not using any future information
    # by using only indicators that look backwards, not forwards
    
    # Create expanded feature list including the new hybrid features
    features = [
        'sma_ratio', 'rsi', 'std', 'bb_position',
        'price_momentum_5d', 'volume_momentum_1d',
        'macd', 'stoch_k', 'atr',
        'adx', 'adx_momentum', 'atr_zscore',  # New momentum-volatility features
        'plus_di', 'minus_di'  # Additional trend indicators
    ]
    
    # Extract features
    X = df_features[features]
    
    # Create target (1 if next day's close > current close, 0 otherwise)
    # The target should be shifted FORWARD (future data)
    y = (df_features['close'].shift(-1) > df_features['close']).astype(int)
    
    # Align X and y by dropping the last row of X (no target for it)
    X = X.iloc[:-1]
    y = y.iloc[:-1]
    
    # Extract dates for reference
    dates = X.index
    
    return X, y, dates

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix
    X_test : pd.DataFrame
        Testing feature matrix
        
    Returns:
    --------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    
    # Fit scaler on training data
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    # Transform test data
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    return X_train_scaled, X_test_scaled, scaler

def audit_features(model, X_train, y_train, X_test, y_test, n_repeats=10, n_top_features=10, random_state=42):
    """
    Perform feature importance audit using permutation importance.
    
    Parameters:
    -----------
    model : object
        Trained model with predict method
    X_train : pd.DataFrame
        Training feature matrix
    y_train : pd.Series
        Training target values
    X_test : pd.DataFrame
        Testing feature matrix
    y_test : pd.Series
        Testing target values
    n_repeats : int
        Number of times to permute each feature (default: 10)
    n_top_features : int
        Number of top features to return (default: 10)
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    tuple
        (train_importances, test_importances, top_features)
        where importances are DataFrames with mean and std of importance
        and top_features is a list of feature names
    """
    # Calculate permutation importance on training data
    train_result = permutation_importance(
        model, X_train, y_train, 
        n_repeats=n_repeats, 
        random_state=random_state
    )
    
    # Calculate permutation importance on test data
    test_result = permutation_importance(
        model, X_test, y_test, 
        n_repeats=n_repeats, 
        random_state=random_state
    )
    
    # Create DataFrames with importance results
    train_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance_mean': train_result.importances_mean,
        'importance_std': train_result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    test_importances = pd.DataFrame({
        'feature': X_test.columns,
        'importance_mean': test_result.importances_mean,
        'importance_std': test_result.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    # Identify top features based on test importance
    top_features = test_importances.iloc[:n_top_features]['feature'].tolist()
    
    return train_importances, test_importances, top_features

def prune_features(X_train, X_test, top_features):
    """
    Prune features to keep only the most important ones.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix
    X_test : pd.DataFrame
        Testing feature matrix
    top_features : list
        List of feature names to keep
        
    Returns:
    --------
    tuple
        (X_train_pruned, X_test_pruned)
    """
    X_train_pruned = X_train[top_features].copy()
    X_test_pruned = X_test[top_features].copy()
    
    return X_train_pruned, X_test_pruned

def check_collinearity(X, threshold=0.8):
    """
    Check for highly correlated features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    threshold : float
        Correlation threshold (default: 0.8)
        
    Returns:
    --------
    list
        List of tuples (feature1, feature2, correlation) for correlated features
    """
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find highly correlated feature pairs
    correlated_pairs = []
    for col in upper_tri.columns:
        for idx, val in upper_tri[col].items():
            if val > threshold:
                correlated_pairs.append((idx, col, val))
                
    return correlated_pairs

def prepare_train_test_data(df, train_end_date=None, prune_features_flag=False, top_n_features=10):
    """
    Prepare training and testing data for model development.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data with OHLCV columns
    train_end_date : str or None
        End date for training data (e.g., '2022-12-31')
        If None, all data is used for both training and testing
    prune_features_flag : bool
        Whether to prune features based on importance (default: False)
    top_n_features : int
        Number of top features to keep if pruning (default: 10)
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, dates_train, dates_test, scaler)
    """
    # Add technical indicators
    df_features = add_technical_indicators(df)
    
    # Engineer features
    X, y, dates = engineer_features(df_features)
    
    # Split data into training and testing sets
    if train_end_date is not None:
        train_mask = (dates <= train_end_date)
        X_train = X[train_mask]
        y_train = y[train_mask]
        dates_train = dates[train_mask]
        
        X_test = X[~train_mask]
        y_test = y[~train_mask]
        dates_test = dates[~train_mask]
    else:
        # Use all data for both training and testing
        X_train = X
        y_train = y
        dates_train = dates
        X_test = X
        y_test = y
        dates_test = dates
    
    # Check for highly correlated features
    correlated_pairs = check_collinearity(X_train, threshold=0.8)
    if correlated_pairs:
        print(f"Found {len(correlated_pairs)} highly correlated feature pairs:")
        for feat1, feat2, corr in correlated_pairs:
            print(f"  {feat1} and {feat2}: {corr:.3f}")
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, dates_train, dates_test, scaler