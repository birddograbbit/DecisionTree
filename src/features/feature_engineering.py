"""
Feature engineering for trading strategies.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import talib

def calculate_technical_indicators(data):
    """
    Calculate technical indicators for the given price data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price data with columns: open, high, low, close, volume
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with technical indicators
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Make sure we have the required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame must have columns: {required_cols}")
    
    # Price-based indicators
    # Moving averages
    df['sma_5'] = talib.SMA(df['close'], timeperiod=5)
    df['sma_10'] = talib.SMA(df['close'], timeperiod=10)
    df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
    df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
    df['ema_5'] = talib.EMA(df['close'], timeperiod=5)
    df['ema_10'] = talib.EMA(df['close'], timeperiod=10)
    df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
    
    # Price ratios
    df['close_to_sma_5'] = df['close'] / df['sma_5']
    df['close_to_sma_10'] = df['close'] / df['sma_10']
    df['close_to_sma_20'] = df['close'] / df['sma_20']
    df['close_to_sma_50'] = df['close'] / df['sma_50']
    
    # Returns
    df['return_1d'] = df['close'].pct_change(1)
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)
    df['return_20d'] = df['close'].pct_change(20)
    
    # Volatility indicators
    df['atr_5'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=5)
    df['atr_10'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=10)
    df['atr_20'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=20)
    
    # Normalize ATR by price to get percentage volatility
    df['atr_5_pct'] = df['atr_5'] / df['close']
    df['atr_10_pct'] = df['atr_10'] / df['close']
    df['atr_20_pct'] = df['atr_20'] / df['close']
    
    # Standard deviation of returns
    df['std_5'] = df['return_1d'].rolling(window=5).std()
    df['std_10'] = df['return_1d'].rolling(window=10).std()
    df['std_20'] = df['return_1d'].rolling(window=20).std()
    
    # Momentum indicators
    df['rsi_5'] = talib.RSI(df['close'], timeperiod=5)
    df['rsi_10'] = talib.RSI(df['close'], timeperiod=10)
    df['rsi_14'] = talib.RSI(df['close'], timeperiod=14)
    
    df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
        df['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    # Volume indicators
    df['volume_sma_5'] = talib.SMA(df['volume'], timeperiod=5)
    df['volume_sma_10'] = talib.SMA(df['volume'], timeperiod=10)
    df['volume_ratio_5'] = df['volume'] / df['volume_sma_5']
    df['volume_ratio_10'] = df['volume'] / df['volume_sma_10']
    
    df['obv'] = talib.OBV(df['close'], df['volume'])
    df['obv_sma_5'] = talib.SMA(df['obv'], timeperiod=5)
    df['obv_ratio'] = df['obv'] / df['obv_sma_5']
    
    # Bollinger Bands
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
        df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Commodity Channel Index
    df['cci_5'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=5)
    df['cci_14'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Average Directional Index
    df['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    
    # Drop NaN values
    df = df.dropna()
    
    return df

def create_target(data, horizon=1, threshold=0.0):
    """
    Create target variable for classification.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price data with close column
    horizon : int, default=1
        Prediction horizon in days
    threshold : float, default=0.0
        Return threshold for positive class
        
    Returns:
    --------
    pd.Series
        Target variable (1 for up, 0 for down/flat)
    """
    # Calculate future return
    future_return = data['close'].pct_change(horizon).shift(-horizon)
    
    # Create binary target
    target = (future_return > threshold).astype(int)
    
    return target

def engineer_features(data, lookback_period=10):
    """
    Generate features and target for model training.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price data with columns: open, high, low, close, volume
    lookback_period : int, default=10
        Number of days to look back for feature creation
        
    Returns:
    --------
    tuple
        (X, y, dates)
    """
    # Calculate technical indicators
    df_with_indicators = calculate_technical_indicators(data)
    
    # Create target variable
    y = create_target(df_with_indicators)
    
    # Select features (all columns except target)
    feature_cols = [col for col in df_with_indicators.columns 
                    if col not in ['open', 'high', 'low', 'close', 'volume', 'target']]
    
    X = df_with_indicators[feature_cols]
    
    # Get dates
    dates = df_with_indicators.index
    
    return X, y, dates

def prepare_train_test_data(data, train_end_date):
    """
    Prepare training and testing data.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Price data with columns: open, high, low, close, volume
    train_end_date : str
        End date for training data (format: 'YYYY-MM-DD')
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, dates_train, dates_test, scaler)
    """
    # Generate features and target
    X, y, dates = engineer_features(data)
    
    # Split data by date
    train_mask = dates <= train_end_date
    X_train = X[train_mask]
    y_train = y[train_mask]
    dates_train = dates[train_mask]
    
    X_test = X[~train_mask]
    y_test = y[~train_mask]
    dates_test = dates[~train_mask]
    
    # Scale features
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
    
    return X_train_scaled, X_test_scaled, y_train, y_test, dates_train, dates_test, scaler