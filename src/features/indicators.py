# src/features/indicators.py

"""
Module for calculating technical indicators.
"""

import numpy as np
import pandas as pd

def calculate_rsi(df, window=14, column='close'):
    """
    Calculate Relative Strength Index (RSI).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data
    window : int
        Lookback window (default: 14)
    column : str
        Column to use for calculation (default: 'close')
        
    Returns:
    --------
    pd.Series
        RSI values
    """
    # Calculate price changes
    delta = df[column].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(df, fast=12, slow=26, signal=9, column='close'):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data
    fast : int
        Fast EMA period (default: 12)
    slow : int
        Slow EMA period (default: 26)
    signal : int
        Signal period (default: 9)
    column : str
        Column to use for calculation (default: 'close')
        
    Returns:
    --------
    tuple
        (MACD line, Signal line)
    """
    # Calculate EMAs
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line and signal line
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    return macd_line, signal_line

def calculate_bollinger_bands(df, window=20, num_std=2, column='close'):
    """
    Calculate Bollinger Bands.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data
    window : int
        Lookback window (default: 20)
    num_std : int
        Number of standard deviations (default: 2)
    column : str
        Column to use for calculation (default: 'close')
        
    Returns:
    --------
    tuple
        (Upper band, Middle band, Lower band)
    """
    # Calculate middle band (SMA)
    middle_band = df[column].rolling(window=window).mean()
    
    # Calculate standard deviation
    std = df[column].rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    return upper_band, middle_band, lower_band

def calculate_vwap(df):
    """
    Calculate Volume-Weighted Average Price (VWAP).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data with columns: high, low, close, volume
        
    Returns:
    --------
    pd.Series
        VWAP values
    """
    # Calculate typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate VWAP
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    return vwap

def calculate_obv(df):
    """
    Calculate On-Balance Volume (OBV).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data with columns: close, volume
        
    Returns:
    --------
    pd.Series
        OBV values
    """
    # Calculate price changes
    price_change = df['close'].diff()
    
    # Create OBV
    obv = pd.Series(index=df.index)
    obv.iloc[0] = df['volume'].iloc[0]
    
    # Calculate OBV based on price changes
    for i in range(1, len(df)):
        if price_change.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
        elif price_change.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_atr(df, window=14):
    """
    Calculate Average True Range (ATR).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data with columns: high, low, close
    window : int
        Lookback window (default: 14)
        
    Returns:
    --------
    pd.Series
        ATR values
    """
    # Calculate true range
    high_low = df['high'] - df['low']
    high_close_prev = abs(df['high'] - df['close'].shift(1))
    low_close_prev = abs(df['low'] - df['close'].shift(1))
    
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    
    # Calculate ATR
    atr = true_range.rolling(window=window).mean()
    
    return atr

def calculate_atr_zscore(df, atr_window=14, zscore_window=60):
    """
    Calculate ATR Z-Score (volatility normalization).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data with columns: high, low, close
    atr_window : int
        Lookback window for ATR calculation (default: 14)
    zscore_window : int
        Window for Z-Score normalization (default: 60)
        
    Returns:
    --------
    pd.Series
        ATR Z-Score values
    """
    # Calculate ATR
    atr = calculate_atr(df, window=atr_window)
    
    # Calculate ATR Z-Score
    atr_mean = atr.rolling(window=zscore_window).mean()
    atr_std = atr.rolling(window=zscore_window).std()
    
    # Avoid division by zero
    atr_std = atr_std.replace(0, np.nan)
    
    atr_zscore = (atr - atr_mean) / atr_std
    
    return atr_zscore

def calculate_adx(df, window=14):
    """
    Calculate Average Directional Index (ADX).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data with columns: high, low, close
    window : int
        Lookback window (default: 14)
        
    Returns:
    --------
    tuple
        (ADX, +DI, -DI)
    """
    # Calculate the true range
    true_range = calculate_atr(df, window=1)
    
    # Calculate directional movement
    high_diff = df['high'].diff()
    low_diff = df['low'].diff()
    
    # Calculate directional indicators
    plus_dm = (high_diff > 0) & (high_diff > low_diff.abs()) * high_diff
    minus_dm = (low_diff < 0) & (low_diff.abs() > high_diff) * low_diff.abs()
    
    # Smooth directional movements
    plus_di = 100 * plus_dm.rolling(window=window).sum() / true_range.rolling(window=window).sum()
    minus_di = 100 * minus_dm.rolling(window=window).sum() / true_range.rolling(window=window).sum()
    
    # Calculate directional index (DX)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    
    # Calculate ADX
    adx = dx.rolling(window=window).mean()
    
    return adx, plus_di, minus_di

def calculate_adx_momentum(df, adx_window=14, roc_window=10):
    """
    Calculate ADX Momentum (trend strength and direction).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data with columns: high, low, close
    adx_window : int
        Lookback window for ADX calculation (default: 14)
    roc_window : int
        Rate of change window (default: 10)
        
    Returns:
    --------
    pd.Series
        ADX Momentum values
    """
    # Calculate ADX and directional indicators
    adx, plus_di, minus_di = calculate_adx(df, window=adx_window)
    
    # Direction component (+1 for uptrend, -1 for downtrend)
    direction = pd.Series(np.where(plus_di > minus_di, 1, -1), index=df.index)
    
    # Normalize ADX from 0-100 to 0-1 range
    adx_norm = adx / 100
    
    # Combine trend strength (ADX) with direction
    adx_dir = adx_norm * direction
    
    # Calculate momentum (rate of change)
    adx_momentum = adx_dir.diff(roc_window)
    
    return adx_momentum

def calculate_stochastic(df, k_period=14, d_period=3):
    """
    Calculate Stochastic Oscillator.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data with columns: high, low, close
    k_period : int
        %K period (default: 14)
    d_period : int
        %D period (default: 3)
        
    Returns:
    --------
    tuple
        (%K, %D)
    """
    # Calculate %K
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    
    # Calculate %D
    d = k.rolling(window=d_period).mean()
    
    return k, d