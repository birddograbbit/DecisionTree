"""
Advanced indicators for JFK_DSRSI and MPO-3TF strategies.

This module contains complex indicator calculations including:
- Jurik Moving Average (JMA)
- Phase Shift Transform (PST)  
- Kase Permission Stochastic (KPS)
- MPO (RSI + Stochastic + MFI composite)
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple


def jurik_moving_average(series: pd.Series, length: int = 14, phase: float = 0.0, power: float = 2.0) -> pd.Series:
    """
    Calculate Jurik Moving Average (JMA).
    
    This is a simplified version that approximates JMA behavior using adaptive EMA.
    The real JMA is proprietary, but this provides similar smoothing characteristics.
    
    Parameters:
    -----------
    series : pd.Series
        Input price series
    length : int
        Lookback period
    phase : float
        Phase shift parameter (-100 to 100)
    power : float
        Power parameter for volatility adaptation
    
    Returns:
    --------
    pd.Series
        JMA values
    """
    if len(series) < length:
        return pd.Series(index=series.index, dtype=float)
    
    # Calculate volatility for adaptation
    volatility = series.rolling(length).std()
    volatility_ratio = volatility / volatility.rolling(length * 2).mean()
    volatility_ratio = volatility_ratio.fillna(1.0)
    
    # Adaptive alpha based on volatility
    base_alpha = 2.0 / (length + 1)
    phase_adj = 1.0 + (phase / 100.0) * 0.5  # Phase adjustment
    power_adj = np.power(volatility_ratio, power)
    
    alpha = base_alpha * phase_adj * power_adj
    alpha = alpha.clip(0.01, 0.99)
    
    # Calculate adaptive EMA using a fixed alpha (simplified JMA)
    # For true adaptive JMA, we'd need to iterate, but for now use mean alpha
    mean_alpha = float(alpha.mean()) if hasattr(alpha, 'mean') else alpha
    mean_alpha = np.clip(mean_alpha, 0.01, 0.99)
    jma = series.ewm(alpha=mean_alpha, adjust=False).mean()
    
    return jma


def phase_shift_transform(series: pd.Series, source: str, length: int = 14, 
                         smooth: int = 3, x_shift: int = 3, jphase: float = 0.0) -> pd.Series:
    """
    Calculate Phase Shift Transform (PST) using JMA smoothing.
    
    Parameters:
    -----------
    series : pd.Series
        Input price series (not used if DataFrame passed separately)
    source : str
        Price source ('open', 'high', 'low', 'close')
    length : int
        PST period
    smooth : int
        Smoothing period
    x_shift : int
        Phase shift bars
    jphase : float
        Jurik phase parameter
    
    Returns:
    --------
    pd.Series
        PST values
    """
    # Apply JMA smoothing
    jma_smooth = jurik_moving_average(series, length=length, phase=jphase)
    
    # Apply additional smoothing
    if smooth > 1:
        pst = jma_smooth.rolling(smooth).mean()
    else:
        pst = jma_smooth
    
    # Apply phase shift (look-ahead bars)
    pst_shifted = pst.shift(-x_shift)
    
    # Fill NaN values at the end
    pst_shifted = pst_shifted.fillna(pst.iloc[-1] if len(pst) > 0 else np.nan)
    
    return pst_shifted


def kase_permission_stochastic(df: pd.DataFrame, length: int = 14, smooth: int = 3,
                               pst_length: int = 14, pst_smooth: int = 3, 
                               pst_x: int = 3, jphase: float = 0.0) -> pd.Series:
    """
    Calculate Kase Permission Stochastic with PST filter.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLC DataFrame
    length : int
        Stochastic period
    smooth : int
        Smoothing period
    pst_length : int
        PST period
    pst_smooth : int
        PST smoothing
    pst_x : int
        PST phase shift
    jphase : float
        Jurik phase
    
    Returns:
    --------
    pd.Series
        KPS values (0-100)
    """
    # Calculate basic stochastic
    lowest = df['low'].rolling(length).min()
    highest = df['high'].rolling(length).max()
    
    # Raw stochastic
    k_raw = 100 * (df['close'] - lowest) / (highest - lowest + 1e-10)
    
    # Apply JMA smoothing to stochastic
    k_smooth = jurik_moving_average(k_raw, length=smooth, phase=jphase)
    
    # Apply PST filter for permissions
    pst = phase_shift_transform(df['close'], 'close', pst_length, pst_smooth, pst_x, jphase)
    
    # Permission logic: enhance signal when PST aligns
    pst_rising = pst > pst.shift(1)
    
    # Adjust KPS based on PST alignment
    kps = k_smooth.copy()
    kps[pst_rising & (k_smooth > 50)] = np.minimum(kps[pst_rising & (k_smooth > 50)] * 1.1, 100)
    kps[~pst_rising & (k_smooth < 50)] = np.maximum(kps[~pst_rising & (k_smooth < 50)] * 0.9, 0)
    
    return kps


def calculate_mpo(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Calculate MPO (Money Flow Oscillator) - composite of RSI, Stochastic, and MFI.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLC + Volume DataFrame
    length : int
        Period for all oscillators
    
    Returns:
    --------
    pd.Series
        MPO values (0-100)
    """
    # RSI calculation
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic calculation
    lowest = df['low'].rolling(length).min()
    highest = df['high'].rolling(length).max()
    stoch_k = 100 * (df['close'] - lowest) / (highest - lowest + 1e-10)
    
    # MFI calculation
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    raw_money_flow = typical_price * df.get('volume', 1)
    
    delta_tp = typical_price.diff()
    positive_flow = raw_money_flow.where(delta_tp > 0, 0)
    negative_flow = raw_money_flow.where(delta_tp < 0, 0)
    
    positive_mf = positive_flow.rolling(length).sum()
    negative_mf = negative_flow.rolling(length).sum()
    
    mf_ratio = positive_mf / (negative_mf + 1e-10)
    mfi = 100 - (100 / (1 + mf_ratio))
    
    # Combine into MPO (equal weighted average)
    mpo = (rsi + stoch_k + mfi) / 3
    
    return mpo.fillna(50)


def calculate_mbrsi(df: pd.DataFrame, rsi_len: int = 14, fast: int = 9, slow: int = 21) -> pd.Series:
    """
    Calculate Money Flow RSI (MB-RSI) - RSI with money flow adjustment.
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLC DataFrame
    rsi_len : int
        RSI period
    fast : int
        Fast EMA period
    slow : int
        Slow EMA period
    
    Returns:
    --------
    pd.Series
        MB-RSI values (0-100)
    """
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/rsi_len, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_len, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate trend ratio
    fast_ma = df['close'].ewm(span=fast, adjust=False).mean()
    slow_ma = df['close'].ewm(span=slow, adjust=False).mean()
    trend_ratio = fast_ma / (slow_ma + 1e-10)
    
    # Adjust RSI by trend
    mbrsi = rsi * trend_ratio
    mbrsi = mbrsi.clip(0, 100)
    
    return mbrsi.fillna(50)


def calculate_dsrsi(df: pd.DataFrame, source: str = 'close', length: int = 14, 
                    smoothing: int = 3, volume_weighted: bool = False) -> pd.Series:
    """
    Calculate Double Smoothed RSI (DS-RSI).
    
    Parameters:
    -----------
    df : pd.DataFrame
        OHLC + Volume DataFrame
    source : str
        Price source
    length : int
        RSI period
    smoothing : int
        Second RSI smoothing period
    volume_weighted : bool
        Use volume weighting
    
    Returns:
    --------
    pd.Series
        DS-RSI values (0-100)
    """
    # Get source price
    if source == 'open':
        price = df['open']
    elif source == 'high':
        price = df['high']
    elif source == 'low':
        price = df['low']
    else:
        price = df['close']
    
    # Apply volume weighting if requested
    if volume_weighted and 'volume' in df.columns:
        typical = (df['high'] + df['low'] + df['close']) / 3
        volume_norm = df['volume'] / df['volume'].rolling(20).mean()
        price = typical * volume_norm
    
    # First RSI
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi1 = 100 - (100 / (1 + rs))
    
    # Second RSI on first RSI
    delta2 = rsi1.diff()
    gain2 = delta2.clip(lower=0)
    loss2 = -delta2.clip(upper=0)
    
    avg_gain2 = gain2.ewm(alpha=1/smoothing, adjust=False).mean()
    avg_loss2 = loss2.ewm(alpha=1/smoothing, adjust=False).mean()
    
    rs2 = avg_gain2 / (avg_loss2 + 1e-10)
    dsrsi = 100 - (100 / (1 + rs2))
    
    return dsrsi.fillna(50)