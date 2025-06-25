"""
Technical indicators for transformer models.

This module provides technical indicator calculations compatible with
the transformer model's requirements.
"""

import pandas as pd
import numpy as np

# Try to import ta library, fall back to custom implementations if not available
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


def add_technical_indicators(df, window_sizes=None):
    """
    Add technical indicators to the dataframe.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLCV data
    window_sizes : dict or None
        Custom window sizes for indicators
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with added technical indicators
    """
    # Default window sizes
    if window_sizes is None:
        window_sizes = {
            'rsi': 14,
            'bb': 20,
            'ma_short': 20,
            'ma_long': 50,
            'volume_ma': 20
        }
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            # Try case-insensitive match
            for df_col in df.columns:
                if df_col.lower() == col:
                    df[col] = df[df_col]
                    break
    
    # Calculate indicators
    if TA_AVAILABLE:
        # Use ta library if available
        df = _add_indicators_with_ta(df, window_sizes)
    else:
        # Use custom implementations
        df = _add_indicators_custom(df, window_sizes)
    
    # Calculate additional features
    df = _add_price_features(df)
    df = _add_volume_features(df, window_sizes)
    
    # Fill NaN values
    df = _handle_nan_values(df)
    
    return df


def _add_indicators_with_ta(df, window_sizes):
    """Add indicators using ta library."""
    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=window_sizes['rsi'])
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(
        close=df['close'], 
        window=window_sizes['bb'], 
        window_dev=2
    )
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_middle'] = bollinger.bollinger_mavg()
    df['bb_width'] = df['bb_high'] - df['bb_low']
    df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
    
    # Moving Averages
    df['ma_20'] = df['close'].rolling(window=window_sizes['ma_short']).mean()
    df['ma_50'] = df['close'].rolling(window=window_sizes['ma_long']).mean()
    df['ma_20_slope'] = df['ma_20'].diff()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # ATR (Average True Range)
    df['atr'] = ta.volatility.average_true_range(
        df['high'], df['low'], df['close'], 
        window=window_sizes['rsi']
    )
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(
        df['high'], df['low'], df['close']
    )
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    return df


def _add_indicators_custom(df, window_sizes):
    """Add indicators using custom implementations."""
    # RSI
    df['rsi'] = _calculate_rsi(df['close'], window_sizes['rsi'])
    
    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=window_sizes['bb']).mean()
    bb_std = df['close'].rolling(window=window_sizes['bb']).std()
    df['bb_high'] = df['bb_middle'] + (2 * bb_std)
    df['bb_low'] = df['bb_middle'] - (2 * bb_std)
    df['bb_width'] = df['bb_high'] - df['bb_low']
    df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
    
    # Moving Averages
    df['ma_20'] = df['close'].rolling(window=window_sizes['ma_short']).mean()
    df['ma_50'] = df['close'].rolling(window=window_sizes['ma_long']).mean()
    df['ma_20_slope'] = df['ma_20'].diff()
    
    # Simple MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema_12 - ema_26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']
    
    # ATR
    df['atr'] = _calculate_atr(df, window_sizes['rsi'])
    
    # Stochastic
    df['stoch_k'], df['stoch_d'] = _calculate_stochastic(df)
    
    return df


def _calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def _calculate_atr(df, period=14):
    """Calculate Average True Range."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    atr = true_range.rolling(window=period).mean()
    
    return atr


def _calculate_stochastic(df, k_period=14, d_period=3):
    """Calculate Stochastic Oscillator."""
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return stoch_k, stoch_d


def _add_price_features(df):
    """Add price-based features."""
    # Price ratios
    df['high_low_ratio'] = df['high'] / df['low']
    df['close_open_ratio'] = df['close'] / df['open']
    
    # Price changes
    df['price_change'] = df['close'].pct_change()
    df['price_change_abs'] = df['close'].diff()
    
    # Candlestick patterns
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_shadow'] = df[['close', 'open']].min(axis=1) - df['low']
    
    # Price position in range
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    return df


def _add_volume_features(df, window_sizes):
    """Add volume-based features."""
    # Volume moving average
    df['volume_ma'] = df['volume'].rolling(
        window=window_sizes['volume_ma']
    ).mean()
    
    # Volume ratio
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Price-Volume trend
    df['pv_trend'] = (df['close'].pct_change() * df['volume']).cumsum()
    
    # On-Balance Volume (simplified)
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    
    return df


def _handle_nan_values(df):
    """Handle NaN values in the dataframe."""
    # Forward fill first
    df = df.fillna(method='ffill')
    
    # Then backward fill for any remaining NaNs at the beginning
    df = df.fillna(method='bfill')
    
    # If still NaN, fill with 0 (should be rare)
    df = df.fillna(0)
    
    return df


def prepare_features_for_transformer(df, feature_list=None):
    """
    Prepare and select features for transformer model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with technical indicators
    feature_list : list or None
        List of features to use. If None, uses default features
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with selected features
    """
    if feature_list is None:
        # Default feature set for transformer
        feature_list = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'bb_high', 'bb_low', 'ma_20', 'ma_20_slope',
            'macd', 'macd_signal', 'atr', 'volume_ratio'
        ]
    
    # Select available features
    available_features = []
    for feature in feature_list:
        if feature in df.columns:
            available_features.append(feature)
    
    if not available_features:
        raise ValueError("No valid features found in dataframe")
    
    return df[available_features]


def create_target_variable(df, forward_periods=1, threshold=0.0):
    """
    Create target variable for prediction.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price data
    forward_periods : int
        Number of periods to look forward
    threshold : float
        Threshold for positive return (default 0.0)
        
    Returns:
    --------
    pd.Series
        Binary target variable (1 for up, 0 for down)
    """
    # Calculate forward returns
    forward_returns = df['close'].shift(-forward_periods) / df['close'] - 1
    
    # Create binary target
    target = (forward_returns > threshold).astype(int)
    
    return target
