import numpy as np
import pandas as pd

from src.features.feature_engineering import engineer_features


def _make_intraday_dataframe(rows: int = 100, freq: str = '5min') -> pd.DataFrame:
    index = pd.date_range('2024-01-02 09:30', periods=rows, freq=freq)
    base = np.arange(rows, dtype=float)
    data = pd.DataFrame({
        'open': base,
        'high': base + 1,
        'low': base - 1,
        'close': base + 0.5,
        'volume': np.full(rows, 1000)
    }, index=index)
    return data


def _make_daily_dataframe(rows: int = 100) -> pd.DataFrame:
    index = pd.date_range('2024-01-01', periods=rows, freq='D')
    base = np.arange(rows, dtype=float)
    data = pd.DataFrame({
        'open': base,
        'high': base + 1,
        'low': base - 1,
        'close': base + 0.5,
        'volume': np.full(rows, 1000)
    }, index=index)
    return data


def test_engineer_features_adds_intraday_columns():
    df = _make_intraday_dataframe()
    X, y, dates = engineer_features(df, lookback_period=5, timeframe='5min')
    expected = {
        'hour', 'minute', 'minutes_from_open', 'minutes_to_close', 'is_first_30m',
        'is_power_hour', 'volatility_5', 'volatility_15', 'rsi_7', 'vwap_distance',
        'ret_1bar', 'ret_3bar', 'ret_6bar'
    }
    assert expected.issubset(set(X.columns))


def test_engineer_features_adds_intraday_columns_1min():
    df = _make_intraday_dataframe(freq='1min')
    X, y, dates = engineer_features(df, lookback_period=5, timeframe='1min')
    expected = {
        'hour', 'minute', 'minutes_from_open', 'minutes_to_close', 'is_first_30m',
        'is_power_hour', 'volatility_5', 'volatility_15', 'rsi_7', 'vwap_distance',
        'ret_1bar', 'ret_3bar', 'ret_6bar'
    }
    assert expected.issubset(set(X.columns))


def test_engineer_features_excludes_intraday_columns_for_daily():
    df = _make_daily_dataframe()
    X, y, dates = engineer_features(df, lookback_period=5, timeframe='daily')
    assert 'hour' not in X.columns
