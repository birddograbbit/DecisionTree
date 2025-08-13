import numpy as np
import pandas as pd

from src.features.feature_engineering import engineer_features


def _make_intraday_dataframe(rows: int = 100) -> pd.DataFrame:
    index = pd.date_range('2024-01-02 09:30', periods=rows, freq='5min')
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
        'hour', 'minute', 'ema_5', 'rsi_5', 'volatility_5', 'lag_return_1', 'lag_return_3'
    }
    assert expected.issubset(set(X.columns))


def test_engineer_features_excludes_intraday_columns_for_daily():
    df = _make_daily_dataframe()
    X, y, dates = engineer_features(df, lookback_period=5, timeframe='daily')
    assert 'hour' not in X.columns
