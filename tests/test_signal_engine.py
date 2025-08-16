import numpy as np
import pandas as pd
from src.engines.signal_engine import SignalEngine


def test_position_size_bounds():
    se = SignalEngine()
    size = se.get_position_size(0.6)
    assert 0 <= size <= 1
    assert se.get_position_size(0.51) == 0.0


def test_daily_trade_limit():
    se = SignalEngine()
    preds = np.array([0.6]*20)
    dates = pd.date_range('2024-01-01', periods=20, freq='T')
    sigs = se.generate_signals(preds, dates)
    filtered = se.apply_filters(sigs, max_trades_per_day=5)
    assert (filtered['signal'] != 0).sum() <= 5
