import pandas as pd
from src.backtesting.engine import BacktestEngine


def test_intraday_positions_flattened():
    dates = pd.to_datetime([
        '2024-01-02 13:30', '2024-01-02 19:55',
        '2024-01-03 13:30', '2024-01-03 19:55'
    ])
    prices = pd.DataFrame({'close': [100, 101, 102, 103]}, index=dates)
    signals = pd.DataFrame({
        'date': dates,
        'symbol': ['SPY'] * 4,
        'signal': [1, 0, 1, 0],
        'probability': [0.6] * 4,
        'position_size': [1] * 4,
    }).set_index('date')

    engine = BacktestEngine(initial_capital=100000, commission=0.0, slippage=0.0)
    result = engine.run_backtest(signals, {'SPY': prices}, timeframe='5min')

    assert engine.positions == {}
    trades = result['trades']
    assert len(trades) == 2
    assert all(trades['exit_date'].dt.date == trades['entry_date'].dt.date)

