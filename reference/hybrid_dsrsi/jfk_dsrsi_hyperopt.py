#!/usr/bin/env python
# coding: utf-8
"""
jfk_dsrsi_hyperopt.py

Hyperopt optimization for the JFKPS/DSRSI strategy using market-order execution.
This version optimizes BOTH:
  - JFKPS parameters (pstLength, pstX, pstSmooth, Jurik length/phase)
  - DSRSI & mode toggle (use DSRSI entries with KPS-trend filter vs KPS-only)
  - Risk settings (ATR length, SL/TP/TS, etc.)

Notes:
- Executes 1-unit trades for pure strategy quality evaluation (position sizing
  is ignored for scoring; engine still holds SL/TP for realism).
- Entries are filled at next bar's open to avoid look-ahead.

References:
- Loxx JFKPS (TradingView id kIBAXchQ).
- Kase Permission Stochastic concept & thresholds.
- Jurik filter implementation mirrored from pandas_ta JMA.

Citations:
- https://www.tradingview.com/script/kIBAXchQ-Jurik-Filtered-Kase-Permission-Stochastic-Loxx/
- https://www.kaseco.com/wp-content/uploads/2016/03/Kase-StatWare-Manual-CQG.pdf
- https://tradingstrategy.ai/docs/_modules/pandas_ta/overlap/jma.html
"""

import argparse
import numpy as np
import pickle
import logging
import os
import sys
from datetime import datetime
from hyperopt import fmin, tpe, hp, Trials

# Local imports (backtest + metrics/data utils)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from jfk_dsrsi_backtest import (
    BacktestEngine,
    calculate_dsrsi_indicators,
    calculate_kps_indicators,
    generate_signals
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tema_backtest_v3 import (
    load_data, resample_data, calculate_performance_metrics
)

# ────────────────────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────────────────────
def setup_logging(symbol, timeframe):
    os.makedirs('logs', exist_ok=True)
    log_filename = os.path.join('logs', f'jfk_dsrsi_hyperopt_{symbol}_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return log_filename

# ────────────────────────────────────────────────────────────────────────────────
# Run a single backtest with parameter dict
# ────────────────────────────────────────────────────────────────────────────────
def run_backtest(data, params, initial_capital=100000, commission=None,
                 market_hours_only=True, close_at_eod=True):
    df = data.copy()

    # --- Indicators ---
    # DSRSI (always computed; may or may not be used for entries)
    df = calculate_dsrsi_indicators(
        df,
        source=params['source'],
        length=int(params['dsrsi_length']),
        smoothing_period=int(params['smoothing_period']),
        volume_weighted=params['volume_weighted']
    )

    # JFKPS
    df = calculate_kps_indicators(
        df,
        pst_length=int(params['pst_length']),
        pst_x=int(params['pst_x']),
        pst_smooth=int(params['pst_smooth']),
        smooth_period=int(params['kps_smooth_period']),
        jphase=float(params['jphase'])
    )

    # Signals
    signals = generate_signals(
        df,
        use_dsrsi=params['use_dsrsi'],
        market_hours_only=market_hours_only,
        atr_length=int(params['atr_length']),
        sl_atr_ratio=float(params['sl_atr_ratio']),
        tp_sl_ratio=float(params['tp_sl_ratio'])
    )

    # --- Engine (1-unit trading for objective) ---
    engine = BacktestEngine(
        symbol='TEST',
        initial_capital=initial_capital,
        risk_percent=3.0,            # Ignored by our "1-unit evaluation"
        position_multiplier=1.0,     # "
        commission=commission,
        use_dsrsi=params['use_dsrsi'],
        close_at_eod=close_at_eod,
        market_hours_only=market_hours_only,
        use_sl=params['use_sl'],
        use_tp=params['use_tp'],
        use_ts=params['use_ts'],
        ts_swing_lookback=int(params['ts_swing_lookback']),
        ts_method=params['ts_method'],
        ts_source=params['ts_source'],
        ts_atr_multiplier=float(params['ts_atr_multiplier']),
        ts_percent=float(params['ts_percent']),
        exit_at_opposite_signal=params['exit_at_opposite_signal'],
        kps_long_exit_threshold=float(params['kps_long_exit_threshold']),
        kps_short_exit_threshold=float(params['kps_short_exit_threshold'])
    )

    trades_df = engine.run(signals)

    if len(trades_df) == 0:
        # Penalize no-trade configurations
        return {
            'total_return': 0.0, 'sharpe_ratio': 0.0, 'profit_factor': 0.0,
            'max_drawdown': 0.0, 'total_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0
        }

    metrics = calculate_performance_metrics(trades_df, engine.equity_curve, initial_capital)
    return metrics

# ────────────────────────────────────────────────────────────────────────────────
# Combined score (reward consistency + profitability)
# ────────────────────────────────────────────────────────────────────────────────
def combined_score(m):
    tr = m['total_return']
    pf = m['profit_factor']
    n = m['total_trades']
    if n == 0:
        return -1e6
    if tr < 0 and pf > 0:
        return tr + min((pf - 1) * 50, 100)  # small bonus for PF even if loss
    if tr > 0:
        return tr * max(pf, 0.1)
    return tr

# ────────────────────────────────────────────────────────────────────────────────
# Objective
# ────────────────────────────────────────────────────────────────────────────────
trial_counter = 0
trial_dump_interval = 100
pkl_filename = None
trials = None
data = None
market_hours_only = True
close_at_eod = True
commission = None
initial_capital = 100000

def objective(params):
    global trial_counter, trials, pkl_filename, data, market_hours_only, close_at_eod, commission, initial_capital
    try:
        metrics = run_backtest(
            data, params,
            initial_capital=initial_capital,
            commission=commission,
            market_hours_only=market_hours_only,
            close_at_eod=close_at_eod
        )
        loss = -combined_score(metrics)
    except Exception as e:
        logging.exception(f"Objective error: {e}")
        metrics = {'total_return': 0, 'sharpe_ratio': 0, 'profit_factor': 0, 'total_trades': 0, 'win_rate': 0, 'max_drawdown': 0, 'total_pnl': 0}
        loss = 1e10

    trial_counter += 1
    cs = combined_score(metrics)
    msg = (f"Iter {trial_counter}: Ret {metrics['total_return']:.2f}% | Sharpe {metrics['sharpe_ratio']:.2f} | "
           f"PF {metrics['profit_factor']:.2f} | Trades {metrics['total_trades']} | Win {metrics['win_rate']:.2f}% | "
           f"MaxDD {metrics['max_drawdown']:.2f}% | Score {cs:.2f} | Loss {loss:.4f}")
    logging.info(msg)
    print(msg)

    # periodic checkpoint
    if trial_counter % trial_dump_interval == 0:
        with open(pkl_filename, "wb") as fh:
            pickle.dump(trials, fh)
        print(f"Trials checkpoint @ {trial_counter} -> {pkl_filename}")
    return loss

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="JFKPS/DSRSI market-order strategy optimization (HyperOpt)")
    parser.add_argument('--datafile', type=str, required=True, help='CSV with historical bar data (IBKR format)')
    parser.add_argument('--max_evals', type=int, default=500, help='Number of evaluations to run')
    parser.add_argument('--symbol', type=str, default='SPX', help='Symbol name')
    parser.add_argument('--timeframe', type=str, default='5_mins', help='Timeframe label for logs')
    parser.add_argument('--start_date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--resample', type=str, default=None, help='Resample timeframe (e.g., 1h, 4h)')
    parser.add_argument('--close_at_eod', action='store_true', help='Close positions at end of day')
    parser.add_argument('--no_market_hours_only', action='store_true', help='Allow trading outside market hours')
    parser.add_argument('--commission', type=float, default=None, help='Commission per trade in dollars (default: None)')
    parser.add_argument('--initial_capital', type=float, default=100000, help='Initial capital for backtesting')
    args = parser.parse_args()

    # Logging
    log_file = setup_logging(args.symbol, args.timeframe)
    print(f"Logging -> {log_file}")

    # Globals
    market_hours_only = not args.no_market_hours_only
    close_at_eod = args.close_at_eod
    commission = args.commission
    initial_capital = args.initial_capital

    # Trials pkl
    os.makedirs('logs', exist_ok=True)
    pkl_filename = os.path.join('logs', f"jfk_dsrsi_hyperopt_{args.symbol}_{args.timeframe}.pkl")

    # Load or create trials
    if os.path.exists(pkl_filename):
        with open(pkl_filename, "rb") as f:
            trials = pickle.load(f)
        print(f"Loaded {len(trials.trials)} prior trials from {pkl_filename}")
        total_evals = len(trials.trials) + args.max_evals
    else:
        trials = Trials()
        total_evals = args.max_evals

    # Data loading
    print(f"Loading data: {args.datafile}")
    data = load_data(args.datafile, args.start_date, args.end_date)
    if args.resample:
        print(f"Resampling -> {args.resample}")
        data = resample_data(data, args.resample)
    print(f"Data bars: {len(data)} from {data.index[0]} to {data.index[-1]}")

    # Search space
    space = {
        # Mode
        'use_dsrsi': hp.choice('use_dsrsi', [False, True]),

        # DSRSI
        'source': hp.choice('source', ['open', 'high', 'low', 'close']),
        'dsrsi_length': hp.quniform('dsrsi_length', 5, 30, 1),
        'smoothing_period': hp.quniform('smoothing_period', 1, 10, 1),
        'volume_weighted': hp.choice('volume_weighted', [False, True]),

        # JFKPS core
        'pst_length': hp.quniform('pst_length', 5, 20, 1),
        'pst_x': hp.quniform('pst_x', 3, 8, 1),
        'pst_smooth': hp.quniform('pst_smooth', 1, 6, 1),
        'kps_smooth_period': hp.quniform('kps_smooth_period', 5, 20, 1),
        'jphase': hp.uniform('jphase', -100.0, 100.0),

        # Exits (KPS thresholds)
        'kps_long_exit_threshold': hp.quniform('kps_long_exit_threshold', 70.0, 95.0, 1),
        'kps_short_exit_threshold': hp.quniform('kps_short_exit_threshold', 5.0, 30.0, 1),

        # Risk management
        'atr_length': hp.quniform('atr_length', 5, 30, 1),
        'sl_atr_ratio': hp.uniform('sl_atr_ratio', 0.5, 3.0),
        'tp_sl_ratio': hp.uniform('tp_sl_ratio', 1.0, 3.0),
        'use_sl': hp.choice('use_sl', [False, True]),
        'use_tp': hp.choice('use_tp', [False, True]),
        'use_ts': hp.choice('use_ts', [False, True]),
        'ts_swing_lookback': hp.quniform('ts_swing_lookback', 5, 20, 1),
        'ts_atr_multiplier': hp.uniform('ts_atr_multiplier', 0.5, 2.0),
        'ts_method': hp.choice('ts_method', ['ATR', 'Percent']),
        'ts_source': hp.choice('ts_source', ['Open', 'Close', 'SwingHL']),
        'ts_percent': hp.uniform('ts_percent', 1.0, 5.0),
        'exit_at_opposite_signal': hp.choice('exit_at_opposite_signal', [False, True]),
    }

    print(f"\nStarting optimization: total_evals={total_evals}")
    print("=" * 80)

    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=total_evals, trials=trials, rstate=np.random.default_rng(42))

    # Save trials
    with open(pkl_filename, "wb") as f:
        pickle.dump(trials, f)
    print(f"\nTrials saved -> {pkl_filename}")

    # Convert categorical indices back to values
    # (Only necessary if you want to run a final backtest with exact chosen params)
    cat_maps = {
        'use_dsrsi': [False, True],
        'source': ['open', 'high', 'low', 'close'],
        'volume_weighted': [False, True],
        'use_sl': [False, True],
        'use_tp': [False, True],
        'use_ts': [False, True],
        'ts_method': ['ATR', 'Percent'],
        'ts_source': ['Open', 'Close', 'SwingHL'],
        'exit_at_opposite_signal': [False, True],
    }
    best_params = {}
    for k, v in best.items():
        if k in cat_maps:
            best_params[k] = cat_maps[k][int(v)]
        else:
            best_params[k] = float(v)

    # Fill any missing fields due to casting
    for key in ['dsrsi_length', 'smoothing_period', 'pst_length', 'pst_x', 'pst_smooth', 'kps_smooth_period',
                'atr_length', 'ts_swing_lookback', 'kps_long_exit_threshold', 'kps_short_exit_threshold']:
        if key in best_params:
            best_params[key] = int(round(best_params[key]))

    print("\nBest parameter indices/values:")
    for k, v in best.items():
        print(f"  {k}: {v}")
    print("\nDecoded best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    # Final backtest with best params (optional display)
    print("\nRunning final backtest with best parameters...")
    final = run_backtest(
        data, {**best_params},
        initial_capital=initial_capital,
        commission=commission,
        market_hours_only=market_hours_only,
        close_at_eod=close_at_eod
    )
    print("\nFinal performance:")
    for k, v in final.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  combined_score: {combined_score(final):.2f}")
