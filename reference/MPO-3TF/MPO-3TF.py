#!/usr/bin/env python3
"""
MPO-3TF.py – Optuna optimizer + walk-forward for "MPO 3TF" strategy (NumPy/optional Numba)

Fixes in this revision:
- Resampling: use '5min','10min','15min' (avoid deprecated 'T' alias).
- report_by_day: robust handling of tz-aware DateTimeIndex (no .astype('datetime64[D]')).
"""

import argparse
import logging
from typing import Tuple, Dict, Any, Optional, Callable, List

import pandas as pd
import numpy as np
import optuna

# ── Numba (optional) ───────────────────────────────────────────
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(func=None, **kwargs):
        if callable(func):
            return func
        def decorator(f):
            return f
        return decorator

# ────────────────────────────────────────────────────────────────
# Logging
# ────────────────────────────────────────────────────────────────

def setup_logger() -> logging.Logger:
    logger = logging.getLogger("MPO-3TF")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger

# ────────────────────────────────────────────────────────────────
#  Resampling (tema_backtest_v3 aggregation policy)
# ────────────────────────────────────────────────────────────────

def resample_ibkr(df: pd.DataFrame, timeframe: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Resample to timeframe like '5min','10min','15min','60min'. Carries over avg/barCount when present."""
    # Canonicalize legacy alias
    if timeframe.endswith('T'):
        timeframe = timeframe.replace('T', 'min')
    if logger:
        logger.info(f"Resampling data to {timeframe}")
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    if 'average' in df.columns:
        ohlc_dict['average'] = 'mean'
    if 'barCount' in df.columns:
        ohlc_dict['barCount'] = 'sum'

    res = df.resample(timeframe).agg(ohlc_dict).dropna()
    return res

def infer_bar_minutes(df: pd.DataFrame) -> int:
    d = (df.index.to_series().diff().dropna().median())
    return int(round(d.total_seconds() / 60.0))

# ────────────────────────────────────────────────────────────────
#  Indicators – RSI, Stoch (K), MFI, ATR, MB-RSI
# ────────────────────────────────────────────────────────────────

def rma(series: pd.Series, length: int) -> pd.Series:
    alpha = 1.0 / max(1, length)
    return series.ewm(alpha=alpha, adjust=False).mean()

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(50.0)

def stoch_k(close: pd.Series, high: pd.Series, low: pd.Series, length: int = 14) -> pd.Series:
    ll = low.rolling(window=length, min_periods=length).min()
    hh = high.rolling(window=length, min_periods=length).max()
    k = 100.0 * (close - ll) / (hh - ll).replace(0.0, np.nan)
    return k.fillna(50.0)

def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    mf = tp * volume
    delta_tp = tp.diff()
    pos_mf = mf.where(delta_tp > 0.0, 0.0)
    neg_mf = mf.where(delta_tp < 0.0, 0.0)
    pos = pos_mf.rolling(window=length, min_periods=length).sum()
    neg = neg_mf.rolling(window=length, min_periods=length).sum()
    mfr = pos / neg.replace(0.0, np.nan)
    out = 100.0 - (100.0 / (1.0 + mfr))
    out = out.replace([np.inf, -np.inf], np.nan).fillna(50.0)
    return out

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high_low  = df['high'] - df['low']
    high_prev = (df['high'] - df['close'].shift()).abs()
    low_prev  = (df['low']  - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)
    return tr.rolling(window=length, min_periods=length).mean()

def mpo_from_df(df: pd.DataFrame, length: int = 14) -> pd.Series:
    r = rsi(df['close'], length)
    s = stoch_k(df['close'], df['high'], df['low'], length)
    m = mfi(df['high'], df['low'], df['close'], df.get('volume', pd.Series(0.0, index=df.index)), length)
    return (r + s + m) / 3.0

def mbrsi_on_df(df: pd.DataFrame, rsi_len: int, fast: int, slow: int) -> pd.Series:
    r = rsi(df['close'], rsi_len)
    f = df['close'].ewm(span=fast, adjust=False).mean()
    s = df['close'].ewm(span=slow, adjust=False).mean()
    val = r * (f / s.replace(0.0, np.nan))
    return val.fillna(50.0)

# ────────────────────────────────────────────────────────────────
#  Multi-timeframe prep (5/10/15m) – resample ONCE then align to 5m
# ────────────────────────────────────────────────────────────────

def prepare_multi_tf(
    df_raw: pd.DataFrame,
    logger: logging.Logger,
    base_minutes: int = 5,
    use_mbrsi_gate: bool = False,
    mbrsi_fast: int = 9, mbrsi_slow: int = 21, mbrsi_rsi_len: int = 12
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (df5, df10, df15, unified5) where:
      - df5/df10/df15 are resampled OHLCV frames
      - unified5 is df5 with columns: atr5, mpo5, mpo10_on_5, mpo15_on_5, (mbrsi15_on_5 if gate)
    """
    in_tf = infer_bar_minutes(df_raw)
    logger.info(f"Inferred input bar size: {in_tf} minutes")

    if in_tf == 1:
        df1 = df_raw
        df5  = resample_ibkr(df1, '5min', logger)
        df10 = resample_ibkr(df1, '10min', logger)
        df15 = resample_ibkr(df1, '15min', logger)
    elif in_tf == 5:
        df5  = df_raw
        df10 = resample_ibkr(df5, '10min', logger)
        df15 = resample_ibkr(df5, '15min', logger)
    else:
        raise ValueError("Only 1-minute or 5-minute input CSVs are supported for this script.")

    # Indicators per TF
    df5['atr5'] = atr(df5, 14)
    df5['mpo5'] = mpo_from_df(df5, 14)

    df10['mpo10'] = mpo_from_df(df10, 14)
    df15['mpo15'] = mpo_from_df(df15, 14)

    # Optional MB-RSI on TF3
    if use_mbrsi_gate:
        df15['mbrsi15'] = mbrsi_on_df(df15, mbrsi_rsi_len, mbrsi_fast, mbrsi_slow)

    # Align to 5m index by forward fill (non-repainting HTF values)
    unified = df5[['open','high','low','close','atr5','mpo5']].copy()
    unified['mpo10_on_5'] = df10['mpo10'].reindex(unified.index).ffill()
    unified['mpo15_on_5'] = df15['mpo15'].reindex(unified.index).ffill()
    if use_mbrsi_gate:
        unified['mbrsi15_on_5'] = df15['mbrsi15'].reindex(unified.index).ffill()

    return df5, df10, df15, unified.dropna()

# ────────────────────────────────────────────────────────────────
#  Data I/O & window prep
# ────────────────────────────────────────────────────────────────

def load_data(csv_file: str) -> pd.DataFrame:
    df = pd.read_csv(csv_file, parse_dates=['date'])
    # Ensure UTC tz-aware index
    if df['date'].dt.tz is None:
        df['date'] = df['date'].dt.tz_localize('UTC')
    else:
        df['date'] = df['date'].dt.tz_convert('UTC')
    df = df.sort_values('date').set_index('date')
    df.columns = [c.lower() for c in df.columns]
    return df

def last_rth_open(bar_minutes: int) -> int:
    """Minute-of-day for the bar whose OPEN is the last tradable before 16:00 ET (20:00 UTC)."""
    end_minutes = 20 * 60  # 20:00 UTC
    return end_minutes - bar_minutes

def prep_window_multi(
    df5_all: pd.DataFrame,
    df10_all: pd.DataFrame,
    df15_all: pd.DataFrame,
    unified5_all: pd.DataFrame,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    bar_minutes: int,
    warmup_bars: int = 60,
) -> pd.DataFrame:
    """Slice [start_dt, end_dt) on the unified 5m frame. (Indicators already computed)."""
    if start_dt.tzinfo is None:
        start_dt = start_dt.tz_localize('UTC')
    if end_dt.tzinfo is None:
        end_dt = end_dt.tz_localize('UTC')

    warmup = pd.Timedelta(minutes=bar_minutes * warmup_bars)
    win = unified5_all[(unified5_all.index >= (start_dt - warmup)) & (unified5_all.index < end_dt)].copy()
    return win[win.index >= start_dt]

# ────────────────────────────────────────────────────────────────
#  Masks & arrays for backtest
# ────────────────────────────────────────────────────────────────

def build_runtime_arrays(df: pd.DataFrame, symbol: str, bar_minutes: int, rth_only: bool,
                         report_by_day: bool, report_interval_bars: int):
    """Return arrays and masks for fast loop."""
    o = df['open' ].to_numpy(float)
    h = df['high' ].to_numpy(float)
    l = df['low'  ].to_numpy(float)
    c = df['close'].to_numpy(float)
    a = df['atr5' ].to_numpy(float)
    osc1 = df['mpo5'       ].to_numpy(float)
    osc2 = df['mpo10_on_5' ].to_numpy(float)
    osc3 = df['mpo15_on_5' ].to_numpy(float)
    gate = df['mbrsi15_on_5'].to_numpy(float) if 'mbrsi15_on_5' in df.columns else np.full_like(osc1, 50.0)

    idx = df.index

    # --- RTH mask in local ET (DST-safe) ---
    idx_ny = (idx.tz_convert('America/New_York')
              if getattr(idx, 'tz', None) is not None else idx.tz_localize('America/New_York'))
    minutes_local = (idx_ny.hour.values * 60) + idx_ny.minute.values

    if rth_only and symbol.upper() in {"SPY", "TSLA", "SPX"}:
        rth_start_local = 9 * 60 + 30   # 09:30 ET
        rth_end_local   = 16 * 60       # 16:00 ET
        rth_mask = (minutes_local >= rth_start_local) & (minutes_local < rth_end_local)
    else:
        rth_mask = np.ones(len(df), dtype=bool)

    # EOD flatten at the OPEN of the last bar before 16:00 ET
    eod_min_local = (16 * 60) - bar_minutes
    day_exit_open_mask = (minutes_local == eod_min_local)

    # --- Reporting steps: day boundaries in UTC (no DST ambiguity) ---
    report_steps_idx = np.empty(0, dtype=np.int64)
    if report_by_day:
        idx_utc = (idx.tz_convert('UTC') if getattr(idx, 'tz', None) is not None else idx.tz_localize('UTC'))
        idx_utc_naive = idx_utc.tz_localize(None) if idx_utc.tz is not None else idx_utc
        days = idx_utc_naive.values.astype('datetime64[D]')
        change = np.nonzero(days[1:] != days[:-1])[0] + 1
        report_steps_idx = change.astype(np.int64)

    if report_interval_bars and report_interval_bars > 0:
        extra = np.arange(report_interval_bars, len(df), report_interval_bars, dtype=np.int64)
        report_steps_idx = np.unique(np.concatenate([report_steps_idx, extra]))

    return (o, h, l, c, a, osc1, osc2, osc3, gate,
            rth_mask.astype(np.uint8), day_exit_open_mask.astype(np.uint8), report_steps_idx)

# ────────────────────────────────────────────────────────────────
#  Cross helpers (NumPy friendly)
# ────────────────────────────────────────────────────────────────

@njit(cache=True)
def _cross_over(prev: float, curr: float, level: float) -> bool:
    return (prev < level) and (curr >= level)

@njit(cache=True)
def _cross_under(prev: float, curr: float, level: float) -> bool:
    return (prev > level) and (curr <= level)

# ────────────────────────────────────────────────────────────────
#  Fast backtest – MPO 3TF state machine (no visuals)
# ────────────────────────────────────────────────────────────────

def backtest_numpy(
    arrays: Tuple[np.ndarray, ...],
    # MPO params
    use_entry1: bool, use_entry2: bool, entry2_min_lm: int,
    ob1: float, ob2: float, ob3: float, os1: float, os2: float, os3: float,
    use_mbrsi_gate: bool, mbrsi_thresh: float,
    # exits
    sl_mult: float, tp_mult: float,
    dynamic_atr: bool, trail_mult: float,
    # misc
    min_bars_warmup: int = 20,
    progress_cb: Optional[Callable[[int, float, float], None]] = None,
    report_steps_idx: Optional[np.ndarray] = None
) -> Dict[str, float]:
    (o, h, l, c, a, osc1, osc2, osc3, gate, rth_mask_u8, eod_open_u8, steps_idx) = arrays
    n = o.shape[0]
    if report_steps_idx is not None:
        steps_idx = report_steps_idx
    rth_mask = rth_mask_u8.astype(np.bool_)
    eod_open = eod_open_u8.astype(np.bool_)

    # state machine
    ST_IDLE, ST_LOOK_LONG1, ST_LOOK_SHORT1, ST_LOOK_LONG2, ST_LOOK_SHORT2 = 0,1,2,3,4
    state = ST_IDLE
    ob1Ar, ob2Ar, ob3Ar = False, False, False
    os1Ar, os2Ar, os3Ar = False, False, False

    # position
    pos = 0           # 0 flat, +1 long, -1 short
    queued = 0        # queued direction for next bar open
    stop_price = 0.0
    take_price = 0.0
    entry_price = 0.0
    entry_idx = -10_000
    best_high = 0.0
    best_low = 1e18

    # stats
    gross_profit = 0.0
    gross_loss   = 0.0  # negative sums
    net_profit   = 0.0
    closed       = 0
    wins         = 0
    equity       = 0.0
    peak         = 0.0
    max_dd       = 0.0

    step_ptr = 0
    next_step = steps_idx[0] if steps_idx.size > 0 else -1

    for i in range(1, n):
        # 0) Exit at EOD open
        if pos != 0 and eod_open[i]:
            exit_price = o[i]
            pnl = (exit_price - entry_price) if pos == 1 else (entry_price - exit_price)
            net_profit += pnl
            if pnl >= 0.0: gross_profit += pnl; wins += 1
            else: gross_loss += pnl
            closed += 1; equity += pnl
            if equity > peak: peak = equity
            dd = equity - peak
            if dd < max_dd: max_dd = dd
            pos = 0

        # 1) Progress callback
        if progress_cb is not None and i == next_step:
            pf_so_far = (abs(gross_profit / gross_loss) if gross_loss != 0.0 else 1e10)
            if pf_so_far > 1e9:
                pf_so_far = 10.0
            progress_cb(step_ptr + 1, pf_so_far, net_profit)
            step_ptr += 1
            next_step = steps_idx[step_ptr] if step_ptr < steps_idx.size else -1

        # 2) Skip non-RTH
        if not rth_mask[i]:
            continue

        # 3) Warmup gate (reset state/arms when warming or outside RTH)
        if i < min_bars_warmup:
            state = ST_IDLE
            ob1Ar = ob2Ar = ob3Ar = False
            os1Ar = os2Ar = os3Ar = False
            continue

        
        # 4) Active position: update ATR-based brackets (dynamic/trailing) and check exits
        if pos != 0:
            # Update best favorable excursion
            if pos == 1:
                if h[i] > best_high:
                    best_high = h[i]
            else:
                if l[i] < best_low:
                    best_low = l[i]

            # Recompute brackets if dynamic ATR or if trailing is requested
            atr_now_live = a[i]
            if dynamic_atr:
                if pos == 1:
                    stop_price = max(entry_price - sl_mult*atr_now_live,
                                     (best_high - trail_mult*atr_now_live) if trail_mult > 0 else -1e18)
                    take_price = entry_price + tp_mult*atr_now_live
                else:
                    stop_price = min(entry_price + sl_mult*atr_now_live,
                                     (best_low + trail_mult*atr_now_live) if trail_mult > 0 else 1e18)
                    take_price = entry_price - tp_mult*atr_now_live
            else:
                # Static brackets from entry, but allow trailing stop to ratchet using current ATR
                if trail_mult > 0:
                    if pos == 1:
                        trail_candidate = best_high - trail_mult*atr_now_live
                        if trail_candidate > stop_price:
                            stop_price = trail_candidate
                    else:
                        trail_candidate = best_low + trail_mult*atr_now_live
                        if trail_candidate < stop_price:
                            stop_price = trail_candidate

            if pos == 1:
                if l[i] <= stop_price:
                    exit_price = stop_price
                elif h[i] >= take_price:
                    exit_price = take_price
                else:
                    exit_price = None
                if exit_price is not None:
                    pnl = exit_price - entry_price
                    net_profit += pnl
                    if pnl >= 0.0:
                        gross_profit += pnl; wins += 1
                    else:
                        gross_loss += pnl
                    closed += 1; equity += pnl
                    if equity > peak: peak = equity
                    dd = equity - peak
                    if dd < max_dd: max_dd = dd
                    pos = 0
            else:  # short
                if h[i] >= stop_price:
                    exit_price = stop_price
                elif l[i] <= take_price:
                    exit_price = take_price
                else:
                    exit_price = None
                if exit_price is not None:
                    pnl = entry_price - exit_price
                    net_profit += pnl
                    if pnl >= 0.0:
                        gross_profit += pnl; wins += 1
                    else:
                        gross_loss += pnl
                    closed += 1; equity += pnl
                    if equity > peak: peak = equity
                    dd = equity - peak
                    if dd < max_dd: max_dd = dd
                    pos = 0
# 5) Queue entry fills at next bar open
        if pos == 0 and queued != 0:
            pos = queued
            queued = 0
            entry_price = o[i]
            entry_idx = i
            best_high = entry_price
            best_low = entry_price
            atr_now = a[i]
            if pos == 1:
                stop_price = entry_price - sl_mult * atr_now
                take_price = entry_price + tp_mult * atr_now
            else:
                stop_price = entry_price + sl_mult * atr_now
                take_price = entry_price - tp_mult * atr_now
            # Reset arms after entry (mirrors Pine)
            state = ST_IDLE
            ob1Ar = ob2Ar = ob3Ar = False
            os1Ar = os2Ar = os3Ar = False
            continue  # don't evaluate signals on same bar as fill

        # 6) Gates and helpers
        above50_1 = osc1[i] > 50.0
        below50_1 = osc1[i] < 50.0
        above50_2 = osc2[i] > 50.0
        below50_2 = osc2[i] < 50.0
        above50_3 = osc3[i] > 50.0
        below50_3 = osc3[i] < 50.0

        bull_gate = True
        bear_gate = True
        if use_mbrsi_gate:
            g = gate[i]
            bull_gate = (g >= mbrsi_thresh)
            bear_gate = (g <= mbrsi_thresh)

        # 7) Entry 1 bias + pullback (optional)
        if use_entry1 and state == ST_IDLE:
            if below50_3 and (above50_1 or above50_2) and bear_gate:
                state = ST_LOOK_SHORT1
            elif above50_3 and (below50_1 or below50_2) and bull_gate:
                state = ST_LOOK_LONG1
        if state == ST_LOOK_SHORT1 and above50_3:
            state = ST_IDLE
        if state == ST_LOOK_LONG1 and below50_3:
            state = ST_IDLE

        co1_50 = _cross_over(osc1[i-1], osc1[i], 50.0)
        cu1_50 = _cross_under(osc1[i-1], osc1[i], 50.0)
        co2_50 = _cross_over(osc2[i-1], osc2[i], 50.0)
        cu2_50 = _cross_under(osc2[i-1], osc2[i], 50.0)

        enterLong_E1 = (use_entry1 and state == ST_LOOK_LONG1 and (co1_50 or co2_50) and bull_gate)
        enterShort_E1 = (use_entry1 and state == ST_LOOK_SHORT1 and (cu1_50 or cu2_50) and bear_gate)

        # 8) Entry 2 arming (3× extreme fade)
        if use_entry2:
            # inclusive arming
            if (osc1[i-1] < ob1 and osc1[i] >= ob1) or (osc1[i] >= ob1): ob1Ar = True
            if (osc2[i-1] < ob2 and osc2[i] >= ob2) or (osc2[i] >= ob2): ob2Ar = True
            if (osc3[i-1] < ob3 and osc3[i] >= ob3) or (osc3[i] >= ob3): ob3Ar = True

            if (osc1[i-1] > os1 and osc1[i] <= os1) or (osc1[i] <= os1): os1Ar = True
            if (osc2[i-1] > os2 and osc2[i] <= os2) or (osc2[i] <= os2): os2Ar = True
            if (osc3[i-1] > os3 and osc3[i] <= os3) or (osc3[i] <= os3): os3Ar = True

            if state == ST_IDLE:
                if ob1Ar and ob2Ar and ob3Ar and bear_gate:
                    state = ST_LOOK_SHORT2
                if os1Ar and os2Ar and os3Ar and bull_gate:
                    state = ST_LOOK_LONG2

            # Pullback triggers: require N (1..2) of L/M to leave extreme.
            lmUpCount = 0  # leaving oversold -> long
            if _cross_over(osc1[i-1], osc1[i], os1): lmUpCount += 1
            if _cross_over(osc2[i-1], osc2[i], os2): lmUpCount += 1

            lmDnCount = 0  # leaving overbought -> short
            if _cross_under(osc1[i-1], osc1[i], ob1): lmDnCount += 1
            if _cross_under(osc2[i-1], osc2[i], ob2): lmDnCount += 1

            enterLong_E2  = (state == ST_LOOK_LONG2  and (lmUpCount  >= entry2_min_lm) and bull_gate)
            enterShort_E2 = (state == ST_LOOK_SHORT2 and (lmDnCount >= entry2_min_lm) and bear_gate)
        else:
            enterLong_E2 = False
            enterShort_E2 = False

        # 9) Queue entries (filled next bar open)
        if pos == 0:
            if enterLong_E1 or enterLong_E2:
                queued = 1
            elif enterShort_E1 or enterShort_E2:
                queued = -1

    pf = (abs(gross_profit / gross_loss) if gross_loss != 0.0 else (10.0 if gross_profit > 0.0 else 0.0))
    winrate = (wins / closed) * 100.0 if closed > 0 else 0.0
    max_dd_abs = abs(max_dd)
    return {
        'net_profit': float(net_profit),
        'profit_factor': float(pf),
        'closed_trades': int(closed),
        'winrate': float(winrate),
        'max_drawdown': float(max_dd_abs)
    }

# ────────────────────────────────────────────────────────────────
#  Single-run wrapper
# ────────────────────────────────────────────────────────────────

def run_single_backtest_fast(
    df_prepped: pd.DataFrame,
    symbol: str,
    bar_minutes: int,
    strategy_params: Dict[str, Any],
    rth_only: bool = True,
    trial: Optional[optuna.trial.Trial] = None,
    report_by_day: bool = True,
    report_interval_bars: int = 0,
    use_numba: bool = False,
    dynamic_atr: bool = False,
    trail_mult: float = 0.0,
) -> Dict[str, Any]:
    arrays = build_runtime_arrays(df_prepped, symbol, bar_minutes, rth_only,
                                  report_by_day, report_interval_bars)

    steps_idx = arrays[-1]

    def _progress(step_idx: int, pf_so_far: float, net_so_far: float):
        if trial is None:
            return
        trial.report(pf_so_far * net_so_far, int(step_idx))
        if trial.should_prune():
            raise optuna.TrialPruned()

    progress_cb = _progress if (trial is not None and (not use_numba)) else None

    metrics = backtest_numpy(
        arrays=arrays,
        # MPO params
        use_entry1=bool(strategy_params['use_entry1']),
        use_entry2=bool(strategy_params['use_entry2']),
        entry2_min_lm=int(strategy_params['entry2_min_lm']),
        ob1=float(strategy_params['ob1']), ob2=float(strategy_params['ob2']), ob3=float(strategy_params['ob3']),
        os1=float(strategy_params['os1']), os2=float(strategy_params['os2']), os3=float(strategy_params['os3']),
        use_mbrsi_gate=bool(strategy_params['use_mbrsi_gate']),
        mbrsi_thresh=float(strategy_params['mbrsi_thresh']),
        # exits
        sl_mult=float(strategy_params['sl_mult']),
        tp_mult=float(strategy_params['tp_mult']),
        dynamic_atr=dynamic_atr,
        trail_mult=trail_mult,
        # misc
        min_bars_warmup=int(strategy_params['min_bars_warmup']),
        progress_cb=progress_cb,
        report_steps_idx=steps_idx,
    )
    return metrics

# ────────────────────────────────────────────────────────────────
#  Optuna params & objective
# ────────────────────────────────────────────────────────────────

def suggest_params(trial: optuna.trial.Trial) -> Dict[str, Any]:
    return {
        'use_entry1': trial.suggest_categorical('use_entry1', [False, True]),
        'use_entry2': trial.suggest_categorical('use_entry2', [True]),
        'entry2_min_lm': trial.suggest_categorical('entry2_min_lm', [1, 2]),
        # thresholds
        'ob1': trial.suggest_float('ob1', 70.0, 95.0),
        'ob2': trial.suggest_float('ob2', 60.0, 85.0),
        'ob3': trial.suggest_float('ob3', 50.0, 70.0),
        'os1': trial.suggest_float('os1', 5.0, 40.0),
        'os2': trial.suggest_float('os2', 20.0, 50.0),
        'os3': trial.suggest_float('os3', 30.0, 55.0),
        # MB-RSI gate on TF3
        'use_mbrsi_gate': trial.suggest_categorical('use_mbrsi_gate', [False, True]),
        'mbrsi_thresh': trial.suggest_float('mbrsi_thresh', 45.0, 55.0),
        # exits
        'sl_mult': trial.suggest_float('sl_mult', 0.5, 3.0),
        'tp_mult': trial.suggest_float('tp_mult', 1.0, 5.0),
        # misc
        'min_bars_warmup': trial.suggest_int('min_bars_warmup', 20, 100),
    }

def build_objective(
    df_train: pd.DataFrame,
    symbol: str,
    bar_minutes: int,
    rth_only: bool,
    report_by_day: bool,
    report_interval_bars: int,
    use_numba: bool,
    n_startup_trials: int,
    dynamic_atr: bool,
    trail_mult: float,
):
    def objective(trial: optuna.trial.Trial) -> float:
        params = suggest_params(trial)
        try:
            metrics = run_single_backtest_fast(
                df_prepped=df_train,
                symbol=symbol,
                bar_minutes=bar_minutes,
                strategy_params=params,
                rth_only=rth_only,
                trial=(None if use_numba else trial),
                report_by_day=report_by_day,
                report_interval_bars=report_interval_bars,
                use_numba=use_numba,
                dynamic_atr=dynamic_atr,
                trail_mult=trail_mult,
            )
        except optuna.TrialPruned:
            raise

        pf = float(metrics['profit_factor'])
        pnl = float(metrics['net_profit'])
        score = pf * pnl
        trial.set_user_attr('train_pf', pf)
        trial.set_user_attr('train_net_profit', pnl)
        trial.set_user_attr('train_closed', int(metrics['closed_trades']))
        trial.set_user_attr('train_final_value', float(pnl))
        trial.set_user_attr('train_max_drawdown', float(metrics['max_drawdown']))
        return score
    return objective

# ────────────────────────────────────────────────────────────────
#  Walk-forward evaluation (1 out-of-sample window)
# ────────────────────────────────────────────────────────────────

def evaluate_once(
    df_test: pd.DataFrame,
    symbol: str,
    bar_minutes: int,
    best_params: Dict[str, Any],
    rth_only: bool,
    report_by_day: bool,
    report_interval_bars: int,
    use_numba: bool,
    dynamic_atr: bool,
    trail_mult: float,
) -> Dict[str, Any]:
    if df_test.empty:
        raise ValueError("TEST frame is empty. Check your split or provide more data.")

    # Quick visibility on RTH coverage
    arrays = build_runtime_arrays(df_test, symbol, bar_minutes,
                                  rth_only=rth_only, report_by_day=False, report_interval_bars=0)
    rth_mask = arrays[9].astype(bool)
    print(f"TEST window: {df_test.index.min()} → {df_test.index.max()}  (bars={len(df_test)})")
    print("TEST bars total:", len(rth_mask), "RTH bars:", rth_mask.sum())

    return run_single_backtest_fast(
        df_prepped=df_test,
        symbol=symbol,
        bar_minutes=bar_minutes,
        strategy_params=best_params,
        rth_only=rth_only,
        trial=None,
        report_by_day=report_by_day,
        report_interval_bars=report_interval_bars,
        use_numba=use_numba,
        dynamic_atr=dynamic_atr,
        trail_mult=trail_mult,
    )

def compute_train_test_splits(idx: pd.DatetimeIndex,
                              train_years: int,
                              test_years: int,
                              min_test_days: int = 60) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """
    Compute [earliest, train_to, test_from, test_to] ensuring TEST is non-empty.
    Falls back to 'last min_test_days' for TEST if the dataset is short, or finally 80/20 by bars.
    """
    if len(idx) == 0:
        raise ValueError("Index is empty; no data loaded.")

    earliest = idx.min()
    latest   = idx.max()

    total_days = max(1, int((latest - earliest).total_seconds() // 86400))
    desired_train = max(1, 365 * max(0, train_years))
    desired_test  = max(min_test_days, min(365 * max(0, test_years), total_days // 2))

    # First attempt: earliest → earliest+desired_train as train; next desired_test as test
    train_to  = earliest + pd.Timedelta(days=desired_train)
    test_from = train_to
    test_to   = min(test_from + pd.Timedelta(days=desired_test), latest)

    # If TEST collapses to empty, fallback to "last min_test_days"
    if test_from >= test_to:
        test_to   = latest
        test_from = max(earliest, latest - pd.Timedelta(days=min_test_days))
        # If still empty, fallback to 80/20 by bars
        if test_from >= test_to:
            cut = int(0.8 * len(idx))
            cut = max(1, min(cut, len(idx) - 1))
            boundary = idx[cut]
            train_to  = boundary
            test_from = boundary
            test_to   = latest

    return earliest, train_to, test_from, test_to


def make_train_test_frames(unified5_all: pd.DataFrame,
                           train_years: int,
                           test_years: int,
                           min_test_days: int = 60,
                           logger: Optional[logging.Logger] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Slice the pre-computed unified 5m frame into TRAIN/TEST ensuring TEST has bars.
    """
    idx = unified5_all.index
    earliest, train_to, test_from, test_to = compute_train_test_splits(
        idx, train_years=train_years, test_years=test_years, min_test_days=min_test_days
    )

    df_train = unified5_all[(idx >= earliest) & (idx < train_to)].copy()
    df_test  = unified5_all[(idx >= test_from) & (idx < test_to)].copy()

    if logger:
        logger.info(f"Split → TRAIN: {earliest} → {train_to}  (bars={len(df_train)})")
        logger.info(f"Split → TEST : {test_from} → {test_to}  (bars={len(df_test)})")

    if df_test.empty:
        raise ValueError("TEST window is empty even after fallback. Provide more data or reduce --train-years.")

    return df_train, df_test


# ────────────────────────────────────────────────────────────────
#  Main
# ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Fast Optuna optimizer + walk-forward for MPO 3TF (NumPy/Numba)")
    p.add_argument('--data-file', type=str, required=True)
    p.add_argument('--symbol', type=str, default='SPX')
    p.add_argument('--train-years', type=int, default=2)
    p.add_argument('--test-years', type=int, default=1)

    p.add_argument('--optimize', action='store_true')
    p.add_argument('--evaluate', action='store_true')
    p.add_argument('--trials', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-jobs', type=int, default=1)

    p.add_argument('--storage', type=str, default=None, help='sqlite:///optuna.db or postgresql://...')
    p.add_argument('--study-name', type=str, default='MPO-3TF', help='Shared study name for storage-backed runs')

    p.add_argument('--pruner', type=str, default='percentile', choices=['none','median','percentile'])
    p.add_argument('--pruner-percentile', type=float, default=60.0)
    p.add_argument('--n-startup-trials', type=int, default=20)
    p.add_argument('--n-warmup-steps', type=int, default=50)

    p.add_argument('--rth-on', action='store_true', help='Restrict to 09:30–16:00 ET')
    p.add_argument('--report-by-day', action='store_true')
    p.add_argument('--report-interval-bars', type=int, default=0)

    # Exits
    p.add_argument('--dynamic-atr', action='store_true', help='Recompute ATR-based stop/limit each bar while in position')
    p.add_argument('--trail-mult', type=float, default=0.0, help='ATR multiple for trailing stop (0 disables)')

    p.add_argument('--gate-on', action='store_true', help='Enable MB-RSI gate with default params')
    p.add_argument('--gate-fast', type=int, default=9)
    p.add_argument('--gate-slow', type=int, default=21)
    p.add_argument('--gate-rsi-len', type=int, default=12)

    args = p.parse_args()
    logger = setup_logger()
    np.random.seed(args.seed)

    # Load data
    df_raw = load_data(args.data_file)

    # Pre-resample + indicators once
    df5_all, df10_all, df15_all, unified5_all = prepare_multi_tf(
        df_raw, logger,
        base_minutes=5,
        use_mbrsi_gate=args.gate_on,
        mbrsi_fast=args.gate_fast, mbrsi_slow=args.gate_slow, mbrsi_rsi_len=args.gate_rsi_len)

    # Walk-forward split (robust, guarantees non-empty TEST)
    df_train, df_test = make_train_test_frames(
        unified5_all,
        train_years=args.train_years,
        test_years=args.test_years,
        min_test_days=60,
        logger=logger
    )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    if args.pruner == 'none':
        pruner = optuna.pruners.NopPruner()
    elif args.pruner == 'median':
        pruner = optuna.pruners.MedianPruner(n_startup_trials=args.n_startup_trials,
                                             n_warmup_steps=args.n_warmup_steps, interval_steps=1)
    else:
        pruner = optuna.pruners.PercentilePruner(percentile=args.pruner_percentile,
                                                 n_startup_trials=args.n_startup_trials,
                                                 n_warmup_steps=args.n_warmup_steps, interval_steps=1)

    if args.storage:
        storage = optuna.storages.RDBStorage(url=args.storage)
        study = optuna.create_study(direction='maximize', storage=storage, load_if_exists=True,
                                    sampler=sampler, pruner=pruner, study_name=args.study_name)
    else:
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)

    best_params = None

    if args.optimize:
        objective = build_objective(
            df_train=df_train,
            symbol=args.symbol,
            bar_minutes=5,
            rth_only=args.rth_on,
            report_by_day=args.report_by_day,
            report_interval_bars=args.report_interval_bars,
            use_numba=False,
            n_startup_trials=args.n_startup_trials,
            dynamic_atr=args.dynamic_atr,
            trail_mult=args.trail_mult,
        )
        study.optimize(objective, n_trials=args.trials, n_jobs=args.n_jobs)

        best_params = study.best_params
        print("\\nBest trial:")
        print(f"  Value (score): {study.best_value:.4f}")
        print("\nUsing best params (from study) for EVALUATION:")
        for k, v in best_params.items():
            print(f"    {k}: {v}")
        # Final backtest on TRAIN with the best params (collect richer stats)
        final_train = run_single_backtest_fast(
            df_prepped=df_train,
            symbol=args.symbol,
            bar_minutes=5,
            strategy_params=best_params,
            rth_only=args.rth_on,
            trial=None,
            report_by_day=args.report_by_day,
            report_interval_bars=args.report_interval_bars,
            use_numba=False,
            dynamic_atr=args.dynamic_atr,
            trail_mult=args.trail_mult,
        )
        print("\nFinal TRAIN metrics with best params:")
        for k, v in final_train.items():
            print(f"  {k}: {v}")


    if args.evaluate:
        if best_params is None:
            if len(study.trials) == 0:
                raise RuntimeError("No optimized parameters found. Run with --optimize first.")
            best_params = study.best_params

        metrics = evaluate_once(
            df_test=df_test,
            symbol=args.symbol,
            bar_minutes=5,
            best_params=best_params,
            rth_only=args.rth_on,
            report_by_day=args.report_by_day,
            report_interval_bars=args.report_interval_bars,
            use_numba=False,
            dynamic_atr=args.dynamic_atr,
            trail_mult=args.trail_mult,
        )
        print("\\nEvaluation (OOS) metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
