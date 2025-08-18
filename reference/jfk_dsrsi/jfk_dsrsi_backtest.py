#!/usr/bin/env python3
"""
jfk_dsrsi_backtest.py - JFKPS + DSRSI Strategy Backtest (Market Orders)

A faithful Python port of Loxx's "Jurik-Filtered Kase Permission Stochastic"
with an optional DSRSI (Double Smoothed RSI) overlay for entries — mirroring
the provided PineScript strategy.

WHAT'S NEW (vs your previous version):
- Adds a full Kase Permission Stochastic (KPS) implementation:
  * TripleK / TripleDF / TripleDS (synthetic frame)
  * Jurik filter smoothing for pstBuffer (fast) and pssBuffer (slow)
  * Accurate crossover/trend & threshold exits like Pine
- Aligns entries/exits with Pine modes:
  * Mode A (KPS-only): entries on KPS crossovers; exits on KPS thresholds
  * Mode B (Use DSRSI): entries require DSRSI zero-cross + KPS trend filter;
    exits: KPS threshold OR KPS trend flip
- Keeps robust ATR-based SL/TP/Trailing with no look-ahead (ATR shifted)
- Market-order execution at next bar's open to avoid look-ahead

References:
- Loxx: Jurik-Filtered Kase Permission Stochastic (TV id: kIBAXchQ).
- Kase Permission Stochastic concepts and thresholds (85/15). 
- Jurik filter (JMA-like) implementation adapted from tradingstrategy.ai / pandas_ta.

Citations:
- https://www.tradingview.com/script/kIBAXchQ-Jurik-Filtered-Kase-Permission-Stochastic-Loxx/
- https://www.kaseco.com/wp-content/uploads/2016/03/Kase-StatWare-Manual-CQG.pdf
- https://tradingstrategy.ai/docs/_modules/pandas_ta/overlap/jma.html

"""

import argparse
import logging
import os
import json
import pandas as pd
import numpy as np
import talib
import pytz
from datetime import time
from typing import List

# ────────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────────
ET = pytz.timezone('US/Eastern')
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
EOD_CLOSE = time(15, 59)  # 3:59 PM ET - close positions before market close

# DSRSI defaults (used if DSRSI mode enabled)
SOURCE = 'close'          # Price column for DSRSI
DSRSI_LENGTH = 14
SMOOTHING_PERIOD = 3
VOLUME_WEIGHTED = False

# KPS (JFKPS) defaults matching Pine
PST_LENGTH = 9           # "Period" in Pine
PST_X = 5                # "Synthetic Multiplier"
PST_SMOOTH = 3           # "Stochastic Smooth Period" (alpha = 2/(1+PST_SMOOTH))
SMOOTH_PERIOD = 10       # Jurik smoothing period
JPHASE = 0.0             # Jurik phase

# Exit thresholds (KPS pstBuffer thresholds per Kase guidance)
KPS_LONG_EXIT_THRESHOLD = 85.0
KPS_SHORT_EXIT_THRESHOLD = 15.0

# Risk management defaults
ATR_LENGTH = 14
SL_ATR_RATIO = 2.0
TP_SL_RATIO = 2.0
USE_SL = True
USE_TP = True
USE_TS = False
TS_SWING_LOOKBACK = 10
TS_ATR_MULTIPLIER = 1.0
TS_METHOD = 'ATR'      # 'ATR' or 'Percent'
TS_SOURCE = 'Close'    # 'Open', 'Close', or 'SwingHL'
TS_PERCENT = 2.0
EXIT_AT_OPPOSITE_SIGNAL = True  # exit on opposite signal (mode-aware)

# ────────────────────────────────────────────────────────────────────────────────
# Imports from your shared infrastructure
# ────────────────────────────────────────────────────────────────────────────────
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tema_backtest_v3 import (
    setup_logger, parse_arguments, load_data, filter_market_hours,
    resample_data, validate_data, extract_timeframe_from_filename,
    calculate_position_size, calculate_performance_metrics
)

# ────────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────────
def crossover(prev_a, prev_b, a, b) -> bool:
    """True when a crosses above b (prev_a <= prev_b and a > b)."""
    return (prev_a <= prev_b) and (a > b)

def crossunder(prev_a, prev_b, a, b) -> bool:
    """True when a crosses below b (prev_a >= prev_b and a < b)."""
    return (prev_a >= prev_b) and (a < b)

# ────────────────────────────────────────────────────────────────────────────────
# Jurik Filter (JMA-style) implementation
# Adapted from tradingstrategy.ai pandas_ta JMA module for fidelity.
# Source: https://tradingstrategy.ai/docs/_modules/pandas_ta/overlap/jma.html
# ────────────────────────────────────────────────────────────────────────────────
def jurik_filter(series: pd.Series, length: int = 7, phase: float = 0.0) -> pd.Series:
    s = series.astype(float).copy()
    if len(s) == 0:
        return s

    # Setup buffers
    jma = np.zeros_like(s.values, dtype=float)
    volty = np.zeros_like(jma)
    v_sum = np.zeros_like(jma)

    kv = det0 = det1 = ma2 = 0.0
    jma[0] = ma1 = uBand = lBand = float(s.iloc[0])

    sum_length = 10
    _length = max(int(length), 1)
    length_half = 0.5 * (_length - 1)
    pr = 0.5 if phase < -100 else 2.5 if phase > 100 else 1.5 + phase * 0.01
    length1 = max((np.log(np.sqrt(length_half)) / np.log(2.0)) + 2.0, 0.0)
    pow1 = max(length1 - 2.0, 0.5)
    length2 = length1 * np.sqrt(length_half) if length_half > 0 else 0.0
    bet = length2 / (length2 + 1) if (length2 + 1) != 0 else 0.0
    beta = 0.45 * (_length - 1) / (0.45 * (_length - 1) + 2.0) if _length > 1 else 0.0

    vals = s.values
    for i in range(1, len(vals)):
        price = float(vals[i])
        # Price volatility using Jurik bands
        del1 = price - uBand
        del2 = price - lBand
        volty[i] = max(abs(del1), abs(del2)) if abs(del1) != abs(del2) else 0.0

        # Relative volatility
        v_sum[i] = v_sum[i - 1] + (volty[i] - volty[max(i - sum_length, 0)]) / sum_length
        avg_volty = float(np.average(v_sum[max(i - 65, 0): i + 1]))
        d_volty = 0.0 if avg_volty == 0 else volty[i] / avg_volty
        r_volty = max(1.0, min(np.power(length1, 1.0 / pow1) if pow1 != 0 else 1.0, d_volty))

        # Update bands
        pow2 = np.power(r_volty, pow1)
        kv = np.power(bet, np.sqrt(pow2)) if bet > 0 else 0.0
        uBand = price if (del1 > 0) else price - (kv * del1)
        lBand = price if (del2 < 0) else price - (kv * del2)

        # Dynamic factor
        power = np.power(r_volty, pow1)
        alpha = np.power(beta, power) if beta > 0 else 0.0

        # Stage 1: adaptive EMA
        ma1 = ((1 - alpha) * price) + (alpha * ma1)
        # Stage 2: Kalman-like
        det0 = ((price - ma1) * (1 - beta)) + (beta * det0)
        ma2 = ma1 + pr * det0
        # Stage 3: final Jurik smoothing
        det1 = ((ma2 - jma[i - 1]) * (1 - alpha) * (1 - alpha)) + (alpha * alpha * det1)
        jma[i] = jma[i - 1] + det1

    # Remove initial warmup
    jma[:_length - 1] = np.nan
    out = pd.Series(jma, index=s.index, name=f"JMA_{length}_{phase}")
    return out

# ────────────────────────────────────────────────────────────────────────────────
# DSRSI (Double Smoothed RSI) calculation
# ────────────────────────────────────────────────────────────────────────────────
def calculate_wma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    N = len(values)
    if N == 0:
        return 0.0
    weights = np.arange(1, N + 1, dtype=float)
    return float(np.dot(values, weights) / weights.sum())

def calculate_dsrsi_indicators(
    df: pd.DataFrame,
    source: str = SOURCE,
    length: int = DSRSI_LENGTH,
    smoothing_period: int = SMOOTHING_PERIOD,
    volume_weighted: bool = VOLUME_WEIGHTED
) -> pd.DataFrame:
    """Calculate DSRSI adjusted (-100..100) and smoothed series."""
    logger = logging.getLogger(__name__)
    logger.info(f"Calculating DSRSI (src={source}, len={length}, smooth={smoothing_period}, vw={volume_weighted})")
    out = df.copy()

    out['adjusted_rsi'] = np.nan
    out['smoothed_rsi'] = np.nan

    if len(out) < length + smoothing_period:
        return out

    adjusted_vals = []
    src = out[source].values
    vol = out['volume'].values if ('volume' in out.columns and volume_weighted) else np.ones(len(out))

    for i in range(1, len(out)):
        # rolling window [i-length+1 .. i]
        start = max(1, i - length + 1)
        ups = []
        dns = []
        for j in range(start, i + 1):
            delta = src[j] - src[j - 1]
            wt = vol[j - 1] if volume_weighted else 1.0
            wd = delta * wt
            ups.append(max(wd, 0.0))
            dns.append(max(-wd, 0.0))
        if ups and dns:
            up_wma = calculate_wma(ups)
            dn_wma = calculate_wma(dns)
            if dn_wma == 0:
                raw = 100.0
            elif up_wma == 0:
                raw = 0.0
            else:
                raw = 100.0 - 100.0 / (1.0 + up_wma / dn_wma)
            adj = raw * 2.0 - 100.0
            out.iloc[i, out.columns.get_loc('adjusted_rsi')] = adj
            adjusted_vals.append(adj)
            if len(adjusted_vals) >= smoothing_period:
                sm = calculate_wma(adjusted_vals[-smoothing_period:])
                out.iloc[i, out.columns.get_loc('smoothed_rsi')] = sm

    # fill early NaNs of smoothed with adjusted
    out['smoothed_rsi'] = out['smoothed_rsi'].fillna(out['adjusted_rsi'])
    return out

# ────────────────────────────────────────────────────────────────────────────────
# JFKPS (Kase Permission Stochastic) calculation
# Mirrors the Pine logic: TripleK / TripleDF / TripleDS, then SMA(3) + Jurik filter
# for pstBuffer (fast) and pssBuffer (slow). Also provides trend & cross signals.
# ────────────────────────────────────────────────────────────────────────────────
def calculate_kps_indicators(
    df: pd.DataFrame,
    pst_length: int = PST_LENGTH,
    pst_x: int = PST_X,
    pst_smooth: int = PST_SMOOTH,
    smooth_period: int = SMOOTH_PERIOD,
    jphase: float = JPHASE
) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      - triple_k, triple_df, triple_ds
      - pst_buffer (fast, Jurik filtered), pss_buffer (slow, Jurik filtered)
      - trend (1 / -1 / 0), kps_go_long, kps_go_short
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Calculating JFKPS: length={pst_length}, X={pst_x}, smooth={pst_smooth}, jlen={smooth_period}, jphase={jphase}")

    out = df.copy()
    lb = pst_length * pst_x
    if lb < 1:
        lb = 1

    # TripleK
    fmin = out['low'].rolling(lb, min_periods=1).min()
    fmax = out['high'].rolling(lb, min_periods=1).max() - fmin
    triple_k = np.where(fmax > 0, 100.0 * (out['close'] - fmin) / fmax, 0.0)
    out['triple_k'] = triple_k

    # TripleDF / TripleDS recursions with pst_x-step memory
    alpha = 2.0 / (1.0 + float(pst_smooth))
    df_vals = np.zeros(len(out), dtype=float)
    ds_vals = np.zeros(len(out), dtype=float)

    for i in range(len(out)):
        prev_df = df_vals[i - pst_x] if i - pst_x >= 0 else 0.0
        prev_ds = ds_vals[i - pst_x] if i - pst_x >= 0 else 0.0
        df_vals[i] = prev_df + alpha * (triple_k[i] - prev_df)
        ds_vals[i] = (prev_ds * 2.0 + df_vals[i]) / 3.0

    out['triple_df'] = df_vals
    out['triple_ds'] = ds_vals

    # SMA(3) of each, then Jurik filter
    out['triple_df_s'] = pd.Series(df_vals, index=out.index).rolling(3, min_periods=1).mean()
    out['triple_ds_s'] = pd.Series(ds_vals, index=out.index).rolling(3, min_periods=1).mean()

    out['pst_buffer'] = jurik_filter(out['triple_df_s'], length=smooth_period, phase=jphase)
    out['pss_buffer'] = jurik_filter(out['triple_ds_s'], length=smooth_period, phase=jphase)

    # Trend: persistent regime as in Pine
    trend = np.zeros(len(out), dtype=int)
    for i in range(len(out)):
        if i == 0:
            trend[i] = 0
        else:
            trend[i] = trend[i - 1]
        a = out['pst_buffer'].iloc[i]
        b = out['pss_buffer'].iloc[i]
        if not np.isnan(a) and not np.isnan(b):
            if a > b:
                trend[i] = 1
            elif a < b:
                trend[i] = -1
    out['trend'] = trend

    # Cross signals (KPS)
    pst = out['pst_buffer']
    pss = out['pss_buffer']
    prev_pst = pst.shift(1)
    prev_pss = pss.shift(1)
    out['kps_go_long'] = (prev_pst <= prev_pss) & (pst > pss)
    out['kps_go_short'] = (prev_pst >= prev_pss) & (pst < pss)

    return out

# ────────────────────────────────────────────────────────────────────────────────
# Signal Generation (mode-aware)
# ────────────────────────────────────────────────────────────────────────────────
def generate_signals(
    df: pd.DataFrame,
    use_dsrsi: bool = False,
    market_hours_only: bool = True,
    atr_length: int = ATR_LENGTH,
    sl_atr_ratio: float = SL_ATR_RATIO,
    tp_sl_ratio: float = TP_SL_RATIO
) -> pd.DataFrame:
    """
    Returns a DataFrame with:
      - ATR (and shifted)
      - KPS buffers, trend, KPS entries
      - DSRSI adjusted/smoothed, DSRSI entries
      - long_stop / short_stop / long_target / short_target
      - mode-aware "entry_long" / "entry_short" booleans
    """
    signals = df.copy()

    # ATR for risk management (shifted to avoid look-ahead)
    signals['atr'] = talib.ATR(signals['high'], signals['low'], signals['close'], timeperiod=atr_length)
    signals['atr_shifted'] = signals['atr'].shift(1)

    # KPS entries already computed in df (ensure present)
    if 'kps_go_long' not in signals.columns or 'trend' not in signals.columns:
        raise RuntimeError("KPS indicators missing: call calculate_kps_indicators first.")

    # DSRSI entries if present (we computed them in df)
    if 'smoothed_rsi' not in signals.columns:
        signals['adjusted_rsi'] = np.nan
        signals['smoothed_rsi'] = np.nan

    signals['dsrsi_long_signal'] = (signals['smoothed_rsi'].shift(1) < 0) & (signals['smoothed_rsi'] >= 0)
    signals['dsrsi_short_signal'] = (signals['smoothed_rsi'].shift(1) > 0) & (signals['smoothed_rsi'] <= 0)

    # Mode-aware entries
    if use_dsrsi:
        # Pine: long only if KPS trend==1 AND DSRSI crosses up; short only if trend==-1 and DSRSI crosses down
        signals['entry_long'] = (signals['trend'] == 1) & signals['dsrsi_long_signal']
        signals['entry_short'] = (signals['trend'] == -1) & signals['dsrsi_short_signal']
    else:
        # KPS-only mode: traditional JFKPS crossovers
        signals['entry_long'] = signals['kps_go_long']
        signals['entry_short'] = signals['kps_go_short']

    # Market hours filter
    if market_hours_only:
        idx_et = signals.index.tz_convert(ET) if signals.index.tz is not None else signals.index.tz_localize('UTC', ambiguous=False).tz_convert(ET)
        mask = (idx_et.time >= MARKET_OPEN) & (idx_et.time <= MARKET_CLOSE)
        signals.loc[~mask, ['entry_long', 'entry_short']] = False

    # Risk scaffolding: stops/targets based on shifted ATR
    signals['long_stop'] = signals['close'] - (signals['atr_shifted'] * sl_atr_ratio)
    signals['short_stop'] = signals['close'] + (signals['atr_shifted'] * sl_atr_ratio)
    sl_long = signals['close'] - signals['long_stop']
    sl_short = signals['short_stop'] - signals['close']
    signals['long_target'] = signals['close'] + sl_long * tp_sl_ratio
    signals['short_target'] = signals['close'] - sl_short * tp_sl_ratio

    return signals

# ────────────────────────────────────────────────────────────────────────────────
# Position with mode-aware exits (KPS thresholds + trend flip; optional RSI exits)
# ────────────────────────────────────────────────────────────────────────────────
class Position:
    def __init__(self, symbol: str, side: str, entry_price: float, stop_loss: float,
                 take_profit: float, shares: int, entry_time,
                 use_sl: bool = True, use_tp: bool = True, use_ts: bool = False):
        self.symbol = symbol
        self.side = side  # 'long' or 'short'
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.shares = shares
        self.entry_time = entry_time
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None
        self.pnl = 0.0

        self.use_sl = use_sl
        self.use_tp = use_tp
        self.use_ts = use_ts

        self.tracked_stop = stop_loss if use_sl else None
        self.trailing_stop = None

    def check_exit(
        self,
        bar: pd.Series,
        use_dsrsi: bool,
        kps_long_exit_threshold: float,
        kps_short_exit_threshold: float
    ) -> bool:
        """
        Mode-aware exit checks:
          - Both modes: SL/TP/Trailing first
          - KPS-only: KPS threshold exits
          - DSRSI mode: KPS threshold OR KPS trend flip exits
        """
        current_price = float(bar['close'])

        # 1) Stops
        final_stop = self.tracked_stop
        if self.use_ts and self.trailing_stop is not None:
            if self.side == 'long':
                final_stop = max(final_stop, self.trailing_stop) if final_stop is not None else self.trailing_stop
            else:
                final_stop = min(final_stop, self.trailing_stop) if final_stop is not None else self.trailing_stop

        if self.use_sl and final_stop is not None:
            if (self.side == 'long' and current_price <= final_stop) or (self.side == 'short' and current_price >= final_stop):
                self.exit_price = final_stop
                self.exit_reason = 'trailing_stop' if (self.use_ts and self.trailing_stop is not None) else 'stop_loss'
                return True

        # 2) Take profit
        if self.use_tp and self.take_profit is not None:
            if (self.side == 'long' and current_price >= self.take_profit) or (self.side == 'short' and current_price <= self.take_profit):
                self.exit_price = self.take_profit
                self.exit_reason = 'take_profit'
                return True

        # 3) KPS-based exits
        pst = bar.get('pst_buffer', np.nan)
        pss = bar.get('pss_buffer', np.nan)
        tr = int(bar.get('trend', 0))

        if self.side == 'long':
            # Threshold exit on pstBuffer (>= 85 by default)
            if not np.isnan(pst) and pst >= kps_long_exit_threshold:
                self.exit_price = current_price
                self.exit_reason = 'kps_threshold'
                return True
            # In DSRSI mode: also exit if trend flips bearish
            if use_dsrsi and tr == -1:
                self.exit_price = current_price
                self.exit_reason = 'kps_trend_flip'
                return True
        else:
            if not np.isnan(pst) and pst <= kps_short_exit_threshold:
                self.exit_price = current_price
                self.exit_reason = 'kps_threshold'
                return True
            if use_dsrsi and tr == 1:
                self.exit_price = current_price
                self.exit_reason = 'kps_trend_flip'
                return True

        return False

    def update_trailing_stop(self, bar: pd.Series, atr: float, swing_high: float, swing_low: float,
                             ts_method: str, ts_source: str,
                             ts_atr_multiplier: float, ts_percent: float):
        if not self.use_ts:
            return
        current_price = bar['close'] if ts_source == 'Close' else bar['open']
        if ts_method == 'ATR':
            atr_trail = atr * ts_atr_multiplier
            if ts_source == 'SwingHL':
                next_trail = (swing_low - atr_trail) if self.side == 'long' else (swing_high + atr_trail)
            else:
                next_trail = (current_price - atr_trail) if self.side == 'long' else (current_price + atr_trail)
        else:
            if ts_source == 'SwingHL':
                next_trail = (swing_low * ((100.0 - ts_percent) / 100.0)) if self.side == 'long' else (swing_high * ((100.0 + ts_percent) / 100.0))
            else:
                next_trail = (current_price * ((100.0 - ts_percent) / 100.0)) if self.side == 'long' else (current_price * ((100.0 + ts_percent) / 100.0))

        if self.side == 'long':
            if self.trailing_stop is None or next_trail > self.trailing_stop:
                self.trailing_stop = next_trail
        else:
            if self.trailing_stop is None or next_trail < self.trailing_stop:
                self.trailing_stop = next_trail

    def close(self, exit_price: float, exit_time, exit_reason: str):
        self.exit_price = float(exit_price)
        self.exit_time = exit_time
        self.exit_reason = exit_reason
        if self.side == 'long':
            self.pnl = (self.exit_price - self.entry_price) * self.shares
        else:
            self.pnl = (self.entry_price - self.exit_price) * self.shares

# ────────────────────────────────────────────────────────────────────────────────
# Market Order Backtest Engine
# ────────────────────────────────────────────────────────────────────────────────
class BacktestEngine:
    def __init__(self, symbol: str, initial_capital: float, risk_percent: float,
                 position_multiplier: float, commission: float = None,
                 use_dsrsi: bool = False,
                 close_at_eod: bool = True, market_hours_only: bool = True,
                 use_sl: bool = USE_SL, use_tp: bool = USE_TP, use_ts: bool = USE_TS,
                 ts_swing_lookback: int = TS_SWING_LOOKBACK, ts_method: str = TS_METHOD,
                 ts_source: str = TS_SOURCE, ts_atr_multiplier: float = TS_ATR_MULTIPLIER,
                 ts_percent: float = TS_PERCENT, exit_at_opposite_signal: bool = EXIT_AT_OPPOSITE_SIGNAL,
                 kps_long_exit_threshold: float = KPS_LONG_EXIT_THRESHOLD,
                 kps_short_exit_threshold: float = KPS_SHORT_EXIT_THRESHOLD):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_percent = risk_percent
        self.position_multiplier = position_multiplier
        self.commission = commission
        self.use_dsrsi = use_dsrsi
        self.close_at_eod = close_at_eod
        self.market_hours_only = market_hours_only

        self.use_sl = use_sl
        self.use_tp = use_tp
        self.use_ts = use_ts
        self.ts_swing_lookback = ts_swing_lookback
        self.ts_method = ts_method
        self.ts_source = ts_source
        self.ts_atr_multiplier = ts_atr_multiplier
        self.ts_percent = ts_percent
        self.exit_at_opposite_signal = exit_at_opposite_signal
        self.kps_long_exit_threshold = kps_long_exit_threshold
        self.kps_short_exit_threshold = kps_short_exit_threshold

        self.positions: List[Position] = []
        self.trades: List[Position] = []
        self.equity_curve = []
        self.logger = logging.getLogger(__name__)

    def run(self, signals: pd.DataFrame) -> pd.DataFrame:
        self.logger.info("Starting JFKPS/DSRSI backtest (market orders, next open)")

        for i in range(len(signals) - 1):
            ts = signals.index[i]
            bar = signals.iloc[i]
            nxt = signals.iloc[i + 1]

            # need valid ATR & KPS buffers
            if pd.isna(bar['atr_shifted']) or pd.isna(bar['pst_buffer']) or pd.isna(bar['pss_buffer']):
                continue

            # Swing values for trailing
            if self.use_ts and i >= self.ts_swing_lookback:
                recent = signals.iloc[i - self.ts_swing_lookback + 1: i + 1]
                swing_high = recent['high'].max()
                swing_low = recent['low'].min()
            else:
                swing_high = bar['high']
                swing_low = bar['low']

            # 1) Handle exits first (prevents same-bar trail update then stop-out)
            for pos in self.positions[:]:
                if pos.check_exit(bar, self.use_dsrsi, self.kps_long_exit_threshold, self.kps_short_exit_threshold):
                    pos.close(pos.exit_price, ts, pos.exit_reason)
                    self.capital += pos.pnl - (self.commission or 0.0)
                    self.trades.append(pos)
                    self.positions.remove(pos)

            # 2) Update trailing stops
            for pos in self.positions:
                pos.update_trailing_stop(
                    bar=bar, atr=bar['atr_shifted'], swing_high=swing_high, swing_low=swing_low,
                    ts_method=self.ts_method, ts_source=self.ts_source,
                    ts_atr_multiplier=self.ts_atr_multiplier, ts_percent=self.ts_percent
                )

            # 3) Optional opposite-signal exit (mode-aware)
            if self.exit_at_opposite_signal and len(self.positions) > 0:
                pos = self.positions[0]
                if self.use_dsrsi:
                    opp = (pos.side == 'long' and bar['dsrsi_short_signal']) or (pos.side == 'short' and bar['dsrsi_long_signal'])
                else:
                    opp = (pos.side == 'long' and bar['kps_go_short']) or (pos.side == 'short' and bar['kps_go_long'])
                if opp:
                    pos.close(bar['close'], ts, 'opposite_signal')
                    self.capital += pos.pnl - (self.commission or 0.0)
                    self.trades.append(pos)
                    self.positions.remove(pos)

            # 4) End-of-day forced close
            if self.close_at_eod and len(self.positions) > 0:
                ts_et = ts.tz_convert(ET) if ts.tz else ts.tz_localize('UTC', ambiguous=False).tz_convert(ET)
                if ts_et.time() >= EOD_CLOSE:
                    for pos in self.positions[:]:
                        pos.close(bar['close'], ts, 'eod_close')
                        self.capital += pos.pnl - (self.commission or 0.0)
                        self.trades.append(pos)
                        self.positions.remove(pos)
                    # No new entries at EOD bar
                    open_pnl = sum(self._open_pnl(p, bar) for p in self.positions)
                    self.equity_curve.append({'timestamp': ts, 'equity': self.capital + open_pnl, 'capital': self.capital, 'open_positions': len(self.positions)})
                    continue

            # 5) Entries (if flat)
            if len(self.positions) == 0:
                # skip at/after EOD if configured
                if self.close_at_eod:
                    ts_et = ts.tz_convert(ET) if ts.tz else ts.tz_localize('UTC', ambiguous=False).tz_convert(ET)
                    if ts_et.time() >= EOD_CLOSE:
                        open_pnl = 0.0
                        self.equity_curve.append({'timestamp': ts, 'equity': self.capital, 'capital': self.capital, 'open_positions': 0})
                        continue

                # enforce RTH on entries if configured
                if self.market_hours_only:
                    ts_et = ts.tz_convert(ET) if ts.tz else ts.tz_localize('UTC', ambiguous=False).tz_convert(ET)
                    if ts_et.time() < MARKET_OPEN or ts_et.time() > MARKET_CLOSE:
                        open_pnl = 0.0
                        self.equity_curve.append({'timestamp': ts, 'equity': self.capital, 'capital': self.capital, 'open_positions': 0})
                        continue

                if bar['entry_long']:
                    entry_price = nxt['open']  # next open (no look-ahead)
                    shares = calculate_position_size(self.capital, self.risk_percent, entry_price, bar['long_stop'], self.position_multiplier)
                    if shares > 0:
                        pos = Position(self.symbol, 'long', entry_price, bar['long_stop'], bar['long_target'], shares, ts,
                                       use_sl=self.use_sl, use_tp=self.use_tp, use_ts=self.use_ts)
                        self.positions.append(pos)
                        if self.commission:
                            self.capital -= self.commission
                        self.logger.info(f"Enter LONG @ {entry_price:.2f} (SL {bar['long_stop']:.2f} TP {bar['long_target']:.2f})")

                elif bar['entry_short']:
                    entry_price = nxt['open']
                    shares = calculate_position_size(self.capital, self.risk_percent, entry_price, bar['short_stop'], self.position_multiplier)
                    if shares > 0:
                        pos = Position(self.symbol, 'short', entry_price, bar['short_stop'], bar['short_target'], shares, ts,
                                       use_sl=self.use_sl, use_tp=self.use_tp, use_ts=self.use_ts)
                        self.positions.append(pos)
                        if self.commission:
                            self.capital -= self.commission
                        self.logger.info(f"Enter SHORT @ {entry_price:.2f} (SL {bar['short_stop']:.2f} TP {bar['short_target']:.2f})")

            # 6) Equity
            open_pnl = sum(self._open_pnl(p, bar) for p in self.positions)
            self.equity_curve.append({'timestamp': ts, 'equity': self.capital + open_pnl, 'capital': self.capital, 'open_positions': len(self.positions)})

        # Final bar cleanup
        if len(signals) > 0:
            last_bar = signals.iloc[-1]
            last_ts = signals.index[-1]
            for pos in self.positions[:]:
                # one more trail update & exit check
                if self.use_ts:
                    # use recent window for swings
                    lookback = self.ts_swing_lookback
                    sub = signals.iloc[-lookback:] if lookback < len(signals) else signals
                    swing_high = sub['high'].max()
                    swing_low = sub['low'].min()
                    pos.update_trailing_stop(last_bar, last_bar['atr_shifted'], swing_high, swing_low,
                                             self.ts_method, self.ts_source, self.ts_atr_multiplier, self.ts_percent)
                if pos.check_exit(last_bar, self.use_dsrsi, self.kps_long_exit_threshold, self.kps_short_exit_threshold):
                    pos.close(pos.exit_price, last_ts, pos.exit_reason)
                    self.capital += pos.pnl - (self.commission or 0.0)
                    self.trades.append(pos)
                    self.positions.remove(pos)

            # record final equity
            open_pnl = sum(self._open_pnl(p, last_bar) for p in self.positions)
            self.equity_curve.append({'timestamp': last_ts, 'equity': self.capital + open_pnl, 'capital': self.capital, 'open_positions': len(self.positions)})

        # End-of-backtest force-close remaining positions at last close
        if len(self.positions) > 0:
            last_bar = signals.iloc[-1]
            for pos in self.positions:
                pos.close(last_bar['close'], signals.index[-1], 'end_of_backtest')
                self.capital += pos.pnl - (self.commission or 0.0)
                self.trades.append(pos)
            self.positions.clear()

        self.logger.info(f"Backtest complete. Trades: {len(self.trades)}")
        return self._results_df()

    def _open_pnl(self, position: Position, bar: pd.Series) -> float:
        px = float(bar['close'])
        return (px - position.entry_price) * position.shares if position.side == 'long' else (position.entry_price - px) * position.shares

    def _results_df(self) -> pd.DataFrame:
        rows = []
        for t in self.trades:
            rows.append({
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'side': t.side,
                'shares': t.shares,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'stop_loss': t.stop_loss,
                'take_profit': t.take_profit,
                'pnl': t.pnl,
                'exit_reason': t.exit_reason
            })
        return pd.DataFrame(rows)

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_arguments()
    timeframe = extract_timeframe_from_filename(args.primary_data)
    logger = setup_logger(args.symbol, timeframe)
    logger.info("Starting JFKPS/DSRSI Strategy Backtest (Market Orders)")
    logger.info(f"Config: {vars(args)}")

    # Load data
    df = load_data(args.primary_data, args.start_date, args.end_date)

    # Resample if requested
    if getattr(args, 'resample_primary', None):
        df = resample_data(df, args.resample_primary)
        timeframe = args.resample_primary

    # 1) Compute DSRSI (always compute; we may or may not use for entries)
    df = calculate_dsrsi_indicators(
        df,
        source=getattr(args, 'source', SOURCE),
        length=getattr(args, 'dsrsi_length', DSRSI_LENGTH),
        smoothing_period=getattr(args, 'smoothing_period', SMOOTHING_PERIOD),
        volume_weighted=getattr(args, 'volume_weighted', VOLUME_WEIGHTED)
    )

    # 2) Compute KPS
    df = calculate_kps_indicators(
        df,
        pst_length=getattr(args, 'pst_length', PST_LENGTH),
        pst_x=getattr(args, 'pst_x', PST_X),
        pst_smooth=getattr(args, 'pst_smooth', PST_SMOOTH),
        smooth_period=getattr(args, 'kps_smooth_period', SMOOTH_PERIOD),
        jphase=getattr(args, 'jphase', JPHASE)
    )

    # 3) Build signals (mode-aware)
    use_dsrsi = bool(getattr(args, 'use_dsrsi', False))
    signals = generate_signals(
        df,
        use_dsrsi=use_dsrsi,
        market_hours_only=args.market_hours_only,
        atr_length=getattr(args, 'atr_length', ATR_LENGTH),
        sl_atr_ratio=getattr(args, 'sl_atr_ratio', SL_ATR_RATIO),
        tp_sl_ratio=getattr(args, 'tp_sl_ratio', TP_SL_RATIO)
    )

    # 4) Run engine
    engine = BacktestEngine(
        args.symbol,
        args.initial_capital,
        args.risk_percent,
        args.position_multiplier,
        args.commission,
        use_dsrsi=use_dsrsi,
        close_at_eod=args.close_at_eod,
        market_hours_only=args.market_hours_only,
        use_sl=getattr(args, 'use_sl', USE_SL),
        use_tp=getattr(args, 'use_tp', USE_TP),
        use_ts=getattr(args, 'use_ts', USE_TS),
        ts_swing_lookback=getattr(args, 'ts_swing_lookback', TS_SWING_LOOKBACK),
        ts_method=getattr(args, 'ts_method', TS_METHOD),
        ts_source=getattr(args, 'ts_source', TS_SOURCE),
        ts_atr_multiplier=getattr(args, 'ts_atr_multiplier', TS_ATR_MULTIPLIER),
        ts_percent=getattr(args, 'ts_percent', TS_PERCENT),
        exit_at_opposite_signal=getattr(args, 'exit_at_opposite_signal', EXIT_AT_OPPOSITE_SIGNAL),
        kps_long_exit_threshold=getattr(args, 'kps_long_exit_threshold', KPS_LONG_EXIT_THRESHOLD),
        kps_short_exit_threshold=getattr(args, 'kps_short_exit_threshold', KPS_SHORT_EXIT_THRESHOLD)
    )
    trades_df = engine.run(signals)

    # Outputs
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = f"{args.symbol}_{timeframe}_jfkps_dsrsi_market"

    # Trades
    if len(trades_df) > 0:
        f = os.path.join(args.output_dir, f"{prefix}_trades.csv")
        trades_df.to_csv(f, index=False)
        logger.info(f"Trades -> {f}")

    # Equity curve
    equity_df = pd.DataFrame(engine.equity_curve)
    f_eq = os.path.join(args.output_dir, f"{prefix}_equity.csv")
    equity_df.to_csv(f_eq, index=False)

    # Metrics
    metrics = calculate_performance_metrics(trades_df, engine.equity_curve, args.initial_capital)
    metrics.update({
        'symbol': args.symbol,
        'timeframe': timeframe,
        'strategy': 'JFKPS_DSRSI',
        'order_type': 'market',
        'use_dsrsi': use_dsrsi,
        'start_date': str(df.index[0]),
        'end_date': str(df.index[-1])
    })
    f_js = os.path.join(args.output_dir, f"{prefix}_performance.json")
    with open(f_js, 'w') as fh:
        json.dump(metrics, fh, indent=2, default=str)

    # Pretty print
    print("\n" + "="*64)
    print("JFKPS / DSRSI STRATEGY RESULTS (MARKET ORDERS)")
    print("="*64)
    print(f"Symbol: {args.symbol}  Timeframe: {timeframe}")
    print(f"Period: {df.index[0]}  →  {df.index[-1]}")
    print(f"Initial Capital: ${args.initial_capital:,.2f}   Final: ${metrics['final_capital']:,.2f}")
    print(f"Trades: {metrics['total_trades']}  Win%: {metrics['win_rate']:.1f}%  PF: {metrics['profit_factor']:.2f}")
    print(f"Sharpe: {metrics['sharpe_ratio']:.2f}   MaxDD: {metrics['max_drawdown']:.2f}%")
    print(f"Total Return: {metrics['total_return']:.2f}%   P&L: ${metrics['total_pnl']:,.2f}")
    print("="*64)
    print(f"Saved: {args.output_dir}/")

if __name__ == '__main__':
    main()
