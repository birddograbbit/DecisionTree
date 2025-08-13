#!/usr/bin/env python3
"""
tema_backtest_v3.py - Realistic TEMA Trend Following Backtest

A realistic implementation of the TEMA trend following strategy that fixes
look-ahead bias and implements proper limit order execution logic.
Based on Jesse framework strategy logic with multi-bar order persistence.

KEY IMPROVEMENTS IN V3:
- Fixed look-ahead bias: Signals based on PREVIOUS bar data
- Realistic limit order execution: Orders only fill if price touches within bar's range
- Multi-bar order persistence: Orders remain active for bars_to_enter parameter
- No same-bar entry/exit: Prevents unrealistic instant profits
- Proper order cancellation after timeout

FEATURES:
- Works with any IBKR data file and timeframe
- Optional automatic resampling to create multiple timeframes from single data file
- Single timeframe mode: Uses only primary timeframe indicators
- Dual timeframe mode: Confirms trends across two timeframes for better filtering
- Validates IBKR data format and handles index-specific quirks (volume=0)
- Comprehensive performance metrics and trade logging

USAGE EXAMPLES:

1. Single Timeframe with 5-min data (no resampling):
   python tema_backtest_v3.py --primary_data SPX_5_mins.csv --symbol SPX

2. Single Timeframe with resampling (5-min to 1H):
   python tema_backtest_v3.py --primary_data SPX_5_mins.csv --symbol SPX --resample_primary 1h

3. Dual Timeframe with auto-resampling (5-min to 1H + 4H):
   python tema_backtest_v3.py --primary_data SPX_5_mins.csv --symbol SPX \\
          --resample_primary 1h --resample_secondary 4h

4. Dual Timeframe with pre-resampled files:
   python tema_backtest_v3.py --primary_data SPX_1_hour.csv --secondary_data SPX_4_hour.csv \\
          --symbol SPX

5. With date filtering and market hours only:
   python tema_backtest_v3.py --primary_data SPX_5_mins.csv --symbol SPX \\
          --resample_primary 1h --resample_secondary 4h \\
          --start_date 2023-01-01 --end_date 2023-12-31 --market_hours_only

6. Custom risk parameters:
   python tema_backtest_v3.py --primary_data SPX_5_mins.csv --symbol SPX \\
          --resample_primary 1h --risk_percent 2 --position_multiplier 2

7. Trading any security with 1-unit strategy testing:
   python tema_backtest_v3.py --primary_data SPX_5_mins.csv --symbol SPX \\
          --resample_primary 1h

STRATEGY LOGIC:
- Primary timeframe: TEMA(10) > TEMA(80) for uptrend
- Secondary timeframe: TEMA(20) > TEMA(70) for uptrend (if dual mode)
- Entry filters: ADX > 40 and CMO > 40 (long) or < -40 (short)
- Risk management: 1-unit trading for fair comparison
- Entry: Limit order at 1 ATR from previous close
- Order persistence: Cancelled after bars_to_enter bars (default 6)
- Stop loss: 3 ATR from entry
- Take profit: 3 ATR from entry

VALID TIMEFRAME FORMATS:
- Minutes: 1T, 5T, 15T, 30T
- Hours: 1h, 2h, 4h (lowercase 'h' to avoid pandas warnings)
- Days: 1D, 2D
- Weeks: 1W
"""

import argparse
import logging
import os
import json
import pandas as pd
import numpy as np
import talib
import pytz
from datetime import datetime, time
from typing import Dict, List, Tuple, Optional

# ────────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────────
ET = pytz.timezone('US/Eastern')
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)
EOD_CLOSE = time(15, 55)  # 3:55 PM ET - last 5-min bar before close

# Strategy parameters
TEMA_PRIMARY_FAST = 10
TEMA_PRIMARY_SLOW = 80
TEMA_SECONDARY_FAST = 20
TEMA_SECONDARY_SLOW = 70
ADX_THRESHOLD = 40
CMO_LONG_THRESHOLD = 40
CMO_SHORT_THRESHOLD = -40

# Risk parameters
ATR_ENTRY_OFFSET = 1
ATR_STOP_LOSS = 3 #default 4
ATR_TAKE_PROFIT = 3 #default 3

# ────────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ────────────────────────────────────────────────────────────────────────────────
def setup_logger(symbol: str, timeframe: str) -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs('logs', exist_ok=True)
    log_file = f'logs/tema_v3_{symbol}_{timeframe}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Clear existing handlers to prevent duplication
    root_logger.handlers.clear()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
        force=True  # Force reconfiguration even if logging has already been configured
    )
    return logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────────────
# CLI Arguments
# ────────────────────────────────────────────────────────────────────────────────
def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Flexible TEMA Trend Following Backtest')
    
    # Data inputs
    parser.add_argument('--primary_data', 
                       required=True,
                       help='Primary timeframe data file (required)')
    
    parser.add_argument('--secondary_data', 
                       help='Secondary timeframe data file (optional, for dual timeframe mode)')
    
    parser.add_argument('--symbol', 
                       required=True,
                       help='Symbol name (e.g., SPX, SPY, QQQ)')
    
    # Resampling options
    parser.add_argument('--resample_primary', 
                       help='Resample primary data to specified timeframe (e.g., 1h, 4h, 1D)')
    
    parser.add_argument('--resample_secondary', 
                       help='Resample primary data to create secondary timeframe (e.g., 4h, 1D)')
    
    # Date range
    parser.add_argument('--start_date', 
                       help='Backtest start date (YYYY-MM-DD)')
    
    parser.add_argument('--end_date', 
                       help='Backtest end date (YYYY-MM-DD)')
    
    # Risk parameters
    parser.add_argument('--initial_capital', 
                       type=float, 
                       default=100000,
                       help='Initial capital (default: 100000)')
    
    parser.add_argument('--risk_percent', 
                       type=float, 
                       default=3,
                       help='Risk per trade as percentage (default: 3)')
    
    parser.add_argument('--position_multiplier', 
                       type=float, 
                       default=3,
                       help='Position size multiplier (default: 3)')
    
    parser.add_argument('--commission', 
                       type=float, 
                       default=None,
                       help='Commission per trade in dollars (default: None for no commission)')
    
    # Output
    parser.add_argument('--output_dir', 
                       default='results',
                       help='Output directory for results (default: results)')
    
    parser.add_argument('--market_hours_only', 
                       action='store_true',
                       help='Filter for regular market hours only')
    parser.add_argument('--close_at_eod',
                       action='store_true',
                       help='Close all positions at 3:55 PM ET (no overnight positions)')
    
    # Strategy parameters
    parser.add_argument('--tema_primary_fast',
                       type=int,
                       default=10,
                       help='Fast TEMA period for primary timeframe (default: 10)')
    
    parser.add_argument('--tema_primary_slow',
                       type=int,
                       default=80,
                       help='Slow TEMA period for primary timeframe (default: 80)')
    
    parser.add_argument('--tema_secondary_fast',
                       type=int,
                       default=20,
                       help='Fast TEMA period for secondary timeframe (default: 20)')
    
    parser.add_argument('--tema_secondary_slow',
                       type=int,
                       default=70,
                       help='Slow TEMA period for secondary timeframe (default: 70)')
    
    parser.add_argument('--adx_threshold',
                       type=int,
                       default=40,
                       help='ADX threshold for trend strength (default: 40)')
    
    parser.add_argument('--cmo_long_threshold',
                       type=int,
                       default=40,
                       help='CMO threshold for long entries (default: 40)')
    
    parser.add_argument('--cmo_short_threshold',
                       type=int,
                       default=-40,
                       help='CMO threshold for short entries (default: -40)')
    
    # ATR-based risk parameters
    parser.add_argument('--atr_entry_offset',
                       type=float,
                       default=1.0,
                       help='ATR multiplier for entry offset (default: 1.0)')
    
    parser.add_argument('--atr_stop_loss',
                       type=float,
                       default=3.0,
                       help='ATR multiplier for stop loss (default: 3.0)')
    
    parser.add_argument('--atr_take_profit',
                       type=float,
                       default=3.0,
                       help='ATR multiplier for take profit (default: 3.0)')
    
    parser.add_argument('--bars_to_enter',
                       type=int,
                       default=6,
                       help='Number of bars to wait for limit order fill (default: 6)')
    
    return parser.parse_args()

# ────────────────────────────────────────────────────────────────────────────────
# Data Loading and Processing
# ────────────────────────────────────────────────────────────────────────────────
def load_data(file_path: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load and prepare historical data"""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load CSV with UTC timezone
    df = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    
    # Ensure UTC timezone
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    
    # Filter date range if specified
    if start_date:
        start_dt = pd.to_datetime(start_date).tz_localize('UTC')
        df = df[df.index >= start_dt]
    if end_date:
        end_dt = pd.to_datetime(end_date).tz_localize('UTC')
        df = df[df.index <= end_dt]
    
    logger.info(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    # Validate data
    validate_data(df, file_path)
    
    return df

def filter_market_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Filter data to only include regular market hours"""
    df_et = df.copy()
    df_et.index = df_et.index.tz_convert(ET)
    
    # Filter for market hours
    mask = (df_et.index.time >= MARKET_OPEN) & (df_et.index.time <= MARKET_CLOSE)
    df_filtered = df_et[mask]
    
    # Convert back to UTC
    df_filtered.index = df_filtered.index.tz_convert('UTC')
    
    return df_filtered

def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample data to specified timeframe"""
    logger = logging.getLogger(__name__)
    logger.info(f"Resampling data to {timeframe}")
    
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # Handle additional IBKR columns if present
    if 'average' in df.columns:
        ohlc_dict['average'] = 'mean'
    if 'barCount' in df.columns:
        ohlc_dict['barCount'] = 'sum'
    
    resampled = df.resample(timeframe).agg(ohlc_dict)
    resampled = resampled.dropna()
    
    logger.info(f"Resampled from {len(df)} to {len(resampled)} bars")
    return resampled

def validate_data(df: pd.DataFrame, filename: str) -> None:
    """Validate that data has required columns"""
    logger = logging.getLogger(__name__)
    required_columns = ['open', 'high', 'low', 'close']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Data file {filename} missing required columns: {missing_columns}")
    
    # Check for valid data
    if len(df) == 0:
        raise ValueError(f"Data file {filename} is empty")
    
    # Log data info
    logger.info(f"Data validation passed for {filename}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")
    logger.info(f"Total bars: {len(df)}")
    
    # Check for volume = 0 (common for indices)
    if 'volume' in df.columns and (df['volume'] == 0).all():
        logger.info("Note: All volume values are 0 (typical for index data)")

def extract_timeframe_from_filename(filename: str) -> str:
    """Extract timeframe from filename (e.g., 'SPX_5_mins.csv' -> '5_mins')"""
    base = os.path.basename(filename)
    # Remove common prefixes and extensions
    cleaned = base.replace('historical_data_', '').replace('INDEX_', '').replace('.csv', '')
    parts = cleaned.split('_')
    
    # Look for common timeframe patterns
    timeframe_parts = []
    for i, part in enumerate(parts):
        if part.isdigit() and i + 1 < len(parts):
            if parts[i + 1] in ['min', 'mins', 'hour', 'hours', 'day', 'days', 'week', 'weeks']:
                timeframe_parts = [part, parts[i + 1]]
                break
    
    return '_'.join(timeframe_parts) if timeframe_parts else 'unknown'

# ────────────────────────────────────────────────────────────────────────────────
# Data Alignment for Dual Timeframe Mode
# ────────────────────────────────────────────────────────────────────────────────
def align_timeframes(primary_df: pd.DataFrame, secondary_df: pd.DataFrame) -> pd.DataFrame:
    """Align secondary timeframe data to primary timeframe"""
    logger = logging.getLogger(__name__)
    logger.info("Aligning secondary timeframe to primary timeframe")
    
    # Create a combined dataframe with primary data
    aligned = primary_df.copy()
    
    # For each primary timestamp, find the most recent secondary data
    secondary_data = {}
    for col in ['close', 'tema_fast', 'tema_slow', 'long_trend']:
        if col in secondary_df.columns:
            # Use merge_asof to get the most recent secondary value for each primary timestamp
            temp_df = pd.DataFrame(index=primary_df.index)
            temp_df['primary_time'] = temp_df.index
            
            secondary_temp = secondary_df[[col]].copy()
            secondary_temp['secondary_time'] = secondary_temp.index
            
            merged = pd.merge_asof(
                temp_df.reset_index(drop=True).sort_values('primary_time'),
                secondary_temp.reset_index(drop=True).sort_values('secondary_time'),
                left_on='primary_time',
                right_on='secondary_time',
                direction='backward'
            )
            
            secondary_data[f'secondary_{col}'] = merged.set_index(temp_df.index)[col].values
    
    # Add secondary data to aligned dataframe
    for col, values in secondary_data.items():
        aligned[col] = values
    
    logger.info(f"Aligned {len(secondary_df)} secondary bars to {len(primary_df)} primary bars")
    return aligned

# ────────────────────────────────────────────────────────────────────────────────
# Technical Indicators
# ────────────────────────────────────────────────────────────────────────────────
def calculate_indicators(df: pd.DataFrame, timeframe_type: str = 'primary',
                        tema_primary_fast: int = TEMA_PRIMARY_FAST,
                        tema_primary_slow: int = TEMA_PRIMARY_SLOW,
                        tema_secondary_fast: int = TEMA_SECONDARY_FAST,
                        tema_secondary_slow: int = TEMA_SECONDARY_SLOW) -> pd.DataFrame:
    """Calculate technical indicators for a given timeframe"""
    logger = logging.getLogger(__name__)
    logger.info(f"Calculating {timeframe_type} timeframe indicators")
    
    df = df.copy()
    
    # Select TEMA periods based on timeframe type
    if timeframe_type == 'primary':
        fast_period = tema_primary_fast
        slow_period = tema_primary_slow
    else:  # secondary
        fast_period = tema_secondary_fast
        slow_period = tema_secondary_slow
    
    # Calculate TEMA indicators
    df['tema_fast'] = talib.TEMA(df['close'], timeperiod=fast_period)
    df['tema_slow'] = talib.TEMA(df['close'], timeperiod=slow_period)
    
    # Trend direction
    df['trend'] = np.where(df['tema_fast'] > df['tema_slow'], 1, -1)
    
    # For primary timeframe, calculate additional indicators
    if timeframe_type == 'primary':
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
        df['cmo'] = talib.CMO(df['close'])
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
    
    # Rename trend column for secondary timeframe
    if timeframe_type == 'secondary':
        df['long_trend'] = df['trend']
        df.drop('trend', axis=1, inplace=True)
    
    return df

# ────────────────────────────────────────────────────────────────────────────────
# Signal Generation
# ────────────────────────────────────────────────────────────────────────────────
def generate_signals_single_timeframe(df: pd.DataFrame, market_hours_only: bool = False,
                                    adx_threshold: int = ADX_THRESHOLD,
                                    cmo_long_threshold: int = CMO_LONG_THRESHOLD,
                                    cmo_short_threshold: int = CMO_SHORT_THRESHOLD,
                                    atr_entry_offset: float = ATR_ENTRY_OFFSET,
                                    atr_stop_loss: float = ATR_STOP_LOSS,
                                    atr_take_profit: float = ATR_TAKE_PROFIT) -> pd.DataFrame:
    """Generate signals for single timeframe mode"""
    logger = logging.getLogger(__name__)
    logger.info("Generating signals (single timeframe mode)")
    
    signals = df.copy()
    
    # Long signals (using PREVIOUS bar data to avoid look-ahead)
    signals['long_signal'] = (
        (signals['trend'].shift(1) == 1) &
        (signals['adx'].shift(1) > adx_threshold) &
        (signals['cmo'].shift(1) > cmo_long_threshold)
    )
    
    # Short signals
    signals['short_signal'] = (
        (signals['trend'].shift(1) == -1) &
        (signals['adx'].shift(1) > adx_threshold) &
        (signals['cmo'].shift(1) < cmo_short_threshold)
    )
    
    # Entry prices (limit orders) - based on PREVIOUS bar
    signals['long_entry_price'] = signals['close'].shift(1) - (signals['atr'].shift(1) * atr_entry_offset)
    signals['short_entry_price'] = signals['close'].shift(1) + (signals['atr'].shift(1) * atr_entry_offset)
    
    # Stop loss and take profit levels - based on entry price calculations
    signals['long_stop'] = signals['long_entry_price'] - (signals['atr'].shift(1) * atr_stop_loss)
    signals['long_target'] = signals['long_entry_price'] + (signals['atr'].shift(1) * atr_take_profit)
    
    signals['short_stop'] = signals['short_entry_price'] + (signals['atr'].shift(1) * atr_stop_loss)
    signals['short_target'] = signals['short_entry_price'] - (signals['atr'].shift(1) * atr_take_profit)
    
    # Filter signals to market hours only if requested
    if market_hours_only:
        # Convert to ET for market hours check
        signals_et = signals.copy()
        signals_et.index = signals_et.index.tz_convert(ET)
        market_hours_mask = (signals_et.index.time >= MARKET_OPEN) & (signals_et.index.time <= MARKET_CLOSE)
        
        # Set signals to False outside market hours
        signals.loc[~market_hours_mask, 'long_signal'] = False
        signals.loc[~market_hours_mask, 'short_signal'] = False
        
        logger.info(f"Filtered signals to market hours only")
    
    return signals

def generate_signals_dual_timeframe(df: pd.DataFrame, market_hours_only: bool = False,
                                  adx_threshold: int = ADX_THRESHOLD,
                                  cmo_long_threshold: int = CMO_LONG_THRESHOLD,
                                  cmo_short_threshold: int = CMO_SHORT_THRESHOLD,
                                  atr_entry_offset: float = ATR_ENTRY_OFFSET,
                                  atr_stop_loss: float = ATR_STOP_LOSS,
                                  atr_take_profit: float = ATR_TAKE_PROFIT) -> pd.DataFrame:
    """Generate signals for dual timeframe mode"""
    logger = logging.getLogger(__name__)
    logger.info("Generating signals (dual timeframe mode)")
    
    signals = df.copy()
    
    # Rename primary trend for clarity
    signals['short_trend'] = signals['trend']
    
    # Get long trend from secondary data
    signals['long_trend'] = signals['secondary_long_trend']
    
    # Long signals (both timeframes must agree) - using PREVIOUS bar data
    signals['long_signal'] = (
        (signals['short_trend'].shift(1) == 1) &
        (signals['long_trend'].shift(1) == 1) &
        (signals['adx'].shift(1) > adx_threshold) &
        (signals['cmo'].shift(1) > cmo_long_threshold)
    )
    
    # Short signals - using PREVIOUS bar data
    signals['short_signal'] = (
        (signals['short_trend'].shift(1) == -1) &
        (signals['long_trend'].shift(1) == -1) &
        (signals['adx'].shift(1) > adx_threshold) &
        (signals['cmo'].shift(1) < cmo_short_threshold)
    )
    
    # Entry prices (limit orders) - based on PREVIOUS bar
    signals['long_entry_price'] = signals['close'].shift(1) - (signals['atr'].shift(1) * atr_entry_offset)
    signals['short_entry_price'] = signals['close'].shift(1) + (signals['atr'].shift(1) * atr_entry_offset)
    
    # Stop loss and take profit levels - based on entry price calculations
    signals['long_stop'] = signals['long_entry_price'] - (signals['atr'].shift(1) * atr_stop_loss)
    signals['long_target'] = signals['long_entry_price'] + (signals['atr'].shift(1) * atr_take_profit)
    
    signals['short_stop'] = signals['short_entry_price'] + (signals['atr'].shift(1) * atr_stop_loss)
    signals['short_target'] = signals['short_entry_price'] - (signals['atr'].shift(1) * atr_take_profit)
    
    # Filter signals to market hours only if requested
    if market_hours_only:
        # Convert to ET for market hours check
        signals_et = signals.copy()
        signals_et.index = signals_et.index.tz_convert(ET)
        market_hours_mask = (signals_et.index.time >= MARKET_OPEN) & (signals_et.index.time <= MARKET_CLOSE)
        
        # Set signals to False outside market hours
        signals.loc[~market_hours_mask, 'long_signal'] = False
        signals.loc[~market_hours_mask, 'short_signal'] = False
        
        logger.info(f"Filtered signals to market hours only")
    
    return signals

# ────────────────────────────────────────────────────────────────────────────────
# Position Management (reuse from v1)
# ────────────────────────────────────────────────────────────────────────────────
class Position:
    """Class to track individual positions"""
    def __init__(self, symbol: str, side: str, entry_price: float, 
                 stop_loss: float, take_profit: float, shares: int, 
                 entry_time: pd.Timestamp):
        self.symbol = symbol
        self.side = side  # 'long' or 'short'
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.shares = shares
        self.entry_time = entry_time
        self.exit_price = None
        self.exit_time = None
        self.pnl = 0
        self.status = 'open'  # 'open', 'closed'
        self.exit_reason = None  # 'stop_loss', 'take_profit', 'manual'

    def check_exit(self, bar: pd.Series) -> bool:
        """Check if position should be closed"""
        if self.side == 'long':
            if bar['low'] <= self.stop_loss:
                self.exit_price = self.stop_loss
                self.exit_reason = 'stop_loss'
                return True
            elif bar['high'] >= self.take_profit:
                self.exit_price = self.take_profit
                self.exit_reason = 'take_profit'
                return True
        else:  # short
            if bar['high'] >= self.stop_loss:
                self.exit_price = self.stop_loss
                self.exit_reason = 'stop_loss'
                return True
            elif bar['low'] <= self.take_profit:
                self.exit_price = self.take_profit
                self.exit_reason = 'take_profit'
                return True
        return False

    def close(self, exit_price: float, exit_time: pd.Timestamp, reason: str = 'manual'):
        """Close the position"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.exit_reason = reason
        self.status = 'closed'
        
        if self.side == 'long':
            self.pnl = (self.exit_price - self.entry_price) * self.shares
        else:  # short
            self.pnl = (self.entry_price - self.exit_price) * self.shares

def calculate_position_size(capital: float, risk_percent: float, 
                          entry_price: float, stop_price: float,
                          multiplier: float = 1.0) -> int:
    """Calculate position size for pure strategy testing - always trade 1 unit"""
    # For pure strategy testing, we always trade exactly 1 unit
    # This ensures fair comparison across all securities regardless of price
    return 1

# ────────────────────────────────────────────────────────────────────────────────
# Backtest Engine (reuse from v1 with minor modifications)
# ────────────────────────────────────────────────────────────────────────────────
class BacktestEngine:
    """Main backtest engine with realistic limit order execution"""
    def __init__(self, symbol: str, initial_capital: float, risk_percent: float, 
                 position_multiplier: float, commission: float = None,
                 close_at_eod: bool = False, market_hours_only: bool = False,
                 bars_to_enter: int = 6):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_percent = risk_percent
        self.position_multiplier = position_multiplier
        self.commission = commission  # None means no commission
        self.close_at_eod = close_at_eod
        self.market_hours_only = market_hours_only
        self.bars_to_enter = bars_to_enter  # Number of bars to wait for limit order fill
        self.positions: List[Position] = []
        self.trades: List[Position] = []
        self.equity_curve = []
        self.pending_orders = []  # Track pending limit orders
        self.bar_count_since_entry = 0  # Track bars since last entry
        self.logger = logging.getLogger(__name__)
        
    def run(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Run the backtest with realistic limit order execution"""
        self.logger.info("Starting backtest")
        
        for timestamp, bar in signals.iterrows():
            # Skip if NaN values in critical fields
            if pd.isna(bar['atr']) or pd.isna(bar['adx']) or pd.isna(bar['cmo']):
                continue
            
            # Increment bar count since last entry
            self.bar_count_since_entry += 1
            
            # Check pending orders first
            for order in self.pending_orders[:]:  # Create copy to iterate
                order['bars_waiting'] += 1
                
                # Cancel old orders
                if order['bars_waiting'] > self.bars_to_enter:
                    self.pending_orders.remove(order)
                    self.logger.debug(f"Cancelled {order['side']} order after {self.bars_to_enter} bars")
                    continue
                
                # Check if limit price is hit
                if order['side'] == 'long':
                    if order['entry_price'] >= bar['low'] and order['entry_price'] <= bar['high']:
                        # Enter long position at limit price
                        shares = calculate_position_size(
                            self.capital, self.risk_percent, order['entry_price'], 
                            order['stop_price'], self.position_multiplier
                        )
                        if shares > 0:
                            position = Position(
                                self.symbol, 'long', order['entry_price'], order['stop_price'], 
                                order['target_price'], shares, timestamp
                            )
                            self.positions.append(position)
                            # Deduct commission if set
                            if self.commission is not None:
                                self.capital -= self.commission
                            self.pending_orders.remove(order)
                            self.bar_count_since_entry = 0  # Reset counter
                            self.logger.info(
                                f"Entered long at {order['entry_price']:.2f} "
                                f"(limit hit after {order['bars_waiting']} bars)"
                            )
                elif order['side'] == 'short':
                    if order['entry_price'] >= bar['low'] and order['entry_price'] <= bar['high']:
                        # Enter short position at limit price
                        shares = calculate_position_size(
                            self.capital, self.risk_percent, order['entry_price'], 
                            order['stop_price'], self.position_multiplier
                        )
                        if shares > 0:
                            position = Position(
                                self.symbol, 'short', order['entry_price'], order['stop_price'], 
                                order['target_price'], shares, timestamp
                            )
                            self.positions.append(position)
                            # Deduct commission if set
                            if self.commission is not None:
                                self.capital -= self.commission
                            self.pending_orders.remove(order)
                            self.bar_count_since_entry = 0  # Reset counter
                            self.logger.info(
                                f"Entered short at {order['entry_price']:.2f} "
                                f"(limit hit after {order['bars_waiting']} bars)"
                            )
                
            # Check existing positions for exits
            for position in self.positions[:]:  # Create a copy to iterate
                if position.check_exit(bar):
                    position.close(position.exit_price, timestamp, position.exit_reason)
                    # Apply commission only if set
                    if self.commission is not None:
                        self.capital += position.pnl - self.commission
                    else:
                        self.capital += position.pnl
                    self.trades.append(position)
                    self.positions.remove(position)
                    self.logger.info(
                        f"Closed {position.side} position at {position.exit_price:.2f} "
                        f"({position.exit_reason}), PnL: ${position.pnl:.2f}"
                    )
            
            # Check for end-of-day close if enabled
            if self.close_at_eod and len(self.positions) > 0:
                # Convert timestamp to ET timezone
                timestamp_et = timestamp.tz_convert(ET) if timestamp.tz else timestamp.tz_localize('UTC').tz_convert(ET)
                if timestamp_et.time() >= EOD_CLOSE:
                    # Force close all positions at EOD
                    for position in self.positions[:]:
                        position.close(bar['close'], timestamp, 'eod_close')
                        # Apply commission only if set
                        if self.commission is not None:
                            self.capital += position.pnl - self.commission
                        else:
                            self.capital += position.pnl
                        self.trades.append(position)
                        self.positions.remove(position)
                        self.logger.info(
                            f"EOD close: {position.side} position at {position.exit_price:.2f}, "
                            f"PnL: ${position.pnl:.2f}"
                        )
                    continue  # Skip new signal checks for this bar
            
            # Check for new signals (only if no open positions and no pending orders)
            if len(self.positions) == 0 and len(self.pending_orders) == 0:
                # Skip new entries at or after EOD if close_at_eod is enabled
                if self.close_at_eod:
                    timestamp_et = timestamp.tz_convert(ET) if timestamp.tz else timestamp.tz_localize('UTC').tz_convert(ET)
                    if timestamp_et.time() >= EOD_CLOSE:
                        continue
                
                # Check market hours for position entry if market_hours_only is enabled
                if self.market_hours_only:
                    timestamp_et = timestamp.tz_convert(ET) if timestamp.tz else timestamp.tz_localize('UTC').tz_convert(ET)
                    current_time = timestamp_et.time()
                    # Only allow entry during RTH (9:30 AM to 4:00 PM ET)
                    if current_time < MARKET_OPEN or current_time > MARKET_CLOSE:
                        continue  # Skip position entry outside RTH
                
                if bar['long_signal']:
                    # Add pending long order
                    order = {
                        'side': 'long',
                        'entry_price': bar['long_entry_price'],
                        'stop_price': bar['long_stop'],
                        'target_price': bar['long_target'],
                        'bars_waiting': 0,
                        'signal_time': timestamp
                    }
                    self.pending_orders.append(order)
                    self.logger.info(f"Created long limit order at {order['entry_price']:.2f}")
                elif bar['short_signal']:
                    # Add pending short order
                    order = {
                        'side': 'short',
                        'entry_price': bar['short_entry_price'],
                        'stop_price': bar['short_stop'],
                        'target_price': bar['short_target'],
                        'bars_waiting': 0,
                        'signal_time': timestamp
                    }
                    self.pending_orders.append(order)
                    self.logger.info(f"Created short limit order at {order['entry_price']:.2f}")
            
            # Record equity
            open_pnl = sum(self._calculate_open_pnl(pos, bar) for pos in self.positions)
            total_equity = self.capital + open_pnl
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': total_equity,
                'capital': self.capital,
                'open_positions': len(self.positions)
            })
        
        # Close any remaining open positions at last price
        if len(self.positions) > 0:
            last_bar = signals.iloc[-1]
            for position in self.positions:
                position.close(last_bar['close'], signals.index[-1], 'end_of_backtest')
                # Apply commission only if set
                if self.commission is not None:
                    self.capital += position.pnl - self.commission
                else:
                    self.capital += position.pnl
                self.trades.append(position)
        
        self.logger.info(f"Backtest complete. Total trades: {len(self.trades)}")
        return self._generate_results()
    
    
    def _calculate_open_pnl(self, position: Position, bar: pd.Series) -> float:
        """Calculate open P&L for a position"""
        current_price = bar['close']
        if position.side == 'long':
            return (current_price - position.entry_price) * position.shares
        else:
            return (position.entry_price - current_price) * position.shares
    
    def _generate_results(self) -> pd.DataFrame:
        """Generate backtest results"""
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'side': trade.side,
                'shares': trade.shares,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'pnl': trade.pnl,
                'exit_reason': trade.exit_reason
            })
        
        return pd.DataFrame(trades_data)

# ────────────────────────────────────────────────────────────────────────────────
# Performance Metrics (reuse from v1)
# ────────────────────────────────────────────────────────────────────────────────
def calculate_performance_metrics(trades_df: pd.DataFrame, equity_curve: List[Dict],
                                initial_capital: float) -> Dict:
    """Calculate performance metrics"""
    if len(trades_df) == 0:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_return': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'final_capital': initial_capital
        }
    
    # Basic metrics
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    # Returns
    total_pnl = trades_df['pnl'].sum()
    total_return = (total_pnl / initial_capital) * 100
    
    # Average win/loss
    wins = trades_df[trades_df['pnl'] > 0]['pnl']
    losses = trades_df[trades_df['pnl'] < 0]['pnl']
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses.mean() if len(losses) > 0 else 0
    
    # Equity curve metrics
    equity_df = pd.DataFrame(equity_curve)
    equity_df['returns'] = equity_df['equity'].pct_change()
    
    # Sharpe ratio (annualized, assuming 252 trading days)
    if len(equity_df) > 1 and equity_df['returns'].std() > 0:
        sharpe_ratio = (equity_df['returns'].mean() / equity_df['returns'].std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Maximum drawdown
    equity_df['cummax'] = equity_df['equity'].cummax()
    equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
    max_drawdown = equity_df['drawdown'].min() * 100
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': total_trades - winning_trades,
        'win_rate': win_rate * 100,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': abs(wins.sum() / losses.sum()) if losses.sum() != 0 else 0,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'final_capital': equity_df['equity'].iloc[-1] if len(equity_df) > 0 else initial_capital
    }

# ────────────────────────────────────────────────────────────────────────────────
# Main Function
# ────────────────────────────────────────────────────────────────────────────────
def main():
    """Main function"""
    args = parse_arguments()
    
    # Extract timeframe from primary data filename
    primary_timeframe = extract_timeframe_from_filename(args.primary_data)
    logger = setup_logger(args.symbol, primary_timeframe)
    
    logger.info("Starting TEMA Trend Following Backtest V3 (with realistic limit order execution)")
    logger.info(f"Configuration: {vars(args)}")
    
    # Determine mode
    dual_timeframe_mode = args.secondary_data is not None or args.resample_secondary is not None
    mode = "Dual Timeframe" if dual_timeframe_mode else "Single Timeframe"
    logger.info(f"Running in {mode} mode")
    
    # Log resampling configuration if specified
    if args.resample_primary:
        logger.info(f"Will resample primary data to: {args.resample_primary}")
    if args.resample_secondary:
        logger.info(f"Will resample to create secondary timeframe: {args.resample_secondary}")
    
    try:
        # Load primary data
        primary_df = load_data(args.primary_data, args.start_date, args.end_date)
        
        # Resample primary data if requested
        if args.resample_primary:
            primary_df = resample_data(primary_df, args.resample_primary)
            primary_timeframe = args.resample_primary
        
        # Calculate primary indicators
        primary_df = calculate_indicators(primary_df, 'primary',
                                         tema_primary_fast=args.tema_primary_fast,
                                         tema_primary_slow=args.tema_primary_slow,
                                         tema_secondary_fast=args.tema_secondary_fast,
                                         tema_secondary_slow=args.tema_secondary_slow)
        
        # Handle dual timeframe mode
        if dual_timeframe_mode:
            if args.secondary_data:
                # Load secondary data from file
                secondary_df = load_data(args.secondary_data, args.start_date, args.end_date)
            elif args.resample_secondary:
                # Create secondary data by resampling primary
                logger.info(f"Creating secondary timeframe by resampling to {args.resample_secondary}")
                # Use the original loaded data (before any primary resampling)
                secondary_df = load_data(args.primary_data, args.start_date, args.end_date)
                secondary_df = resample_data(secondary_df, args.resample_secondary)
            
            secondary_df = calculate_indicators(secondary_df, 'secondary',
                                               tema_primary_fast=args.tema_primary_fast,
                                               tema_primary_slow=args.tema_primary_slow,
                                               tema_secondary_fast=args.tema_secondary_fast,
                                               tema_secondary_slow=args.tema_secondary_slow)
            
            # Align timeframes
            signals = align_timeframes(primary_df, secondary_df)
            
            # Generate signals for dual timeframe
            signals = generate_signals_dual_timeframe(signals, args.market_hours_only,
                                                     adx_threshold=args.adx_threshold,
                                                     cmo_long_threshold=args.cmo_long_threshold,
                                                     cmo_short_threshold=args.cmo_short_threshold,
                                                     atr_entry_offset=args.atr_entry_offset,
                                                     atr_stop_loss=args.atr_stop_loss,
                                                     atr_take_profit=args.atr_take_profit)
        else:
            # Single timeframe mode
            signals = generate_signals_single_timeframe(primary_df, args.market_hours_only,
                                                       adx_threshold=args.adx_threshold,
                                                       cmo_long_threshold=args.cmo_long_threshold,
                                                       cmo_short_threshold=args.cmo_short_threshold,
                                                       atr_entry_offset=args.atr_entry_offset,
                                                       atr_stop_loss=args.atr_stop_loss,
                                                       atr_take_profit=args.atr_take_profit)
        
        # Run backtest
        engine = BacktestEngine(
            args.symbol,
            args.initial_capital,
            args.risk_percent,
            args.position_multiplier,
            args.commission,
            args.close_at_eod,
            args.market_hours_only,
            args.bars_to_enter
        )
        trades_df = engine.run(signals)
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save results
        output_prefix = f"{args.symbol}_{primary_timeframe}"
        if dual_timeframe_mode:
            secondary_timeframe = extract_timeframe_from_filename(args.secondary_data) if args.secondary_data else args.resample_secondary
            output_prefix += f"_{secondary_timeframe}"
        
        # Save trades
        if len(trades_df) > 0:
            trades_file = os.path.join(args.output_dir, f"{output_prefix}_trades.csv")
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Trades saved to {trades_file}")
        
        # Save equity curve
        equity_df = pd.DataFrame(engine.equity_curve)
        equity_file = os.path.join(args.output_dir, f"{output_prefix}_equity.csv")
        equity_df.to_csv(equity_file, index=False)
        
        # Calculate and save performance metrics
        metrics = calculate_performance_metrics(
            trades_df, engine.equity_curve, args.initial_capital
        )
        
        # Add configuration to metrics
        metrics['symbol'] = args.symbol
        metrics['primary_timeframe'] = primary_timeframe
        metrics['secondary_timeframe'] = extract_timeframe_from_filename(args.secondary_data) if args.secondary_data else args.resample_secondary if dual_timeframe_mode else None
        metrics['mode'] = mode
        metrics['start_date'] = str(primary_df.index[0])
        metrics['end_date'] = str(primary_df.index[-1])
        
        # Save metrics as JSON
        metrics_file = os.path.join(args.output_dir, f"{output_prefix}_performance.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        # Display results
        print("\n" + "="*60)
        print(f"TEMA TREND FOLLOWING BACKTEST RESULTS - {mode}")
        print("="*60)
        print(f"Symbol: {args.symbol}")
        print(f"Primary Timeframe: {primary_timeframe}")
        if dual_timeframe_mode:
            secondary_timeframe_display = extract_timeframe_from_filename(args.secondary_data) if args.secondary_data else args.resample_secondary
            print(f"Secondary Timeframe: {secondary_timeframe_display}")
        print(f"Period: {primary_df.index[0]} to {primary_df.index[-1]}")
        print(f"Initial Capital: ${args.initial_capital:,.2f}")
        print(f"Final Capital: ${metrics['final_capital']:,.2f}")
        print(f"\nTotal Trades: {metrics['total_trades']}")
        print(f"Winning Trades: {metrics['winning_trades']}")
        print(f"Losing Trades: {metrics['losing_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1f}%")
        print(f"\nTotal P&L: ${metrics['total_pnl']:,.2f}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Average Win: ${metrics['avg_win']:,.2f}")
        print(f"Average Loss: ${metrics['avg_loss']:,.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"\nSharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print("="*60)
        print(f"\nResults saved to: {args.output_dir}/")
        
    except Exception as e:
        logger.error(f"Error during backtest: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()