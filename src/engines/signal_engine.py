"""
Signal engine for generating trading signals from model predictions.
"""

import pandas as pd
import numpy as np
from src.strategies.base_strategy import BUY_THRESHOLD, SELL_THRESHOLD

class SignalEngine:
    """
    Engine for generating trading signals from model predictions.
    """

    def __init__(self, position_sizing=None):
        """
        Initialize the signal engine.

        Parameters
        ----------
        position_sizing : {'fixed', 'confidence'}, optional
            Method for determining position size. ``'fixed'`` uses a full
            position (1.0) for any non-zero signal. ``'confidence'`` scales
            size based on the distance of the probability from ``0.5`` using
            square-root weighting. Defaults to ``'confidence'`` when ``None``.
        """
        # Signals rely on the module-level BUY_THRESHOLD and SELL_THRESHOLD.
        self.position_sizing = position_sizing


    def generate_signals(self, predictions, dates, symbol='SPY'):
        """
        Generate trading signals from model predictions.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Predicted probabilities from model
        dates : pd.DatetimeIndex or list
            Dates corresponding to predictions
        symbol : str, default='SPY'
            Trading symbol
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: date, symbol, signal, probability, position_size
        """
        # Ensure dates is a pandas Series or DatetimeIndex
        if not isinstance(dates, (pd.Series, pd.DatetimeIndex)):
            dates = pd.Series(dates)

        # Create signals DataFrame
        signals = []

        for i, probability in enumerate(predictions):
            # Determine signal using global thresholds
            if probability >= BUY_THRESHOLD:  # Buy signal
                signal = 1
            elif probability <= SELL_THRESHOLD:  # Sell signal
                signal = -1
            else:                                # Hold
                signal = 0

            # Determine position size
            if self.position_sizing == 'fixed':
                # Full size whenever a signal is generated
                position_size = 1.0 if signal != 0 else 0.0
            else:
                # Default or 'confidence' sizing based on prediction confidence
                position_size = (abs(probability - 0.5) * 2) ** 0.5
                if signal == 0:
                    position_size = 0.0

            signals.append({
                'date': dates.iloc[i] if hasattr(dates, 'iloc') else dates[i],
                'symbol': symbol,
                'signal': signal,
                'probability': probability,
                'position_size': position_size
            })

        # Convert to DataFrame
        signals_df = pd.DataFrame(signals)

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(signals_df['date']):
            signals_df['date'] = pd.to_datetime(signals_df['date'])

        return signals_df

    def apply_filters(self, signals, consecutive_buys=False, min_holding_days=1, max_holding_days=None):
        """
        Apply filtering rules to trading signals.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            DataFrame with trading signals
        consecutive_buys : bool, default=False
            Whether to allow consecutive buy signals
        min_holding_days : int, default=1
            Minimum number of days to hold a position
        max_holding_days : int, default=None
            Maximum number of days to hold a position
            
        Returns:
        --------
        pd.DataFrame
            Filtered signals DataFrame
        """
        # Make a copy to avoid modifying the original
        filtered_signals = signals.copy()

        # Apply consecutive buys filter
        if not consecutive_buys:
            # Reset signals where we already have a position (consecutive buys)
            in_position = False
            for i in range(len(filtered_signals)):
                if filtered_signals.iloc[i]['signal'] == 1:  # Buy signal
                    if in_position:
                        # Change to hold
                        filtered_signals.loc[filtered_signals.index[i], 'signal'] = 0
                        filtered_signals.loc[filtered_signals.index[i], 'position_size'] = 0.0
                    else:
                        in_position = True
                elif filtered_signals.iloc[i]['signal'] == -1:  # Sell signal
                    in_position = False

        # Apply minimum holding period
        if min_holding_days > 1:
            # Sort by date
            filtered_signals = filtered_signals.sort_values('date')
            
            # Track positions and their entry dates
            position_start = None
            for i in range(len(filtered_signals)):
                date = filtered_signals.iloc[i]['date']
                signal = filtered_signals.iloc[i]['signal']
                
                if signal == 1 and position_start is None:  # Enter position
                    position_start = date
                elif signal == -1 and position_start is not None:  # Exit position
                    # Check if minimum holding period met
                    holding_days = (date - position_start).days
                    if holding_days < min_holding_days:
                        # Change to hold (delay exit)
                        filtered_signals.loc[filtered_signals.index[i], 'signal'] = 0
                    else:
                        position_start = None
        
        # Apply maximum holding period
        if max_holding_days is not None and max_holding_days > 0:
            # This would need to force an exit after max_holding_days
            # Implementation depends on how the backtesting system works
            pass

        return filtered_signals
    
    def add_position_info(self, signals):
        """
        Add position information to signals DataFrame.
        
        This adds columns for current position, entry date, and entry price,
        which can be useful for backtesting and analysis.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            DataFrame with trading signals
            
        Returns:
        --------
        pd.DataFrame
            Signals DataFrame with additional position info
        """
        # Make a copy to avoid modifying the original
        result = signals.copy()
        
        # Add columns
        result['position'] = 0
        result['entry_date'] = pd.NaT
        result['entry_price'] = np.nan
        
        # Calculate positions
        in_position = False
        entry_date = None
        entry_price = None
        
        for i in range(len(result)):
            signal = result.iloc[i]['signal']
            date = result.iloc[i]['date']
            
            if signal == 1 and not in_position:  # Enter position
                in_position = True
                entry_date = date
                # This assumes close price is available in the DataFrame
                if 'close' in result.columns:
                    entry_price = result.iloc[i]['close']
                
            elif signal == -1 and in_position:  # Exit position
                in_position = False
                entry_date = None
                entry_price = None
            
            # Update position info
            result.loc[result.index[i], 'position'] = 1 if in_position else 0
            if in_position:
                result.loc[result.index[i], 'entry_date'] = entry_date
                result.loc[result.index[i], 'entry_price'] = entry_price
        
        return result
