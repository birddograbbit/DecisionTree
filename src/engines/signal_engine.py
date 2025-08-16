"""
Signal engine for generating trading signals from model predictions.
"""

import pandas as pd
import numpy as np
from src.utils.threshold_manager import ThresholdManager

class SignalEngine:
    """
    Engine for generating trading signals from model predictions.

    This class handles the conversion of model predictions into actionable
    trading signals, with various filtering and position sizing options.
    """

    def __init__(self, position_sizing="confidence", config=None):
        """
        Initialize the signal engine.

        Parameters
        ----------
        position_sizing : {'fixed', 'confidence'}, optional
            Determines how position sizes are calculated. ``'fixed'`` applies a
            constant size of 1.0 whenever a non-zero signal is generated.
            ``'confidence'`` (default) scales size based on the distance of the
            probability from ``0.5`` using square-root weighting.
        config : dict, optional
            Configuration dictionary for threshold management
        """
        self.position_sizing = position_sizing
        self.config = config or {}
        self.threshold_manager = ThresholdManager(self.config)
        
    def check_for_adaptive_thresholds(self, predictions):
        """
        Check if adaptive thresholds should be used based on the prediction distribution.
        
        This method is deprecated - thresholds are now managed centrally.
        Kept for backward compatibility.
        
        Parameters:
        -----------
        predictions : np.ndarray
            Predicted probabilities from model
            
        Returns:
        --------
        tuple
            (should_use_adaptive, buy_threshold, sell_threshold)
        """
        print("Warning: check_for_adaptive_thresholds is deprecated. Thresholds are managed centrally.")
        buy_threshold, sell_threshold = self.threshold_manager.get_thresholds(predictions)
        use_adaptive = self.threshold_manager._should_use_adaptive_thresholds(predictions)
        return use_adaptive, buy_threshold, sell_threshold
        
    def get_position_size(self, prob, max_size=1.0, vol_target=0.10):
        """Edge-based position sizing with simple volatility targeting."""
        edge = abs(prob - 0.5)
        if edge < 0.02:
            return 0.0
        base_size = np.clip(edge * 4, 0, max_size)
        vol_scalar = min(1.0, vol_target / 0.15)
        return base_size * vol_scalar

    def generate_signals(self, predictions, dates, symbol='SPY', custom_thresholds=None):
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
        custom_thresholds : tuple, optional
            Custom (buy_threshold, sell_threshold) to use
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with columns: date, symbol, signal, probability, position_size
        """
        # Ensure dates is a pandas Series or DatetimeIndex
        if not isinstance(dates, (pd.Series, pd.DatetimeIndex)):
            dates = pd.Series(dates)
            
        # Get thresholds from centralized manager
        if custom_thresholds:
            buy_threshold, sell_threshold = custom_thresholds
        else:
            buy_threshold, sell_threshold = self.threshold_manager.get_thresholds(predictions)

        # Create signals DataFrame
        signals = []

        for i, probability in enumerate(predictions):
            # Determine signal using thresholds
            signal = self.threshold_manager.prob_to_signal(probability, predictions)

            # Determine position size using centralized logic
            position_size = self.get_position_size(probability)
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

    def apply_filters(self, signals, consecutive_buys=False, min_holding_days=1, max_holding_days=None,
                      max_trades_per_day=None):
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

        # Sort by date to ensure chronological processing
        filtered_signals = filtered_signals.sort_values('date').reset_index(drop=True)

        # Apply consecutive buys filter
        if not consecutive_buys:
            # Reset signals where we already have a position (consecutive buys)
            in_position = False
            for i in range(len(filtered_signals)):
                if filtered_signals.iloc[i]['signal'] == 1:  # Buy signal
                    if in_position:
                        # Change to hold
                        filtered_signals.loc[i, 'signal'] = 0
                        filtered_signals.loc[i, 'position_size'] = 0.0
                    else:
                        in_position = True
                elif filtered_signals.iloc[i]['signal'] == -1:  # Sell signal
                    in_position = False

        # Apply minimum and maximum holding period constraints
        if min_holding_days > 1 or (max_holding_days is not None and max_holding_days > 0):
            # Track positions and their entry dates
            position_start = None
            in_position = False
            
            for i in range(len(filtered_signals)):
                date = filtered_signals.iloc[i]['date']
                signal = filtered_signals.iloc[i]['signal']
                
                # Handle position entry
                if signal == 1 and not in_position:  # Enter position
                    position_start = date
                    in_position = True
                    
                # Handle position exit
                elif signal == -1 and in_position:  # Exit position
                    holding_days = (date - position_start).days
                    
                    # Check minimum holding period
                    if holding_days < min_holding_days:
                        # Change to hold (delay exit) - minimum not met
                        filtered_signals.loc[i, 'signal'] = 0
                        filtered_signals.loc[i, 'position_size'] = 0.0
                        # Keep position_start unchanged, still in position
                    else:
                        # Minimum holding period met, allow exit
                        position_start = None
                        in_position = False
                
                # Force exit if maximum holding period exceeded
                elif in_position and max_holding_days is not None and max_holding_days > 0:
                    holding_days = (date - position_start).days
                    
                    if holding_days >= max_holding_days:
                        # Force exit due to maximum holding period
                        filtered_signals.loc[i, 'signal'] = -1  # Force sell signal
                        # Calculate position size for forced exit using centralized logic
                        probability = filtered_signals.iloc[i]['probability']
                        filtered_signals.loc[i, 'position_size'] = self.threshold_manager.get_position_size(
                            probability, self.position_sizing
                        )
                        
                        # Reset position tracking
                        position_start = None
                        in_position = False
                        
                        print(f"Forced exit on {date.strftime('%Y-%m-%d')} due to max holding period of {max_holding_days} days")

        # Limit number of trades per day if specified
        if max_trades_per_day is not None:
            filtered_signals['date_only'] = filtered_signals['date'].dt.date
            for date in filtered_signals['date_only'].unique():
                mask = filtered_signals['date_only'] == date
                date_signals = filtered_signals[mask & (filtered_signals['signal'] != 0)]
                if len(date_signals) > max_trades_per_day:
                    filtered_signals.loc[mask, 'edge'] = abs(
                        filtered_signals.loc[mask, 'probability'] - 0.5)
                    keep_idx = filtered_signals.loc[mask].nlargest(
                        max_trades_per_day, 'edge').index
                    drop_idx = filtered_signals.loc[mask].index.difference(keep_idx)
                    filtered_signals.loc[drop_idx, 'signal'] = 0
                    filtered_signals.loc[drop_idx, 'position_size'] = 0.0
            filtered_signals = filtered_signals.drop(columns=['date_only', 'edge'], errors='ignore')

        return filtered_signals
    
    def add_position_info(self, signals):
        """
        Add position information to signals DataFrame.

        This adds columns for current position, entry date, entry price, and holding days,
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
        
        # Sort by date to ensure chronological processing
        result = result.sort_values('date').reset_index(drop=True)
        
        # Add columns
        result['position'] = 0
        result['entry_date'] = pd.NaT
        result['entry_price'] = np.nan
        result['holding_days'] = 0
        
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
            result.loc[i, 'position'] = 1 if in_position else 0
            if in_position:
                result.loc[i, 'entry_date'] = entry_date
                result.loc[i, 'entry_price'] = entry_price
                # Calculate holding days
                holding_days = (date - entry_date).days
                result.loc[i, 'holding_days'] = holding_days
        
        return result

    def get_position_summary(self, signals_with_position_info):
        """
        Get a summary of positions from signals with position information.
        
        Parameters:
        -----------
        signals_with_position_info : pd.DataFrame
            DataFrame with position information (from add_position_info)
            
        Returns:
        --------
        pd.DataFrame
            Summary of positions with entry/exit dates, holding periods, etc.
        """
        # Filter to only position entries (signal == 1)
        entries = signals_with_position_info[signals_with_position_info['signal'] == 1].copy()
        
        # Filter to only position exits (signal == -1)
        exits = signals_with_position_info[signals_with_position_info['signal'] == -1].copy()
        
        # Create position summary
        positions = []
        
        for i, entry in entries.iterrows():
            entry_date = entry['date']
            entry_price = entry.get('entry_price', np.nan)
            
            # Find corresponding exit
            exit_after_entry = exits[exits['date'] > entry_date]
            
            if len(exit_after_entry) > 0:
                # Found exit
                exit_info = exit_after_entry.iloc[0]
                exit_date = exit_info['date']
                exit_price = exit_info.get('close', np.nan)
                holding_days = (exit_date - entry_date).days
                
                # Calculate return if prices available
                if pd.notna(entry_price) and pd.notna(exit_price):
                    position_return = (exit_price - entry_price) / entry_price
                else:
                    position_return = np.nan
                    
            else:
                # No exit found (still holding or end of data)
                exit_date = None
                exit_price = np.nan
                position_return = np.nan
                
                # Check if position is still active at end of data
                last_date = signals_with_position_info['date'].max()
                last_position_info = signals_with_position_info[
                    signals_with_position_info['date'] == last_date
                ].iloc[0]
                
                if last_position_info['position'] == 1:
                    holding_days = (last_date - entry_date).days
                else:
                    holding_days = np.nan
            
            positions.append({
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'holding_days': holding_days,
                'return': position_return,
                'entry_probability': entry['probability'],
                'position_size': entry['position_size']
            })
        
        return pd.DataFrame(positions)

    def get_threshold_configuration(self):
        """
        Get summary of current threshold configuration.
        
        Returns:
        --------
        dict
            Threshold configuration summary
        """
        return self.threshold_manager.get_configuration_summary()
