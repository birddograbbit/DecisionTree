# src/backtesting/signal_generation.py
"""
Module for generating trading signals from model predictions.
"""

import pandas as pd
import numpy as np
from src.strategies.base_strategy import BUY_THRESHOLD, SELL_THRESHOLD


def generate_signals(model, X, dates, symbol='SPY'):
    """
    Generate trading signals from model predictions.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier
        Trained model
    X : pd.DataFrame
        Feature matrix
    dates : list or pd.DatetimeIndex
        Dates corresponding to the feature matrix rows
    symbol : str, default='SPY'
        Trading symbol
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: date, symbol, signal, probability
    """
    # Get predicted probabilities
    proba = model.predict_proba(X)[:, 1]  # Probability of class 1 (up)
    
    # Convert dates to pandas Series if it's not already
    if not isinstance(dates, pd.Series):
        dates = pd.Series(dates)
    
    # Create signals DataFrame using global thresholds
    signals = []
    for i in range(len(X)):
        if proba[i] > BUY_THRESHOLD:  # Buy signal
            signals.append({
                'date': dates.iloc[i],
                'symbol': symbol,
                'signal': 1,  # Buy
                'probability': proba[i]
            })
        elif proba[i] < SELL_THRESHOLD:  # Sell signal
            signals.append({
                'date': dates.iloc[i],
                'symbol': symbol,
                'signal': -1,  # Sell
                'probability': proba[i]
            })
        else:  # Hold
            signals.append({
                'date': dates.iloc[i],
                'symbol': symbol,
                'signal': 0,  # Hold
                'probability': proba[i]
            })
    
    # Convert to DataFrame
    signals_df = pd.DataFrame(signals)
    
    # Ensure date column is datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(signals_df['date']):
        signals_df['date'] = pd.to_datetime(signals_df['date'])
    
    return signals_df


def apply_signal_rules(signals_df, consecutive_buys=False, min_probability=0.0):
    """
    Apply additional rules to trading signals.
    
    Parameters:
    -----------
    signals_df : pd.DataFrame
        DataFrame with signals
    consecutive_buys : bool, default=False
        Whether to allow consecutive buy signals
    min_probability : float, default=0.0
        Minimum probability threshold
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with filtered signals
    """
    # Make a copy to avoid modifying the original
    filtered_signals = signals_df.copy()
    
    # Apply minimum probability filter
    if min_probability > 0:
        mask = filtered_signals['probability'] >= min_probability
        filtered_signals.loc[~mask, 'signal'] = 0  # Change to hold
    
    # Apply consecutive buys filter
    if not consecutive_buys:
        # Reset signals where we already have a position (consecutive buys)
        in_position = False
        for i in range(len(filtered_signals)):
            if filtered_signals.iloc[i]['signal'] == 1:  # Buy signal
                if in_position:
                    filtered_signals.iloc[i, filtered_signals.columns.get_loc('signal')] = 0  # Change to hold
                else:
                    in_position = True
            elif filtered_signals.iloc[i]['signal'] == -1:  # Sell signal
                in_position = False
    
    return filtered_signals


def calculate_positions(signals_df, initial_capital=100000, price_data=None):
    """
    Calculate positions and capital based on signals.
    
    Parameters:
    -----------
    signals_df : pd.DataFrame
        DataFrame with signals
    initial_capital : float, default=100000
        Initial capital
    price_data : pd.DataFrame, optional
        Price data with columns: date, close
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with positions and capital
    """
    # Make a copy to avoid modifying the original
    positions_df = signals_df.copy()
    
    # Add columns for positions and capital
    positions_df['position'] = 0
    positions_df['capital'] = initial_capital
    
    # Calculate positions
    position = 0
    capital = initial_capital
    prev_position = 0

    for i in range(len(positions_df)):
        signal = positions_df.iloc[i]['signal']
        prev_position = position

        # Update position based on signal
        if signal == 1:  # Buy
            position = 1
        elif signal == -1:  # Sell
            position = 0

        # Store position
        positions_df.iloc[i, positions_df.columns.get_loc('position')] = position

        # Update capital if price data is provided
        if price_data is not None:
            date = positions_df.iloc[i]['date']
            if date in price_data.index:
                price = price_data.loc[date, 'close']
                # Deduct cost on buy and add proceeds on sell
                if position > prev_position:
                    capital -= price
                elif position < prev_position:
                    capital += price

        # Store capital
        positions_df.iloc[i, positions_df.columns.get_loc('capital')] = capital
    
    return positions_df
