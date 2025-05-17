"""
Data preprocessing utilities for loading and cleaning price data.
"""

import pandas as pd
import numpy as np
import os

def load_ibkr_data(train_file, test_file=None):
    """
    Load IBKR historical price data from CSV files.
    
    Parameters:
    -----------
    train_file : str
        Path to training data CSV file
    test_file : str, optional
        Path to testing data CSV file
        
    Returns:
    --------
    pd.DataFrame
        Combined price data with columns: open, high, low, close, volume
    """
    # Check if training file exists
    if not os.path.exists(train_file):
        print(f"Training file not found: {train_file}")
        return None
    
    # Load training data
    train_data = pd.read_csv(train_file)
    
    # Rename columns to lowercase
    train_data.columns = [col.lower().strip() for col in train_data.columns]
    
    # Check if required columns exist
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in train_data.columns]
    
    if missing_cols:
        print(f"Missing required columns in training data: {missing_cols}")
        # Try to adapt to different column names
        if 'time' in train_data.columns and 'date' in missing_cols:
            train_data['date'] = train_data['time']
        if 'adjclose' in train_data.columns and 'close' in missing_cols:
            train_data['close'] = train_data['adjclose']
        if 'vol' in train_data.columns and 'volume' in missing_cols:
            train_data['volume'] = train_data['vol']
        
        # Check again
        missing_cols = [col for col in required_cols if col not in train_data.columns]
        if missing_cols:
            print(f"Still missing required columns: {missing_cols}")
            return None
    
    # Load test data if provided
    if test_file and os.path.exists(test_file):
        test_data = pd.read_csv(test_file)
        
        # Rename columns to lowercase
        test_data.columns = [col.lower().strip() for col in test_data.columns]
        
        # Check if required columns exist
        missing_cols = [col for col in required_cols if col not in test_data.columns]
        
        if missing_cols:
            print(f"Missing required columns in testing data: {missing_cols}")
            # Try to adapt to different column names
            if 'time' in test_data.columns and 'date' in missing_cols:
                test_data['date'] = test_data['time']
            if 'adjclose' in test_data.columns and 'close' in missing_cols:
                test_data['close'] = test_data['adjclose']
            if 'vol' in test_data.columns and 'volume' in missing_cols:
                test_data['volume'] = test_data['vol']
            
            # Check again
            missing_cols = [col for col in required_cols if col not in test_data.columns]
            if missing_cols:
                print(f"Still missing required columns: {missing_cols}")
                return None
        
        # Combine data
        combined_data = pd.concat([train_data, test_data], ignore_index=True)
    else:
        combined_data = train_data
    
    # Convert data types
    combined_data['date'] = pd.to_datetime(combined_data['date'])
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    combined_data[numeric_cols] = combined_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
    
    # Sort by date
    combined_data = combined_data.sort_values('date')
    
    # Set date as index
    combined_data.set_index('date', inplace=True)
    
    return combined_data

def preprocess_data(df):
    """
    Preprocess price data for model training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw price data
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed data
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Handle missing values
    processed_df = processed_df.dropna()
    
    # Ensure all values are positive
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in processed_df.columns:
            processed_df = processed_df[processed_df[col] > 0]
    
    # Ensure high >= low
    processed_df = processed_df[processed_df['high'] >= processed_df['low']]
    
    # Ensure high >= close and high >= open
    processed_df = processed_df[processed_df['high'] >= processed_df['close']]
    processed_df = processed_df[processed_df['high'] >= processed_df['open']]
    
    # Ensure low <= close and low <= open
    processed_df = processed_df[processed_df['low'] <= processed_df['close']]
    processed_df = processed_df[processed_df['low'] <= processed_df['open']]
    
    # Remove outliers
    for col in ['open', 'high', 'low', 'close']:
        if col in processed_df.columns:
            # Calculate rolling median and median absolute deviation
            median = processed_df[col].rolling(window=20).median()
            mad = np.abs(processed_df[col] - median).rolling(window=20).median()
            
            # Define outliers as > 5 MAD from the median
            outliers = np.abs(processed_df[col] - median) > 5 * mad
            
            # Remove outliers
            processed_df = processed_df[~outliers]
    
    # Check for abrupt price changes (gaps)
    if len(processed_df) > 1:
        close_to_open = np.abs(processed_df['open'] / processed_df['close'].shift(1) - 1)
        large_gaps = close_to_open > 0.1  # 10% price change overnight
        
        if large_gaps.sum() > 0:
            print(f"Warning: {large_gaps.sum()} large price gaps detected.")
    
    return processed_df