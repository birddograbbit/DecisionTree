# src/data/preprocessing.py
"""
Module for preprocessing raw data.
"""

import pandas as pd
import numpy as np
from typing import Union, List

# src/data/preprocessing.py (Add this function)

def _load_csv_files(files: Union[str, List[str]]) -> pd.DataFrame:
    """Helper to load one or multiple CSV files and concatenate."""
    if isinstance(files, list):
        dfs = [pd.read_csv(f) for f in files]
        return pd.concat(dfs)
    return pd.read_csv(files)


def load_ibkr_data(train_file, test_file):
    """
    Load and combine IBKR historical data files.
    
    Parameters:
    -----------
    train_file : str
        Path to training data file
    test_file : str
        Path to testing data file
        
    Returns:
    --------
    pd.DataFrame
        Combined and preprocessed data
    """
    # Load training and testing data
    print(f"Loading training data from {train_file}...")
    train_data = _load_csv_files(train_file)

    print(f"Loading testing data from {test_file}...")
    test_data = _load_csv_files(test_file)
    
    # Combine the data
    combined_data = pd.concat([train_data, test_data])
    
    # Convert date column to datetime and set as index
    combined_data['date'] = pd.to_datetime(combined_data['date'], utc=True)
    combined_data['date'] = combined_data['date'].dt.tz_convert('UTC').dt.tz_localize(None)
    combined_data.set_index('date', inplace=True)
    
    # Sort by date (no need to reverse since IBKR data is already in chronological order)
    combined_data = combined_data.sort_index()
    
    # Make column names lowercase
    combined_data.columns = combined_data.columns.str.lower()
    
    # Check for missing values
    missing_count = combined_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: Found {missing_count} missing values in the data.")
        combined_data = combined_data.dropna()
        print(f"Dropped rows with missing values. New shape: {combined_data.shape}")
    
    return combined_data


def load_5min_data(train_file, test_file):
    """
    Load and combine 5-minute IBKR historical data files.
    
    Parameters:
    -----------
    train_file : str
        Path to training data file (5-minute bars)
    test_file : str
        Path to testing data file (5-minute bars)
        
    Returns:
    --------
    pd.DataFrame
        Combined and preprocessed 5-minute data
    """
    # Load training and testing data
    print(f"Loading 5-minute training data from {train_file}...")
    train_data = _load_csv_files(train_file)

    print(f"Loading 5-minute testing data from {test_file}...")
    test_data = _load_csv_files(test_file)
    
    # Combine the data
    combined_data = pd.concat([train_data, test_data])
    
    # Convert date column to datetime and handle timezone
    combined_data['date'] = pd.to_datetime(combined_data['date'], utc=True)
    combined_data['date'] = combined_data['date'].dt.tz_convert('UTC').dt.tz_localize(None)
    combined_data.set_index('date', inplace=True)
    
    # Sort by date
    combined_data = combined_data.sort_index()
    
    # Make column names lowercase
    combined_data.columns = combined_data.columns.str.lower()
    
    # Check for missing values
    missing_count = combined_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: Found {missing_count} missing values in the data.")
        # Drop rows with any missing values for 5-minute data
        combined_data = combined_data.dropna()
        print(f"Dropped rows with missing values. New shape: {combined_data.shape}")
    
    print(f"5-minute data loaded and preprocessed. Shape: {combined_data.shape}")
    print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")
    
    return combined_data


def load_1min_data(train_file, test_file):
    """Load and combine 1-minute IBKR historical data files.

    Parameters
    ----------
    train_file : str
        Path to training data file (1-minute bars)
    test_file : str
        Path to testing data file (1-minute bars)

    Returns
    -------
    pd.DataFrame
        Combined and preprocessed 1-minute data
    """
    print(f"Loading 1-minute training data from {train_file}...")
    train_data = _load_csv_files(train_file)

    print(f"Loading 1-minute testing data from {test_file}...")
    test_data = _load_csv_files(test_file)

    combined_data = pd.concat([train_data, test_data])

    # Parse timestamps and convert to timezone-naive UTC
    combined_data['date'] = pd.to_datetime(combined_data['date'], utc=True)
    combined_data['date'] = combined_data['date'].dt.tz_convert('UTC').dt.tz_localize(None)
    combined_data.set_index('date', inplace=True)

    combined_data = combined_data.sort_index()
    combined_data.columns = combined_data.columns.str.lower()

    missing_count = combined_data.isnull().sum().sum()
    if missing_count > 0:
        print(f"Warning: Found {missing_count} missing values in the data.")
        combined_data = combined_data.dropna()
        print(f"Dropped rows with missing values. New shape: {combined_data.shape}")

    print(f"1-minute data loaded and preprocessed. Shape: {combined_data.shape}")
    print(f"Date range: {combined_data.index.min()} to {combined_data.index.max()}")

    return combined_data

def preprocess_data(df):
    """
    Preprocess raw price data for model training.
    
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
    df = df.copy()
    
    # Handle missing values
    df = df.dropna()
    
    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
    
    # Sort by date
    df = df.sort_index()
    
    # Ensure all required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = set(required_cols) - set(df.columns.str.lower())
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Standardize column names (lowercase)
    df.columns = [col.lower() for col in df.columns]
    
    # Add additional preprocessing as needed
    
    return df

def load_and_preprocess_data(file_path):
    """
    Load and preprocess data from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed data
    """
    try:
        # Load data
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Preprocess data
        df = preprocess_data(df)
        
        return df
    except Exception as e:
        print(f"Error loading and preprocessing data from {file_path}: {e}")
        return None
