# src/data/data_acquisition.py
"""
Module for acquiring historical data from IBKR.
"""

import os
import datetime
import pandas as pd
from ib_insync import IB, Stock, util

class DataAcquisition:
    """
    Class for acquiring historical data from IBKR.
    """
    
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        """
        Initialize DataAcquisition with IBKR connection parameters.
        
        Parameters:
        -----------
        host : str
            IBKR host (default: 127.0.0.1)
        port : int
            IBKR port (default: 7497 for IB Gateway paper trading)
        client_id : int
            IBKR client ID (default: 1)
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
    
    def connect(self):
        """
        Connect to IBKR.
        
        Returns:
        --------
        bool
            True if connection successful, False otherwise
        """
        try:
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            return self.ib.isConnected()
        except Exception as e:
            print(f"Error connecting to IBKR: {e}")
            return False
    
    def disconnect(self):
        """
        Disconnect from IBKR.
        """
        if self.ib and self.ib.isConnected():
            self.ib.disconnect()
    
    def fetch_historical_data(self, symbol, start_date, end_date, timeframe='1 day'):
        """
        Fetch historical price data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Stock symbol
        start_date : str
            Start date in format 'YYYYMMDD'
        end_date : str
            End date in format 'YYYYMMDD'
        timeframe : str
            Bar size setting (default: '1 day')
            
        Returns:
        --------
        pd.DataFrame or None
            Historical price data with columns: date, open, high, low, close, volume
            Returns None if error occurs
        """
        try:
            # Connect to IBKR if not already connected
            if not self.ib or not self.ib.isConnected():
                if not self.connect():
                    print("Cannot fetch data: Not connected to IBKR")
                    return None
            
            # Create contract for the symbol
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Calculate duration based on start and end dates
            start_dt = datetime.datetime.strptime(start_date, '%Y%m%d')
            end_dt = datetime.datetime.strptime(end_date, '%Y%m%d')
            duration = f"{(end_dt - start_dt).days + 1} D"
            
            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=timeframe,
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            # Convert to DataFrame
            df = util.df(bars)
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def save_data(self, df, symbol, path='data/raw'):
        """
        Save data to CSV file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to save
        symbol : str
            Stock symbol
        path : str
            Path to save data (default: 'data/raw')
            
        Returns:
        --------
        str or None
            Path to saved file if successful, None otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Create file path
            file_path = os.path.join(path, f"{symbol}.csv")
            
            # Save to CSV
            df.to_csv(file_path, index=True)
            
            print(f"Data saved to {file_path}")
            return file_path
            
        except Exception as e:
            print(f"Error saving data for {symbol}: {e}")
            return None
    
    def fetch_and_save_data(self, symbols, start_date, end_date, timeframe='1 day', path='data/raw'):
        """
        Fetch and save historical data for multiple symbols.
        
        Parameters:
        -----------
        symbols : list
            List of stock symbols
        start_date : str
            Start date in format 'YYYYMMDD'
        end_date : str
            End date in format 'YYYYMMDD'
        timeframe : str
            Bar size setting (default: '1 day')
        path : str
            Path to save data (default: 'data/raw')
            
        Returns:
        --------
        dict
            Dictionary mapping symbols to file paths
        """
        results = {}
        
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            df = self.fetch_historical_data(symbol, start_date, end_date, timeframe)
            
            if df is not None and not df.empty:
                file_path = self.save_data(df, symbol, path)
                if file_path:
                    results[symbol] = file_path
            else:
                print(f"No data available for {symbol}")
        
        self.disconnect()
        return results
