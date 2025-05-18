# src/backtesting/engine.py
"""
Engine for backtesting trading strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os

import config


class BacktestEngine:
    """
    Engine for backtesting trading strategies.
    """
    def __init__(self, initial_capital=100000.0, commission=0.0005, slippage=0.0001):
        """
        Initialize backtesting engine.
        
        Parameters:
        -----------
        initial_capital : float, default=100000.0
            Initial capital for backtesting
        commission : float, default=0.0005
            Commission rate per trade (0.05%)
        slippage : float, default=0.0001
            Slippage rate per trade (0.01%)
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
    def reset(self):
        """Reset backtesting engine."""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, signals, data):
        """
        Run backtest with given signals and price data.
        
        Parameters:
        -----------
        signals : pd.DataFrame
            DataFrame with columns: date, symbol, signal (1 for buy, -1 for sell, 0 for hold)
        data : dict or pd.DataFrame
            If dict: mapping symbols to price DataFrames
            If DataFrame: price data for a single symbol
            
        Returns:
        --------
        dict
            Backtest results including trades, equity curve, and performance metrics
        """
        self.reset()
        
        # If data is a DataFrame, convert it to a dict
        if isinstance(data, pd.DataFrame):
            symbol = signals['symbol'].iloc[0] if 'symbol' in signals.columns else 'SPY'
            data = {symbol: data}
        
        # Sort signals by date
        signals = signals.sort_values('date')
        
        # Ensure date is the index
        if 'date' in signals.columns:
            signals = signals.set_index('date')
        
        # Initialize equity curve
        equity_dates = signals.index.unique()
        self.equity_curve = pd.DataFrame(index=equity_dates, columns=['equity', 'cash', 'holdings'])
        self.equity_curve.loc[equity_dates[0], 'cash'] = self.initial_capital
        self.equity_curve.loc[equity_dates[0], 'holdings'] = 0
        self.equity_curve.loc[equity_dates[0], 'equity'] = self.initial_capital
        
        # Iterate through signals
        for date in equity_dates:
            # Get signals for this date
            date_signals = signals.loc[date]
            
            # Convert to list of dicts if single row
            if not isinstance(date_signals, pd.DataFrame):
                date_signals = [date_signals.to_dict()]
            else:
                date_signals = date_signals.to_dict('records')
            
            # Process signals
            for signal_row in date_signals:
                symbol = signal_row.get('symbol', list(data.keys())[0])
                signal = signal_row.get('signal', 0)
                
                # Get price data for the symbol
                if symbol not in data:
                    continue
                
                price_data = data[symbol]
                if date not in price_data.index:
                    continue
                    
                price = price_data.loc[date, 'close']
                
                # Process signal
                if signal == 1 and symbol not in self.positions:  # Buy
                    # Calculate position size (equal weight for simplicity)
                    available_capital = self.capital * 0.95  # Keep some cash
                    position_size = available_capital / price
                    cost = position_size * price * (1 + self.slippage) * (1 + self.commission)
                    
                    if cost <= self.capital:
                        self.positions[symbol] = {
                            'size': position_size,
                            'entry_price': price,
                            'entry_date': date
                        }
                        self.capital -= cost
                        
                elif signal == -1 and symbol in self.positions:  # Sell
                    position = self.positions[symbol]
                    position_size = position['size']
                    entry_price = position['entry_price']
                    entry_date = position['entry_date']
                    
                    # Calculate proceeds
                    proceeds = position_size * price * (1 - self.slippage) * (1 - self.commission)
                    self.capital += proceeds
                    
                    # Record trade
                    self.trades.append({
                        'symbol': symbol,
                        'entry_date': entry_date,
                        'entry_price': entry_price,
                        'exit_date': date,
                        'exit_price': price,
                        'size': position_size,
                        'pnl': proceeds - (position_size * entry_price),
                        'return': (price / entry_price) - 1,
                        'holding_days': (date - entry_date).days
                    })
                    
                    # Remove position
                    del self.positions[symbol]
            
            # Update equity curve
            total_position_value = sum(
                data[s].loc[date, 'close'] * pos['size'] 
                for s, pos in self.positions.items() 
                if date in data[s].index
            )
            self.equity_curve.loc[date, 'cash'] = self.capital
            self.equity_curve.loc[date, 'holdings'] = total_position_value
            self.equity_curve.loc[date, 'equity'] = self.capital + total_position_value
        
        # Close any remaining positions at the last date
        last_date = equity_dates[-1]
        for symbol, position in list(self.positions.items()):
            if symbol not in data or last_date not in data[symbol].index:
                continue
                
            position_size = position['size']
            entry_price = position['entry_price']
            entry_date = position['entry_date']
            exit_price = data[symbol].loc[last_date, 'close']
            
            # Calculate proceeds
            proceeds = position_size * exit_price * (1 - self.slippage) * (1 - self.commission)
            self.capital += proceeds
            
            # Record trade
            self.trades.append({
                'symbol': symbol,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': last_date,
                'exit_price': exit_price,
                'size': position_size,
                'pnl': proceeds - (position_size * entry_price),
                'return': (exit_price / entry_price) - 1,
                'holding_days': (last_date - entry_date).days
            })
            
            # Remove position
            del self.positions[symbol]
        
        # Update final equity
        self.equity_curve.loc[last_date, 'cash'] = self.capital
        self.equity_curve.loc[last_date, 'holdings'] = 0
        self.equity_curve.loc[last_date, 'equity'] = self.capital
        
        # Calculate performance metrics
        from src.backtesting.performance import calculate_performance_metrics
        performance = calculate_performance_metrics(self.equity_curve, self.trades)
        
        # Prepare results
        results = {
            'trades': pd.DataFrame(self.trades) if self.trades else pd.DataFrame(),
            'equity_curve': self.equity_curve,
            'performance': performance
        }
        
        return results
    
    def plot_equity_curve(self, save_path=None):
        """
        Plot equity curve.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        import matplotlib.dates as mdates
        from matplotlib.ticker import AutoLocator
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convert dates to numeric format matplotlib can handle
        date_nums = mdates.date2num(self.equity_curve.index.to_pydatetime())
        
        # Plot equity components
        ax.plot(date_nums, self.equity_curve['equity'], label='Total Equity')
        ax.plot(date_nums, self.equity_curve['cash'], label='Cash', alpha=0.7)
        ax.plot(date_nums, self.equity_curve['holdings'], label='Holdings', alpha=0.7)
        
        # Format x-axis with dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(AutoLocator())
        plt.xticks(rotation=45)
        
        ax.set_title('Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value ($)')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        return fig


class WalkforwardBacktester:
    """
    Walkforward backtesting implementation.
    """
    def __init__(self, data, model_factory, feature_engineer, train_size=252*10, test_size=252, step_size=63, 
                 initial_capital=100000.0, commission=0.0005, slippage=0.0001):
        """
        Initialize walkforward backtester.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
        model_factory : callable
            Function that creates and trains a model given X_train and y_train
        feature_engineer : callable
            Function that creates features given price data
        train_size : int, default=252*10
            Size of training window in days
        test_size : int, default=252
            Size of testing window in days
        step_size : int, default=63
            Step size in days for moving the window forward
        initial_capital : float, default=100000.0
            Initial capital for backtesting
        commission : float, default=0.0005
            Commission rate per trade
        slippage : float, default=0.0001
            Slippage rate per trade
        """
        self.data = data
        self.model_factory = model_factory
        self.feature_engineer = feature_engineer
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
    def run(self, symbol='SPY', min_train_samples=252):
        """
        Run walkforward backtest.
        
        Parameters:
        -----------
        symbol : str, default='SPY'
            Trading symbol
        min_train_samples : int, default=252
            Minimum number of samples required for training
            (global BUY/SELL thresholds from BaseStrategy are used)
            
        Returns:
        --------
        dict
            Walkforward backtest results
        """
        from src.backtesting.signal_generation import generate_signals
        
        # Calculate total number of days
        total_days = len(self.data)
        
        # Calculate number of windows
        num_windows = (total_days - self.train_size - self.test_size) // self.step_size + 1
        
        # Initialize results
        all_trades = []
        all_predictions = []
        all_equity_curves = []
        window_results = []
        
        print(f"Running walkforward backtest with {num_windows} windows...")
        
        # Loop through windows
        for i in range(num_windows):
            # Calculate window indices
            train_start = i * self.step_size
            train_end = train_start + self.train_size
            test_start = train_end
            test_end = min(test_start + self.test_size, total_days)
            
            # Check if we have enough data
            if test_end - test_start < 10:
                print(f"Window {i+1}/{num_windows}: Insufficient test data, skipping")
                continue
            
            print(f"Window {i+1}/{num_windows}: Training on {train_start}:{train_end}, Testing on {test_start}:{test_end}")
            
            # Get data for this window
            train_data = self.data.iloc[train_start:train_end]
            test_data = self.data.iloc[test_start:test_end]
            
            # Check if we have enough training samples
            if len(train_data) < min_train_samples:
                print(f"Window {i+1}/{num_windows}: Insufficient training data, skipping")
                continue
            
            # Create features
            X_train, y_train, dates_train = self.feature_engineer(train_data)
            X_test, y_test, dates_test = self.feature_engineer(test_data)
            
            if len(X_train) < min_train_samples or len(X_test) < 10:
                print(f"Window {i+1}/{num_windows}: Insufficient samples after feature engineering, skipping")
                continue
            
            # Train model
            print(f"  Training model with {len(X_train)} samples...")
            model = self.model_factory(X_train, y_train)
            
            # Generate signals
            print(f"  Generating signals for {len(X_test)} test samples...")
            signals = generate_signals(model, X_test, dates_test, symbol)
            
            # Run backtest
            print(f"  Running backtest...")
            backtest = BacktestEngine(
                initial_capital=self.initial_capital,
                commission=self.commission,
                slippage=self.slippage
            )
            backtest_results = backtest.run_backtest(signals, {symbol: test_data})
            
            # Store results
            window_result = {
                'window': i,
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'performance': backtest_results['performance'],
                'model': model,
                'signals': signals
            }
            window_results.append(window_result)
            
            if not backtest_results['trades'].empty:
                all_trades.extend(backtest_results['trades'].to_dict('records'))
            
            all_predictions.append({
                'window': i,
                'predictions': model.predict(X_test),
                'actual': y_test,
                'dates': dates_test,
                'accuracy': window_result['performance'].get('accuracy', 0)
            })
            
            all_equity_curves.append(backtest_results['equity_curve'])
            
        # Combine results
        if all_equity_curves:
            combined_equity = pd.concat(all_equity_curves)
            combined_equity = combined_equity[~combined_equity.index.duplicated(keep='first')]
            combined_equity = combined_equity.sort_index()
        else:
            combined_equity = pd.DataFrame(columns=['equity', 'cash', 'holdings'])
        
        # Calculate overall performance
        from src.backtesting.performance import calculate_performance_metrics
        overall_performance = calculate_performance_metrics(combined_equity, all_trades)
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
        
        # Prepare results
        results = {
            'window_results': window_results,
            'trades': trades_df,
            'equity_curve': combined_equity,
            'performance': overall_performance,
            'predictions': all_predictions
        }
        
        return results
