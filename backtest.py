#!/usr/bin/env python
"""
This is a comprehensive backtesting module for trading strategies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def calculate_returns(prices, positions, transaction_cost=0.001):
    """
    Calculate returns based on price data and positions.
    
    Parameters:
    -----------
    prices : pandas.Series
        Historical price data
    positions : pandas.Series
        Position signals (1=long, 0=neutral, -1=short)
    transaction_cost : float, default=0.001
        Transaction cost as a percentage (e.g., 0.001 = 0.1%)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with strategy returns and statistics
    """
    # Ensure positions are aligned with prices
    positions = positions.reindex(prices.index)
    positions = positions.fillna(0)
    
    # Calculate price returns
    price_returns = prices.pct_change().fillna(0)
    
    # Calculate position changes to determine transaction costs
    position_changes = positions.diff().fillna(0).abs()
    transaction_costs = position_changes * transaction_cost
    
    # Calculate strategy returns
    strategy_returns = positions.shift(1) * price_returns - transaction_costs
    strategy_returns = strategy_returns.fillna(0)
    
    # Cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Create result DataFrame
    results = pd.DataFrame({
        'price': prices,
        'position': positions,
        'price_return': price_returns,
        'strategy_return': strategy_returns,
        'cumulative_return': cumulative_returns
    })
    
    return results

def calculate_performance_metrics(returns):
    """
    Calculate performance metrics for a returns series.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of returns
        
    Returns:
    --------
    dict
        Dictionary containing performance metrics
    """
    # Annualization factor (assuming daily data)
    annual_factor = 252
    
    # Total return
    total_return = (1 + returns).prod() - 1
    
    # Annual return (CAGR)
    years = len(returns) / annual_factor
    cagr = (1 + total_return) ** (1 / years) - 1
    
    # Volatility (annualized)
    volatility = returns.std() * np.sqrt(annual_factor)
    
    # Sharpe ratio (assuming Rf=0 for simplicity)
    sharpe_ratio = (returns.mean() * annual_factor) / volatility if volatility != 0 else 0
    
    # Maximum drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_drawdown = drawdown.min()
    
    # Win rate
    win_rate = (returns > 0).mean()
    
    # Profit factor
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    profit_factor = abs(gains / losses) if losses != 0 else float('inf')
    
    # CAGR/MaxDD ratio
    cagr_maxdd_ratio = abs(cagr / max_drawdown) if max_drawdown != 0 else float('inf')
    
    return {
        'total_return': total_return,
        'cagr': cagr,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'cagr_maxdd_ratio': cagr_maxdd_ratio
    }

def plot_backtest_results(results, title="Backtest Results"):
    """
    Plot backtest results with price, cumulative returns, and position signals.
    
    Parameters:
    -----------
    results : pandas.DataFrame
        DataFrame with backtest results
    title : str, default="Backtest Results"
        Plot title
    """
    # Create figure and subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Plot price
    ax1.plot(results.index, results['price'])
    ax1.set_ylabel('Price')
    ax1.set_title(title)
    ax1.grid(True)
    
    # Plot cumulative returns
    ax2.plot(results.index, results['cumulative_return'])
    ax2.set_ylabel('Cumulative Return')
    ax2.grid(True)
    
    # Plot positions
    ax3.fill_between(results.index, 0, results['position'], 
                     where=results['position'] > 0, facecolor='green', alpha=0.5, 
                     label='Long')
    ax3.fill_between(results.index, 0, results['position'], 
                     where=results['position'] < 0, facecolor='red', alpha=0.5, 
                     label='Short')
    ax3.set_ylabel('Position')
    ax3.set_xlabel('Date')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    return fig

def backtest_strategy(prices, strategy_function, strategy_params=None, 
                      transaction_cost=0.001, plot=True):
    """
    Run a backtest for a trading strategy.
    
    Parameters:
    -----------
    prices : pandas.Series
        Historical price data
    strategy_function : function
        Function that generates position signals
    strategy_params : dict, default=None
        Parameters to pass to the strategy function
    transaction_cost : float, default=0.001
        Transaction cost as a percentage
    plot : bool, default=True
        Whether to generate and display a plot
        
    Returns:
    --------
    tuple
        (results DataFrame, performance metrics dict, plot figure if plot=True)
    """
    # Apply strategy function to generate positions
    params = strategy_params if strategy_params is not None else {}
    positions = strategy_function(prices, **params)
    
    # Calculate returns and metrics
    results = calculate_returns(prices, positions, transaction_cost)
    metrics = calculate_performance_metrics(results['strategy_return'])
    
    # Generate plot if requested
    fig = None
    if plot:
        fig = plot_backtest_results(results)
    
    return results, metrics, fig if plot else None

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', periods=252)
    prices = pd.Series(np.random.randn(252).cumsum() + 100, index=dates)
    
    # Simple moving average crossover strategy
    def simple_ma_crossover(prices, short_window=50, long_window=200):
        short_ma = prices.rolling(window=short_window).mean()
        long_ma = prices.rolling(window=long_window).mean()
        signals = pd.Series(0, index=prices.index)
        signals[short_ma > long_ma] = 1
        signals[short_ma < long_ma] = -1
        return signals
    
    results, metrics, fig = backtest_strategy(
        prices, 
        simple_ma_crossover, 
        {'short_window': 20, 'long_window': 50}
    )
    
    print("Performance Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    plt.show()