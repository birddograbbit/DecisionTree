# src/backtesting/performance.py
"""
Module for evaluating trading performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set pandas options to avoid future warnings
pd.set_option('future.no_silent_downcasting', True)


def calculate_performance_metrics(equity_curve, trades):
    """
    Calculate performance metrics from equity curve and trades.
    
    Parameters:
    -----------
    equity_curve : pd.DataFrame
        Equity curve with 'equity' column
    trades : list or pd.DataFrame
        List of trades or DataFrame with trades
        
    Returns:
    --------
    dict
        Performance metrics
    """
    # Ensure equity_curve is a DataFrame
    if not isinstance(equity_curve, pd.DataFrame):
        return {}
    
    # Ensure trades is a DataFrame
    if isinstance(trades, list):
        trades_df = pd.DataFrame(trades)
    else:
        trades_df = trades
    
    # Calculate return metrics
    if len(equity_curve) < 2:
        return {}
    
    equity = equity_curve['equity']
    returns = equity.pct_change().fillna(0).infer_objects(copy=False)
    
    # Calculate daily metrics
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
    
    # Trading days per year (approximate)
    trading_days_per_year = 252
    
    # Calculate annualized metrics
    years = max(len(returns) / trading_days_per_year, 0.01)  # Minimum 0.01 to avoid division by zero
    ann_return = (1 + total_return) ** (1 / years) - 1
    ann_volatility = returns.std() * np.sqrt(trading_days_per_year) if len(returns) > 1 else 0
    
    # Calculate Sharpe ratio (assuming risk-free rate of 0.02)
    risk_free_rate = 0.02
    sharpe_ratio = (ann_return - risk_free_rate) / ann_volatility if ann_volatility != 0 else 0
    
    # Calculate drawdown
    drawdown = 1 - equity / equity.cummax()
    max_drawdown = drawdown.max()
    
    # Calculate CAGR to Max Drawdown ratio
    cagr_dd_ratio = ann_return / max_drawdown if max_drawdown != 0 else np.inf
    
    # Calculate trade metrics
    if not trades_df.empty and 'pnl' in trades_df.columns:
        num_trades = len(trades_df)
        win_rate = (trades_df['pnl'] > 0).mean()
        profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if (trades_df['pnl'] < 0).any() else np.inf
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (trades_df['pnl'] < 0).any() else 0
        avg_holding_period = trades_df['holding_days'].mean() if 'holding_days' in trades_df.columns else np.nan
    else:
        num_trades = 0
        win_rate = 0
        profit_factor = 0
        avg_win = 0
        avg_loss = 0
        avg_holding_period = np.nan
    
    # Calculate model accuracy metrics if we have predictions
    if not trades_df.empty and 'signal' in trades_df.columns and 'return' in trades_df.columns:
        # Consider signal 1 as prediction of positive return
        predictions = (trades_df['signal'] > 0).astype(int)
        actuals = (trades_df['return'] > 0).astype(int)
        
        accuracy = accuracy_score(actuals, predictions)
        try:
            precision = precision_score(actuals, predictions)
            recall = recall_score(actuals, predictions)
            f1 = f1_score(actuals, predictions)
        except:
            precision = recall = f1 = 0
    else:
        accuracy = precision = recall = f1 = np.nan
    
    # Compile metrics
    metrics = {
        'total_return': total_return,
        'ann_return': ann_return,
        'ann_volatility': ann_volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'cagr_dd_ratio': cagr_dd_ratio,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_holding_period': avg_holding_period,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics


def plot_performance(backtest_results, benchmark_data=None, save_dir=None):
    """
    Plot performance visualizations.
    
    Parameters:
    -----------
    backtest_results : dict
        Backtest results from BacktestEngine
    benchmark_data : pd.DataFrame, optional
        Benchmark price data
    save_dir : str, optional
        Directory to save plots
        
    Returns:
    --------
    dict
        Dictionary of figure objects
    """
    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import matplotlib.dates as mdates
    from matplotlib.ticker import AutoLocator
    
    figures = {}
    
    # Extract data
    equity_curve = backtest_results['equity_curve']
    trades = backtest_results['trades']
    performance = backtest_results['performance']
    
    # Create directory if it doesn't exist
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    
    # 1. Equity Curve
    fig_equity, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Convert dates to numeric for better compatibility
    date_nums = mdates.date2num(equity_curve.index.to_pydatetime())
    
    # Plot equity curve
    axs[0].plot(date_nums, equity_curve['equity'], label='Strategy')
    
    # Add benchmark if provided
    if benchmark_data is not None:
        # Calculate benchmark equity curve (assuming initial capital equals strategy's)
        initial_capital = equity_curve['equity'].iloc[0]
        
        # Ensure benchmark data has the same date range
        benchmark_subset = benchmark_data.loc[benchmark_data.index.isin(equity_curve.index)]
        
        if not benchmark_subset.empty:
            # Calculate benchmark equity
            benchmark_returns = benchmark_subset['close'].pct_change().fillna(0).infer_objects(copy=False)
            benchmark_equity = initial_capital * (1 + benchmark_returns).cumprod()
            
            # Only plot dates where benchmark data exists
            benchmark_date_nums = mdates.date2num(benchmark_equity.index.to_pydatetime())
            axs[0].plot(benchmark_date_nums, benchmark_equity.values, label='Benchmark', alpha=0.7)
    
    # Format x-axis for equity curve
    axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axs[0].xaxis.set_major_locator(AutoLocator())
    
    axs[0].set_title('Equity Curve')
    axs[0].set_ylabel('Equity ($)')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot drawdown using fill technique that avoids isfinite issues
    drawdown = 1 - equity_curve['equity'] / equity_curve['equity'].cummax()
    
    # Plot drawdown as line
    axs[1].plot(date_nums, drawdown.values * 100)
    
    # Create filled area under the line
    y1 = drawdown.values * 100
    y2 = np.zeros_like(y1)
    axs[1].fill(np.append(date_nums, date_nums[::-1]),
                np.append(y1, y2[::-1]),
                'red', alpha=0.3)
    
    # Format x-axis for drawdown
    axs[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axs[1].xaxis.set_major_locator(AutoLocator())
    
    axs[1].set_title('Drawdown (%)')
    axs[1].set_ylabel('Drawdown (%)')
    axs[1].set_xlabel('Date')
    axs[1].grid(True)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    figures['equity_curve'] = fig_equity
    
    if save_dir is not None:
        fig_equity.savefig(os.path.join(save_dir, 'equity_curve.png'))
    
    # 2. Trade Analysis
    if not trades.empty and len(trades) > 0:
        fig_trades, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot trade PnL
        axs[0, 0].bar(range(len(trades)), trades['pnl'], color=['g' if pnl > 0 else 'r' for pnl in trades['pnl']])
        axs[0, 0].set_title('Trade PnL')
        axs[0, 0].set_xlabel('Trade #')
        axs[0, 0].set_ylabel('PnL ($)')
        axs[0, 0].grid(True)
        
        # Plot trade returns
        axs[0, 1].bar(range(len(trades)), trades['return'] * 100, color=['g' if ret > 0 else 'r' for ret in trades['return']])
        axs[0, 1].set_title('Trade Returns (%)')
        axs[0, 1].set_xlabel('Trade #')
        axs[0, 1].set_ylabel('Return (%)')
        axs[0, 1].grid(True)
        
        # Plot trade holding period
        if 'holding_days' in trades.columns:
            axs[1, 0].hist(trades['holding_days'], bins=20)
            axs[1, 0].set_title('Holding Period Distribution')
            axs[1, 0].set_xlabel('Holding Days')
            axs[1, 0].set_ylabel('Frequency')
            axs[1, 0].grid(True)
        
        # Plot trade win/loss
        win_trades = trades[trades['pnl'] > 0]
        loss_trades = trades[trades['pnl'] <= 0]
        
        labels = ['Win', 'Loss']
        sizes = [len(win_trades), len(loss_trades)]
        colors = ['green', 'red']
        
        axs[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axs[1, 1].set_title('Win/Loss Ratio')
        axs[1, 1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        plt.tight_layout()
        figures['trade_analysis'] = fig_trades
        
        if save_dir is not None:
            fig_trades.savefig(os.path.join(save_dir, 'trade_analysis.png'))
    
    # 3. Performance Metrics
    fig_metrics = plt.figure(figsize=(10, 8))
    plt.axis('off')
    
    # Create table with performance metrics
    metrics_table = [
        ["Total Return", f"{performance['total_return']:.2%}"],
        ["Annualized Return", f"{performance['ann_return']:.2%}"],
        ["Annualized Volatility", f"{performance['ann_volatility']:.2%}"],
        ["Sharpe Ratio", f"{performance['sharpe_ratio']:.2f}"],
        ["Maximum Drawdown", f"{performance['max_drawdown']:.2%}"],
        ["CAGR/Max DD Ratio", f"{performance['cagr_dd_ratio']:.2f}"],
        ["Number of Trades", f"{performance['num_trades']}"],
        ["Win Rate", f"{performance['win_rate']:.2%}"],
        ["Profit Factor", f"{performance['profit_factor']:.2f}"],
        ["Average Win", f"${performance['avg_win']:.2f}"],
        ["Average Loss", f"${performance['avg_loss']:.2f}"]
    ]
    
    # Add average holding period if available
    if 'avg_holding_period' in performance and not np.isnan(performance['avg_holding_period']):
        metrics_table.append(["Average Holding Period", f"{performance['avg_holding_period']:.1f} days"])
    
    plt.table(cellText=metrics_table, colWidths=[0.6, 0.4], loc='center', cellLoc='left')
    plt.title('Performance Metrics', fontsize=16, pad=20)
    
    figures['metrics'] = fig_metrics
    
    if save_dir is not None:
        fig_metrics.savefig(os.path.join(save_dir, 'performance_metrics.png'))
    
    # 4. Monthly Returns Heatmap (if enough data)
    if len(equity_curve) >= 60:  # At least 60 days of data
        try:
            # Calculate daily returns
            daily_returns = equity_curve['equity'].pct_change().fillna(0).infer_objects(copy=False)
            
            # Convert to monthly returns - using 'ME' instead of deprecated 'M'
            monthly_returns = daily_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
            
            # Reshape to heatmap format
            monthly_returns_matrix = []
            years = sorted(set(monthly_returns.index.year))
            months = range(1, 13)
            
            for year in years:
                year_returns = []
                for month in months:
                    try:
                        # Find the return for this month and year
                        month_return = monthly_returns.loc[(monthly_returns.index.year == year) & 
                                                         (monthly_returns.index.month == month)]
                        if not month_return.empty:
                            year_returns.append(month_return.iloc[0])
                        else:
                            year_returns.append(np.nan)
                    except:
                        year_returns.append(np.nan)
                monthly_returns_matrix.append(year_returns)
            
            # Create heatmap
            fig_heatmap, ax = plt.subplots(figsize=(12, len(years) * 0.6 + 2))
            
            # Create heatmap
            heatmap = ax.imshow(monthly_returns_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.1)
            
            # Add colorbar
            cbar = plt.colorbar(heatmap, ax=ax)
            cbar.set_label('Monthly Return (%)', rotation=270, labelpad=20)
            
            # Configure axis
            ax.set_xticks(np.arange(len(months)))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            ax.set_yticks(np.arange(len(years)))
            ax.set_yticklabels(years)
            
            # Add return values in cells
            for i in range(len(years)):
                for j in range(len(months)):
                    try:
                        value = monthly_returns_matrix[i][j]
                        if not np.isnan(value):
                            text = ax.text(j, i, f"{value:.1%}",
                                         ha="center", va="center", color="black" if abs(value) < 0.05 else "white")
                    except:
                        pass
            
            ax.set_title('Monthly Returns (%)')
            plt.tight_layout()
            
            figures['monthly_returns'] = fig_heatmap
            
            if save_dir is not None:
                fig_heatmap.savefig(os.path.join(save_dir, 'monthly_returns.png'))
        except Exception as e:
            print(f"Error creating monthly returns heatmap: {e}")
    
    return figures

def compare_to_benchmark(backtest_results, benchmark_data):
    """
    Compare strategy performance to benchmark.
    
    Parameters:
    -----------
    backtest_results : dict
        Backtest results from BacktestEngine
    benchmark_data : pd.DataFrame
        Benchmark price data with 'close' column
        
    Returns:
    --------
    dict
        Performance comparison metrics
    """
    # Extract strategy equity curve
    equity_curve = backtest_results['equity_curve']
    strategy_equity = equity_curve['equity']
    
    # Calculate benchmark equity curve (assuming initial capital equals strategy's)
    initial_capital = strategy_equity.iloc[0]
    
    # Align benchmark data with strategy dates
    common_dates = sorted(set(strategy_equity.index) & set(benchmark_data.index))
    strategy_equity = strategy_equity.loc[common_dates]
    benchmark_close = benchmark_data.loc[common_dates, 'close']
    
    # Calculate benchmark returns and equity
    benchmark_returns = benchmark_close.pct_change().fillna(0)
    benchmark_equity = initial_capital * (1 + benchmark_returns).cumprod()
    
    # Calculate comparative metrics
    strategy_return = (strategy_equity.iloc[-1] / strategy_equity.iloc[0]) - 1
    benchmark_return = (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) - 1
    
    strategy_drawdown = 1 - strategy_equity / strategy_equity.cummax()
    benchmark_drawdown = 1 - benchmark_equity / benchmark_equity.cummax()
    
    strategy_max_dd = strategy_drawdown.max()
    benchmark_max_dd = benchmark_drawdown.max()
    
    # Calculate annualized metrics (assuming 252 trading days per year)
    years = len(strategy_equity) / 252
    strategy_cagr = (1 + strategy_return) ** (1 / years) - 1
    benchmark_cagr = (1 + benchmark_return) ** (1 / years) - 1
    
    strategy_cagr_dd = strategy_cagr / strategy_max_dd if strategy_max_dd != 0 else np.inf
    benchmark_cagr_dd = benchmark_cagr / benchmark_max_dd if benchmark_max_dd != 0 else np.inf
    
    # Calculate alpha and beta
    strategy_daily_returns = strategy_equity.pct_change().fillna(0).infer_objects(copy=False)
    benchmark_daily_returns = benchmark_equity.pct_change().fillna(0).infer_objects(copy=False)
    
    # Beta calculation (regression slope)
    covariance = np.cov(strategy_daily_returns, benchmark_daily_returns)[0, 1]
    variance = np.var(benchmark_daily_returns)
    beta = covariance / variance if variance != 0 else 0
    
    # Alpha calculation (annualized)
    alpha = strategy_cagr - (0.02 + beta * (benchmark_cagr - 0.02))  # Assuming risk-free rate of 2%
    
    # Create comparison dictionary
    comparison = {
        'strategy_return': strategy_return,
        'benchmark_return': benchmark_return,
        'strategy_cagr': strategy_cagr,
        'benchmark_cagr': benchmark_cagr,
        'strategy_max_dd': strategy_max_dd,
        'benchmark_max_dd': benchmark_max_dd,
        'strategy_cagr_dd': strategy_cagr_dd,
        'benchmark_cagr_dd': benchmark_cagr_dd,
        'alpha': alpha,
        'beta': beta,
        'outperformance': strategy_return - benchmark_return
    }
    
    return comparison