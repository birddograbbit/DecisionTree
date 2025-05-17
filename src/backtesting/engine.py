"""
Backtesting engine for trading strategies.
"""

import pandas as pd
import numpy as np

class BacktestEngine:
    """
    Engine for backtesting trading strategies.
    
    This class handles the simulation of trading signals against historical
    price data to evaluate strategy performance.
    """

    def __init__(self, initial_capital=100000.0, commission=0.001, slippage=0.001):
        """
        Initialize backtesting engine.
        
        Parameters:
        -----------
        initial_capital : float, default=100000.0
            Initial capital for backtesting
        commission : float, default=0.001
            Commission rate per trade (e.g., 0.001 = 0.1%)
        slippage : float, default=0.001
            Slippage rate per trade (e.g., 0.001 = 0.1%)
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
        data : dict
            Dictionary mapping symbols to price DataFrames
            
        Returns:
        --------
        dict
            Backtest results including trades, equity curve, and performance metrics
        """
        self.reset()
        
        # Sort signals by date
        signals = signals.sort_values('date')
        
        # Initialize equity curve
        equity_dates = sorted(signals['date'].unique())
        self.equity_curve = pd.DataFrame(index=equity_dates, columns=['equity'])
        self.equity_curve.loc[equity_dates[0], 'equity'] = self.initial_capital
        
        # Iterate through signals
        for _, row in signals.iterrows():
            date = row['date']
            symbol = row['symbol']
            signal = row['signal']
            
            # Get price data for the symbol
            if symbol not in data:
                continue
            
            price_data = data[symbol]
            if date not in price_data.index:
                continue
                
            price = price_data.loc[date, 'close']
            
            # Process signal
            if signal == 1 and symbol not in self.positions:  # Buy
                # Calculate position size
                position_size = row.get('position_size', 1.0)
                available_capital = self.capital * 0.95  # Keep some cash
                position_value = available_capital * position_size
                shares = position_value / price
                cost = shares * price * (1 + self.slippage) * (1 + self.commission)
                
                if cost <= self.capital:
                    self.positions[symbol] = {
                        'shares': shares,
                        'entry_price': price,
                        'entry_date': date,
                        'position_size': position_size
                    }
                    self.capital -= cost
                    
            elif signal == -1 and symbol in self.positions:  # Sell
                position = self.positions[symbol]
                shares = position['shares']
                entry_price = position['entry_price']
                entry_date = position['entry_date']
                
                # Calculate proceeds
                proceeds = shares * price * (1 - self.slippage) * (1 - self.commission)
                self.capital += proceeds
                
                # Record trade
                pnl = proceeds - (shares * entry_price)
                ret = (price / entry_price) - 1
                
                self.trades.append({
                    'symbol': symbol,
                    'entry_date': entry_date,
                    'entry_price': entry_price,
                    'exit_date': date,
                    'exit_price': price,
                    'shares': shares,
                    'pnl': pnl,
                    'return': ret,
                    'position_size': position['position_size']
                })
                
                # Remove position
                del self.positions[symbol]
            
            # Update equity curve
            total_position_value = sum(
                data[s].loc[date, 'close'] * pos['shares'] 
                for s, pos in self.positions.items() 
                if date in data[s].index
            )
            self.equity_curve.loc[date, 'equity'] = self.capital + total_position_value
        
        # Close any remaining positions at the last date
        last_date = equity_dates[-1]
        for symbol, position in list(self.positions.items()):
            if symbol not in data or last_date not in data[symbol].index:
                continue
                
            shares = position['shares']
            entry_price = position['entry_price']
            entry_date = position['entry_date']
            exit_price = data[symbol].loc[last_date, 'close']
            
            # Calculate proceeds
            proceeds = shares * exit_price * (1 - self.slippage) * (1 - self.commission)
            self.capital += proceeds
            
            # Record trade
            pnl = proceeds - (shares * entry_price)
            ret = (exit_price / entry_price) - 1
            
            self.trades.append({
                'symbol': symbol,
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': last_date,
                'exit_price': exit_price,
                'shares': shares,
                'pnl': pnl,
                'return': ret,
                'position_size': position['position_size']
            })
            
            # Remove position
            del self.positions[symbol]
        
        # Update final equity
        self.equity_curve.loc[last_date, 'equity'] = self.capital
        
        # Calculate performance metrics
        performance = self.calculate_performance()
        
        # Prepare results
        results = {
            'trades': pd.DataFrame(self.trades) if self.trades else pd.DataFrame(),
            'equity_curve': self.equity_curve,
            'performance': performance
        }
        
        return results
    
    def calculate_performance(self):
        """
        Calculate performance metrics.
        
        Returns:
        --------
        dict
            Performance metrics
        """
        equity = self.equity_curve['equity']
        
        # Calculate returns
        returns = equity.pct_change().dropna()
        
        # Calculate metrics
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        
        # Annualized metrics (assuming 252 trading days per year)
        trading_days = len(returns)
        years = trading_days / 252
        
        if years > 0:
            ann_return = (1 + total_return) ** (1 / years) - 1
        else:
            ann_return = total_return
        
        ann_volatility = returns.std() * np.sqrt(252)
        
        # Risk-adjusted metrics
        sharpe_ratio = ann_return / ann_volatility if ann_volatility != 0 else 0
        
        # Drawdown analysis
        drawdown = 1 - equity / equity.cummax()
        max_drawdown = drawdown.max()
        
        # CAGR to Max Drawdown ratio
        cagr_dd_ratio = ann_return / max_drawdown if max_drawdown != 0 else np.inf
        
        # Trade statistics
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        if not trades_df.empty:
            win_rate = (trades_df['pnl'] > 0).mean()
            profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / 
                               trades_df[trades_df['pnl'] <= 0]['pnl'].sum()) if trades_df[trades_df['pnl'] <= 0]['pnl'].sum() != 0 else np.inf
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] <= 0]) > 0 else 0
            avg_return = trades_df['return'].mean()
            avg_trade = trades_df['pnl'].mean()
        else:
            win_rate = 0
            profit_factor = 0
            avg_win = 0
            avg_loss = 0
            avg_return = 0
            avg_trade = 0
        
        return {
            'total_return': total_return,
            'ann_return': ann_return,
            'ann_volatility': ann_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'cagr_dd_ratio': cagr_dd_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_return': avg_return,
            'avg_trade': avg_trade,
            'num_trades': len(trades_df)
        }