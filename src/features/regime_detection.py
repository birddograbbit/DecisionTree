"""
Market regime detection module.

This module provides functionality to detect different market regimes
(e.g., trending, ranging, high/low volatility) to adapt trading strategies
to changing market conditions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


class RegimeDetector:
    """
    Market regime detection class.
    
    This class implements various methods for detecting market regimes
    based on price action, volatility, and other market characteristics.
    """
    
    def __init__(self, method='trend_volatility', **params):
        """
        Initialize regime detector.
        
        Parameters:
        -----------
        method : str, default='trend_volatility'
            Method for regime detection. Options:
            - 'trend_volatility': Combined trend and volatility regime
            - 'ma_crossover': Moving average crossover for trend
            - 'volatility_regimes': Volatility-based regimes
            - 'statistical': Statistical regime detection
        params : dict
            Additional parameters for the chosen method
        """
        self.method = method
        self.params = params
        self.regime_history = None
        
    def detect_regime(self, data):
        """
        Detect market regime from price data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data with at least 'close' column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with regime information
        """
        if self.method == 'trend_volatility':
            return self._detect_trend_volatility_regime(data)
        elif self.method == 'ma_crossover':
            return self._detect_ma_crossover_regime(data)
        elif self.method == 'volatility_regimes':
            return self._detect_volatility_regime(data)
        elif self.method == 'statistical':
            return self._detect_statistical_regime(data)
        else:
            raise ValueError(f"Unknown regime detection method: {self.method}")
    
    def _detect_trend_volatility_regime(self, data):
        """
        Detect regime based on trend and volatility.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with regime information
        """
        # Get parameters
        fast_window = self.params.get('fast_window', 20)
        slow_window = self.params.get('slow_window', 50)
        vol_window = self.params.get('vol_window', 20)
        vol_threshold = self.params.get('vol_threshold', 0.75)
        
        # Ensure data has the required columns
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        # Calculate moving averages for trend
        result['fast_ma'] = result['close'].rolling(window=fast_window).mean()
        result['slow_ma'] = result['close'].rolling(window=slow_window).mean()
        
        # Determine trend
        result['trend'] = 0  # 1 for uptrend, -1 for downtrend, 0 for no clear trend
        result.loc[result['fast_ma'] > result['slow_ma'], 'trend'] = 1
        result.loc[result['fast_ma'] < result['slow_ma'], 'trend'] = -1
        
        # Calculate returns for volatility
        result['returns'] = result['close'].pct_change()
        
        # Calculate historical volatility
        result['volatility'] = result['returns'].rolling(window=vol_window).std() * np.sqrt(252)  # Annualized
        
        # Calculate volatility percentile
        result['vol_rank'] = result['volatility'].rolling(window=252).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        
        # Determine volatility regime
        result['vol_regime'] = 0  # 1 for high volatility, 0 for normal, -1 for low volatility
        result.loc[result['vol_rank'] > vol_threshold, 'vol_regime'] = 1
        result.loc[result['vol_rank'] < (1 - vol_threshold), 'vol_regime'] = -1
        
        # Combined regime classification
        result['regime'] = result['trend'] * 10 + result['vol_regime']
        
        # Assign regime labels
        regime_labels = {
            11: 'strong_uptrend',    # Uptrend, high volatility
            10: 'uptrend',           # Uptrend, normal volatility
            9: 'weak_uptrend',       # Uptrend, low volatility
            1: 'volatile_neutral',   # No trend, high volatility
            0: 'neutral',            # No trend, normal volatility
            -1: 'low_vol_neutral',   # No trend, low volatility
            -9: 'weak_downtrend',    # Downtrend, low volatility
            -10: 'downtrend',        # Downtrend, normal volatility
            -11: 'strong_downtrend'  # Downtrend, high volatility
        }
        
        result['regime_label'] = result['regime'].map(regime_labels)
        
        # Store regime history
        self.regime_history = result[['close', 'fast_ma', 'slow_ma', 'volatility', 'vol_rank', 'trend', 'vol_regime', 'regime', 'regime_label']]
        
        return self.regime_history
    
    def _detect_ma_crossover_regime(self, data):
        """
        Detect regime based on moving average crossovers.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with regime information
        """
        # Get parameters
        short_window = self.params.get('short_window', 10)
        medium_window = self.params.get('medium_window', 50)
        long_window = self.params.get('long_window', 200)
        
        # Ensure data has the required columns
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        # Calculate moving averages
        result['short_ma'] = result['close'].rolling(window=short_window).mean()
        result['medium_ma'] = result['close'].rolling(window=medium_window).mean()
        result['long_ma'] = result['close'].rolling(window=long_window).mean()
        
        # Define conditions for different regimes
        result['regime'] = 0  # Default is neutral
        
        # Strong uptrend: short > medium > long
        strong_uptrend = (result['short_ma'] > result['medium_ma']) & (result['medium_ma'] > result['long_ma'])
        
        # Weak uptrend: short > medium, medium <= long
        weak_uptrend = (result['short_ma'] > result['medium_ma']) & (result['medium_ma'] <= result['long_ma'])
        
        # Strong downtrend: short < medium < long
        strong_downtrend = (result['short_ma'] < result['medium_ma']) & (result['medium_ma'] < result['long_ma'])
        
        # Weak downtrend: short < medium, medium >= long
        weak_downtrend = (result['short_ma'] < result['medium_ma']) & (result['medium_ma'] >= result['long_ma'])
        
        # Assign regime values
        result.loc[strong_uptrend, 'regime'] = 2   # Strong uptrend
        result.loc[weak_uptrend, 'regime'] = 1     # Weak uptrend
        result.loc[weak_downtrend, 'regime'] = -1  # Weak downtrend
        result.loc[strong_downtrend, 'regime'] = -2  # Strong downtrend
        
        # Assign regime labels
        regime_labels = {
            2: 'strong_uptrend',
            1: 'weak_uptrend',
            0: 'neutral',
            -1: 'weak_downtrend',
            -2: 'strong_downtrend'
        }
        
        result['regime_label'] = result['regime'].map(regime_labels)
        
        # Store regime history
        self.regime_history = result[['close', 'short_ma', 'medium_ma', 'long_ma', 'regime', 'regime_label']]
        
        return self.regime_history
    
    def _detect_volatility_regime(self, data):
        """
        Detect regime based on volatility levels.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with regime information
        """
        # Get parameters
        vol_window = self.params.get('vol_window', 20)
        high_vol_threshold = self.params.get('high_vol_threshold', 0.8)
        low_vol_threshold = self.params.get('low_vol_threshold', 0.2)
        lookback = self.params.get('lookback', 252)  # Lookback for percentile calculation
        
        # Ensure data has the required columns
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        # Calculate returns
        result['returns'] = result['close'].pct_change()
        
        # Calculate volatility (annualized standard deviation of returns)
        result['volatility'] = result['returns'].rolling(window=vol_window).std() * np.sqrt(252)
        
        # Calculate ATR if OHLC data is available
        if all(col in data.columns for col in ['open', 'high', 'low']):
            result['tr1'] = abs(result['high'] - result['low'])
            result['tr2'] = abs(result['high'] - result['close'].shift())
            result['tr3'] = abs(result['low'] - result['close'].shift())
            result['true_range'] = result[['tr1', 'tr2', 'tr3']].max(axis=1)
            result['atr'] = result['true_range'].rolling(window=vol_window).mean()
            result['atr_pct'] = result['atr'] / result['close']
            
            # Calculate ATR percentile
            result['atr_rank'] = result['atr_pct'].rolling(window=lookback).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        
        # Calculate volatility percentile
        result['vol_rank'] = result['volatility'].rolling(window=lookback).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        
        # Determine volatility regime
        result['regime'] = 0  # 1 for high volatility, 0 for normal, -1 for low volatility
        result.loc[result['vol_rank'] > high_vol_threshold, 'regime'] = 1
        result.loc[result['vol_rank'] < low_vol_threshold, 'regime'] = -1
        
        # Assign regime labels
        regime_labels = {
            1: 'high_volatility',
            0: 'normal_volatility',
            -1: 'low_volatility'
        }
        
        result['regime_label'] = result['regime'].map(regime_labels)
        
        # Select columns for regime history
        cols = ['close', 'volatility', 'vol_rank', 'regime', 'regime_label']
        if 'atr_rank' in result.columns:
            cols.insert(3, 'atr_rank')
            cols.insert(2, 'atr')
        
        # Store regime history
        self.regime_history = result[cols]
        
        return self.regime_history
    
    def _detect_statistical_regime(self, data):
        """
        Detect regime using statistical methods.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with regime information
        """
        # Get parameters
        ma_window = self.params.get('ma_window', 50)
        std_window = self.params.get('std_window', 50)
        z_threshold = self.params.get('z_threshold', 1.0)
        
        # Ensure data has the required columns
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        # Make a copy to avoid modifying original data
        result = data.copy()
        
        # Calculate moving average
        result['ma'] = result['close'].rolling(window=ma_window).mean()
        
        # Calculate deviation from MA
        result['deviation'] = result['close'] - result['ma']
        
        # Calculate rolling standard deviation of deviations
        result['deviation_std'] = result['deviation'].rolling(window=std_window).std()
        
        # Calculate z-score
        result['z_score'] = result['deviation'] / result['deviation_std']
        
        # Determine regime based on z-score
        result['regime'] = 0  # 0 for mean-reverting
        result.loc[result['z_score'] > z_threshold, 'regime'] = 1  # 1 for overbought
        result.loc[result['z_score'] < -z_threshold, 'regime'] = -1  # -1 for oversold
        
        # Calculate slope of moving average (trend direction)
        result['ma_slope'] = result['ma'].diff(20) / 20
        
        # Normalize slope
        result['ma_slope_norm'] = result['ma_slope'] / result['ma'] * 100  # Percentage slope
        
        # Determine trend based on slope
        result['trend'] = 0  # 0 for no significant trend
        slope_threshold = self.params.get('slope_threshold', 0.1)  # % per day
        result.loc[result['ma_slope_norm'] > slope_threshold, 'trend'] = 1  # 1 for uptrend
        result.loc[result['ma_slope_norm'] < -slope_threshold, 'trend'] = -1  # -1 for downtrend
        
        # Combined regime (trend and mean-reversion)
        result['combined_regime'] = result['trend'] * 10 + result['regime']
        
        # Assign regime labels
        regime_labels = {
            11: 'uptrend_overbought',
            10: 'uptrend_neutral',
            9: 'uptrend_oversold',
            1: 'sideways_overbought',
            0: 'sideways_neutral',
            -1: 'sideways_oversold',
            -9: 'downtrend_oversold',
            -10: 'downtrend_neutral',
            -11: 'downtrend_overbought'
        }
        
        result['regime_label'] = result['combined_regime'].map(regime_labels)
        
        # Store regime history
        self.regime_history = result[['close', 'ma', 'deviation', 'z_score', 'ma_slope_norm', 'regime', 'trend', 'combined_regime', 'regime_label']]
        
        return self.regime_history
    
    def plot_regimes(self, price_data=None, figsize=(15, 10)):
        """
        Plot price chart with highlighted regimes.
        
        Parameters:
        -----------
        price_data : pd.DataFrame, optional
            Price data with 'close' column. If None, uses stored regime history.
        figsize : tuple, default=(15, 10)
            Figure size for the plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.regime_history is None:
            if price_data is None:
                raise ValueError("No regime history available. Run detect_regime first or provide price_data.")
            self.detect_regime(price_data)
        
        # Create a copy of regime history for plotting
        df = self.regime_history.copy()
        
        # Create figure and axes
        fig, axes = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price and moving averages on the first axis
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Close Price')
        
        # Plot different moving averages based on the method
        if self.method == 'trend_volatility':
            ax1.plot(df.index, df['fast_ma'], label=f"Fast MA ({self.params.get('fast_window', 20)})")
            ax1.plot(df.index, df['slow_ma'], label=f"Slow MA ({self.params.get('slow_window', 50)})")
        elif self.method == 'ma_crossover':
            ax1.plot(df.index, df['short_ma'], label=f"Short MA ({self.params.get('short_window', 10)})")
            ax1.plot(df.index, df['medium_ma'], label=f"Medium MA ({self.params.get('medium_window', 50)})")
            ax1.plot(df.index, df['long_ma'], label=f"Long MA ({self.params.get('long_window', 200)})")
        elif self.method == 'statistical':
            ax1.plot(df.index, df['ma'], label=f"MA ({self.params.get('ma_window', 50)})")
        
        # Plot regions with different regimes
        if 'regime_label' in df.columns:
            labels = df['regime_label'].unique()
            
            # Define color mapping based on regime types
            regime_colors = {}
            
            # For trend_volatility method
            if self.method == 'trend_volatility':
                regime_colors = {
                    'strong_uptrend': 'darkgreen',
                    'uptrend': 'green',
                    'weak_uptrend': 'lightgreen',
                    'volatile_neutral': 'orange',
                    'neutral': 'gray',
                    'low_vol_neutral': 'lightgray',
                    'weak_downtrend': 'pink',
                    'downtrend': 'red',
                    'strong_downtrend': 'darkred'
                }
            # For ma_crossover method
            elif self.method == 'ma_crossover':
                regime_colors = {
                    'strong_uptrend': 'darkgreen',
                    'weak_uptrend': 'lightgreen',
                    'neutral': 'gray',
                    'weak_downtrend': 'pink',
                    'strong_downtrend': 'darkred'
                }
            # For volatility_regimes method
            elif self.method == 'volatility_regimes':
                regime_colors = {
                    'high_volatility': 'red',
                    'normal_volatility': 'gray',
                    'low_volatility': 'green'
                }
            # For statistical method
            elif self.method == 'statistical':
                regime_colors = {
                    'uptrend_overbought': 'darkgreen',
                    'uptrend_neutral': 'green',
                    'uptrend_oversold': 'lightgreen',
                    'sideways_overbought': 'orange',
                    'sideways_neutral': 'gray',
                    'sideways_oversold': 'lightblue',
                    'downtrend_overbought': 'darkred',
                    'downtrend_neutral': 'red',
                    'downtrend_oversold': 'pink'
                }
            
            # Use a default color for any missing labels
            for label in labels:
                if label not in regime_colors and label is not None:
                    regime_colors[label] = 'gray'
            
            # Color the background of the price chart based on regime
            for label in labels:
                if label is None:
                    continue
                
                mask = df['regime_label'] == label
                if mask.any():
                    # Find continuous segments of the same regime
                    regions = []
                    start_idx = None
                    
                    for i, val in enumerate(mask):
                        if val and start_idx is None:
                            start_idx = i
                        elif not val and start_idx is not None:
                            regions.append((start_idx, i))
                            start_idx = None
                    
                    # Handle case where last region extends to end of data
                    if start_idx is not None:
                        regions.append((start_idx, len(mask)))
                    
                    # Shade each region
                    for start, end in regions:
                        if end > start:
                            ax1.axvspan(df.index[start], df.index[end-1], 
                                       alpha=0.2, color=regime_colors.get(label, 'gray'))
        
        # Set up legend for the price chart
        ax1.set_title('Price with Market Regimes')
        ax1.set_ylabel('Price')
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Plot regime indicator on the second axis
        ax2 = axes[1]
        
        # Determine what to plot based on the method
        if self.method == 'trend_volatility':
            # Create a colormap for regime
            scatter = ax2.scatter(df.index, df['trend'], c=df['vol_regime'], 
                                 cmap='coolwarm', marker='o', s=30, alpha=0.7)
            ax2.set_ylim(-1.5, 1.5)
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels(['Downtrend', 'Neutral', 'Uptrend'])
            colorbar = plt.colorbar(scatter, ax=ax2)
            colorbar.set_label('Volatility Regime')
            colorbar.set_ticks([-1, 0, 1])
            colorbar.set_ticklabels(['Low', 'Normal', 'High'])
            
        elif self.method == 'ma_crossover':
            # Plot regime directly
            ax2.plot(df.index, df['regime'], marker='o', markersize=3, linewidth=1)
            ax2.set_ylim(-2.5, 2.5)
            ax2.set_yticks([-2, -1, 0, 1, 2])
            ax2.set_yticklabels(['Strong Downtrend', 'Weak Downtrend', 'Neutral', 
                                'Weak Uptrend', 'Strong Uptrend'])
            
        elif self.method == 'volatility_regimes':
            # Plot volatility and its rank
            ax2_vol = ax2
            ax2_vol.plot(df.index, df['volatility'], color='blue', label='Volatility')
            ax2_vol.set_ylabel('Volatility', color='blue')
            ax2_vol.tick_params(axis='y', labelcolor='blue')
            
            ax2_rank = ax2.twinx()
            ax2_rank.plot(df.index, df['vol_rank'], color='red', label='Vol Rank')
            ax2_rank.set_ylabel('Volatility Rank', color='red')
            ax2_rank.tick_params(axis='y', labelcolor='red')
            ax2_rank.set_ylim(0, 1)
            
            # Add horizontal lines at thresholds
            high_thresh = self.params.get('high_vol_threshold', 0.8)
            low_thresh = self.params.get('low_vol_threshold', 0.2)
            ax2_rank.axhline(y=high_thresh, color='gray', linestyle='--', alpha=0.5)
            ax2_rank.axhline(y=low_thresh, color='gray', linestyle='--', alpha=0.5)
            
            # Combine legends
            lines1, labels1 = ax2_vol.get_legend_handles_labels()
            lines2, labels2 = ax2_rank.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
        elif self.method == 'statistical':
            # Plot z-score and MA slope
            ax2_z = ax2
            ax2_z.plot(df.index, df['z_score'], color='blue', label='Z-Score')
            ax2_z.set_ylabel('Z-Score', color='blue')
            ax2_z.tick_params(axis='y', labelcolor='blue')
            ax2_z.axhline(y=self.params.get('z_threshold', 1.0), color='gray', linestyle='--', alpha=0.5)
            ax2_z.axhline(y=-self.params.get('z_threshold', 1.0), color='gray', linestyle='--', alpha=0.5)
            
            ax2_slope = ax2.twinx()
            ax2_slope.plot(df.index, df['ma_slope_norm'], color='red', label='MA Slope')
            ax2_slope.set_ylabel('MA Slope (%)', color='red')
            ax2_slope.tick_params(axis='y', labelcolor='red')
            
            # Add horizontal lines at thresholds
            slope_thresh = self.params.get('slope_threshold', 0.1)
            ax2_slope.axhline(y=slope_thresh, color='gray', linestyle='--', alpha=0.5)
            ax2_slope.axhline(y=-slope_thresh, color='gray', linestyle='--', alpha=0.5)
            
            # Combine legends
            lines1, labels1 = ax2_z.get_legend_handles_labels()
            lines2, labels2 = ax2_slope.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        ax2.set_title('Market Regime Indicators')
        ax2.grid(True)
        
        # Create legend patches for regimes
        regime_patches = []
        for label, color in regime_colors.items():
            if label in df['regime_label'].unique():
                patch = mpatches.Patch(color=color, alpha=0.2, label=label)
                regime_patches.append(patch)
        
        # Add a second legend for regime colors
        if regime_patches:
            ax1.legend(handles=regime_patches, loc='upper right', 
                      title='Market Regimes', bbox_to_anchor=(1.15, 1))
        
        plt.tight_layout()
        
        return fig
    
    def get_current_regime(self):
        """
        Get the current market regime.
        
        Returns:
        --------
        dict
            Information about the current regime
        """
        if self.regime_history is None or len(self.regime_history) == 0:
            raise ValueError("No regime history available. Run detect_regime first.")
        
        # Get the latest row from regime history
        last_row = self.regime_history.iloc[-1]
        
        # Create a dictionary with regime information
        regime_info = {
            'date': last_row.name,
            'close': last_row['close'],
            'regime_label': last_row.get('regime_label', None)
        }
        
        # Add method-specific information
        if self.method == 'trend_volatility':
            regime_info.update({
                'trend': last_row['trend'],
                'volatility_regime': last_row['vol_regime'],
                'volatility': last_row['volatility'],
                'volatility_rank': last_row['vol_rank']
            })
        elif self.method == 'ma_crossover':
            regime_info.update({
                'regime_value': last_row['regime']
            })
        elif self.method == 'volatility_regimes':
            regime_info.update({
                'volatility': last_row['volatility'],
                'volatility_rank': last_row['vol_rank'],
                'regime_value': last_row['regime']
            })
            if 'atr' in last_row:
                regime_info.update({
                    'atr': last_row['atr'],
                    'atr_rank': last_row['atr_rank']
                })
        elif self.method == 'statistical':
            regime_info.update({
                'z_score': last_row['z_score'],
                'ma_slope': last_row['ma_slope_norm'],
                'trend': last_row['trend'],
                'regime_value': last_row['regime'],
                'combined_regime': last_row['combined_regime']
            })
        
        return regime_info
    
    def get_regime_stats(self):
        """
        Get statistics about different regimes.
        
        Returns:
        --------
        pd.DataFrame
            Statistics for each regime
        """
        if self.regime_history is None or len(self.regime_history) == 0:
            raise ValueError("No regime history available. Run detect_regime first.")
        
        # Create a copy of regime history
        df = self.regime_history.copy()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        
        # Calculate statistics for each regime
        regime_stats = df.groupby('regime_label')['returns'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('median', 'median'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('positive_pct', lambda x: (x > 0).mean())
        ])
        
        # Annualize returns and risk
        trading_days = 252
        regime_stats['ann_return'] = regime_stats['mean'] * trading_days
        regime_stats['ann_risk'] = regime_stats['std'] * np.sqrt(trading_days)
        regime_stats['sharpe'] = regime_stats['ann_return'] / regime_stats['ann_risk']
        
        # Calculate percentage of time in each regime
        regime_stats['time_pct'] = regime_stats['count'] / len(df)
        
        # Order columns
        cols = ['count', 'time_pct', 'mean', 'median', 'ann_return', 'ann_risk', 'sharpe', 'positive_pct', 'min', 'max', 'std']
        regime_stats = regime_stats[cols]
        
        return regime_stats
