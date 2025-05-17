"""
Regime detection module for identifying market regimes.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class RegimeDetector:
    """
    Detects market regimes based on price and volume patterns.
    
    This class provides functionality to identify distinct market regimes
    (e.g., bull, bear, sideways, volatile) based on various market indicators.
    """
    
    def __init__(self, method='trend_volatility', n_regimes=4, lookback=20):
        """
        Initialize the regime detector.
        
        Parameters:
        -----------
        method : str, default='trend_volatility'
            Regime detection method to use
            Options: 'trend_volatility', 'kmeans', 'hmm'
        n_regimes : int, default=4
            Number of regimes to detect
        lookback : int, default=20
            Lookback period for calculating indicators
        """
        self.method = method
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.regime_history = None
        self.regime_stats = None
        self.model = None
    
    def detect_regime(self, data):
        """
        Detect market regimes in the given data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data with columns: open, high, low, close, volume
            
        Returns:
        --------
        pd.DataFrame
            Data with regime information
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        
        if self.method == 'trend_volatility':
            regimes = self._detect_trend_volatility(data)
        elif self.method == 'kmeans':
            regimes = self._detect_kmeans(data)
        elif self.method == 'hmm':
            regimes = self._detect_hmm(data)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Store regime history for later analysis
        self.regime_history = regimes
        
        # Calculate regime statistics
        self._calculate_regime_stats(data, regimes)
        
        return regimes
    
    def _detect_trend_volatility(self, data):
        """
        Detect regimes based on trend and volatility.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data with columns: open, high, low, close, volume
            
        Returns:
        --------
        pd.DataFrame
            Data with regime information
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate trend indicators
        df['sma_short'] = df['close'].rolling(window=self.lookback//2).mean()
        df['sma_long'] = df['close'].rolling(window=self.lookback).mean()
        df['trend'] = df['sma_short'] / df['sma_long'] - 1
        
        # Calculate volatility
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=self.lookback).std()
        
        # Classify trend
        df['trend_regime'] = pd.cut(
            df['trend'],
            bins=[-float('inf'), -0.01, 0.01, float('inf')],
            labels=[0, 1, 2]
        )
        
        # Classify volatility
        volatility_median = df['volatility'].median()
        df['vol_regime'] = (df['volatility'] > volatility_median).astype(int)
        
        # Combine trend and volatility regimes
        df['regime'] = df['trend_regime'] * 2 + df['vol_regime']
        
        # Map regime codes to labels
        regime_labels = {
            0: 'bear_low_vol',
            1: 'bear_high_vol',
            2: 'neutral_low_vol',
            3: 'neutral_high_vol',
            4: 'bull_low_vol',
            5: 'bull_high_vol'
        }
        df['regime_label'] = df['regime'].map(regime_labels)
        
        # Keep only the relevant columns for output
        regime_df = df[['regime', 'regime_label', 'trend', 'volatility', 'trend_regime', 'vol_regime']].copy()
        
        return regime_df
    
    def _detect_kmeans(self, data):
        """
        Detect regimes using K-means clustering.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data with columns: open, high, low, close, volume
            
        Returns:
        --------
        pd.DataFrame
            Data with regime information
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate features for clustering
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=self.lookback).std()
        df['log_volume'] = np.log(df['volume'])
        df['volume_ma'] = df['log_volume'].rolling(window=self.lookback).mean()
        df['volume_ratio'] = df['log_volume'] / df['volume_ma']
        
        # Calculate additional features
        df['rsi'] = 100 - (100 / (1 + (df['returns'].rolling(window=14).apply(
            lambda x: sum(x[x > 0]) / abs(sum(x[x < 0])) if sum(x[x < 0]) != 0 else 9999))))
        
        # Select features for clustering
        features = ['returns', 'volatility', 'volume_ratio', 'rsi']
        
        # Drop NaN values
        df_features = df[features].dropna()
        
        if len(df_features) < self.n_regimes:
            # Not enough data for clustering
            regime_df = pd.DataFrame(index=df.index)
            regime_df['regime'] = np.nan
            regime_df['regime_label'] = np.nan
            return regime_df
        
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df_features)
        
        # Fit K-means
        kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        
        # Assign clusters to data points
        df_features['regime'] = kmeans.predict(scaled_features)
        
        # Calculate regime characteristics
        regime_stats = df_features.groupby('regime').mean()
        
        # Label regimes based on characteristics
        regime_labels = {}
        for regime in range(self.n_regimes):
            stats = regime_stats.loc[regime]
            
            if stats['returns'] > 0.001:
                trend = 'bull'
            elif stats['returns'] < -0.001:
                trend = 'bear'
            else:
                trend = 'neutral'
                
            if stats['volatility'] > regime_stats['volatility'].median():
                vol = 'high_vol'
            else:
                vol = 'low_vol'
                
            if stats['volume_ratio'] > regime_stats['volume_ratio'].median():
                vol_str = 'high_volume'
            else:
                vol_str = 'low_volume'
                
            regime_labels[regime] = f"{trend}_{vol}_{vol_str}"
        
        # Create regime DataFrame
        regime_df = pd.DataFrame(index=df.index)
        regime_df.loc[df_features.index, 'regime'] = df_features['regime']
        regime_df['regime_label'] = regime_df['regime'].map(regime_labels)
        
        # Store model for future predictions
        self.model = (kmeans, scaler, features, regime_labels)
        
        return regime_df
    
    def _detect_hmm(self, data):
        """
        Detect regimes using Hidden Markov Model.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data with columns: open, high, low, close, volume
            
        Returns:
        --------
        pd.DataFrame
            Data with regime information
        """
        # This is a placeholder for HMM implementation
        # In a real implementation, this would use the hmmlearn package
        print("HMM regime detection not implemented yet.")
        
        # Return empty regime DataFrame with the same index as data
        regime_df = pd.DataFrame(index=data.index)
        regime_df['regime'] = np.nan
        regime_df['regime_label'] = np.nan
        
        return regime_df
    
    def _calculate_regime_stats(self, data, regimes):
        """
        Calculate statistics for each regime.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Price data with columns: open, high, low, close, volume
        regimes : pd.DataFrame
            Data with regime information
        """
        if 'regime_label' not in regimes.columns:
            return
        
        # Filter out NaN regimes
        valid_regimes = regimes.dropna(subset=['regime_label'])
        
        if valid_regimes.empty:
            return
        
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Join returns with regimes
        stats_df = pd.DataFrame({'returns': returns})
        stats_df['regime_label'] = regimes['regime_label']
        
        # Group by regime and calculate statistics
        grouped = stats_df.groupby('regime_label')
        
        stats = {}
        for name, group in grouped:
            if group.empty or len(group) < 5:  # Skip if not enough data
                continue
                
            regime_returns = group['returns'].dropna()
            if regime_returns.empty:
                continue
                
            mean_return = regime_returns.mean()
            std_return = regime_returns.std()
            
            # Annualize (assuming daily data)
            ann_return = mean_return * 252
            ann_std = std_return * np.sqrt(252)
            
            # Calculate Sharpe ratio
            sharpe = ann_return / ann_std if ann_std != 0 else 0
            
            # Calculate win rate
            positive_pct = (regime_returns > 0).mean()
            
            # Calculate time spent in regime
            time_pct = len(group) / len(stats_df)
            
            stats[name] = {
                'count': len(group),
                'mean_return': mean_return,
                'std_return': std_return,
                'ann_return': ann_return,
                'ann_std': ann_std,
                'sharpe': sharpe,
                'positive_pct': positive_pct,
                'time_pct': time_pct
            }
        
        # Convert to DataFrame
        self.regime_stats = pd.DataFrame.from_dict(stats, orient='index')
    
    def get_current_regime(self):
        """
        Get the current market regime.
        
        Returns:
        --------
        dict
            Current regime information
        """
        if self.regime_history is None or self.regime_history.empty:
            raise ValueError("No regime history available. Call detect_regime() first.")
        
        # Get the most recent non-NaN regime
        recent_regimes = self.regime_history.dropna(subset=['regime_label'])
        
        if recent_regimes.empty:
            raise ValueError("No valid regimes detected.")
        
        current_regime = recent_regimes.iloc[-1]
        
        return {
            'regime': current_regime['regime'],
            'regime_label': current_regime['regime_label'],
            'date': current_regime.name
        }
    
    def get_regime_stats(self):
        """
        Get statistics for each regime.
        
        Returns:
        --------
        pd.DataFrame
            Regime statistics
        """
        if self.regime_stats is None:
            raise ValueError("No regime statistics available. Call detect_regime() first.")
        
        return self.regime_stats