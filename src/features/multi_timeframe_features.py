"""
Multi-timeframe Feature Aggregator for momentum strategies.

This module handles the creation and alignment of features across multiple timeframes,
enabling strategies to use indicators from different time periods simultaneously.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from src.features.indicators import *
from src.features.feature_engineering import add_technical_indicators

# Configure logging
logger = logging.getLogger(__name__)


class MultiTimeframeAggregator:
    """
    Aggregates features across multiple timeframes for strategy use.
    
    This class handles:
    - Resampling data to different timeframes
    - Computing indicators for each timeframe
    - Aligning features across timeframes
    - Caching computed features for efficiency
    """
    
    def __init__(self, base_features: Optional[List[str]] = None):
        """
        Initialize the multi-timeframe aggregator.
        
        Parameters:
        -----------
        base_features : List[str], optional
            List of base features to compute for each timeframe
        """
        self.base_features = base_features or [
            'returns', 'log_returns', 'sma', 'ema', 'rsi', 'macd', 
            'bb_upper', 'bb_lower', 'atr', 'adx', 'volume_momentum'
        ]
        self.feature_cache = {}
        
    def resample_data(self, data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data to a target timeframe.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Original OHLCV data
        target_timeframe : str
            Target timeframe (e.g., '4h', '1D')
            
        Returns:
        --------
        pd.DataFrame
            Resampled OHLCV data
        """
        # Ensure we have a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data = data.set_index('date')
            else:
                raise ValueError("Data must have datetime index or 'date' column")
        
        # Define aggregation rules for OHLCV
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Only apply rules for columns that exist
        rules_to_apply = {col: rule for col, rule in agg_rules.items() 
                         if col in data.columns}
        
        # Resample
        resampled = data.resample(target_timeframe).agg(rules_to_apply)
        
        # Remove any rows with NaN values
        resampled = resampled.dropna()
        
        logger.debug(f"Resampled data from {len(data)} to {len(resampled)} rows "
                    f"for timeframe {target_timeframe}")
        
        return resampled
    
    def compute_features_for_timeframe(self, data: pd.DataFrame, 
                                     timeframe: str,
                                     features: List[str]) -> pd.DataFrame:
        """
        Compute specified features for a given timeframe.
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data
        timeframe : str
            Timeframe identifier
        features : List[str]
            List of features to compute
            
        Returns:
        --------
        pd.DataFrame
            Computed features with timeframe suffix
        """
        # Check cache first
        cache_key = f"{timeframe}_{hash(tuple(features))}"
        if cache_key in self.feature_cache:
            logger.debug(f"Using cached features for {timeframe}")
            return self.feature_cache[cache_key]
        
        # Add standard technical indicators
        data_with_indicators = add_technical_indicators(data)
        
        # Initialize feature dataframe
        feature_df = pd.DataFrame(index=data_with_indicators.index)
        
        # Add requested features
        for feature in features:
            if feature in data_with_indicators.columns:
                # Add with timeframe suffix
                feature_df[f"{feature}_{timeframe}"] = data_with_indicators[feature]
            else:
                # Try to compute custom features
                if feature == 'trend_strength':
                    feature_df[f"{feature}_{timeframe}"] = self._compute_trend_strength(data_with_indicators)
                elif feature == 'momentum_score':
                    feature_df[f"{feature}_{timeframe}"] = self._compute_momentum_score(data_with_indicators)
                elif feature == 'volatility_regime':
                    feature_df[f"{feature}_{timeframe}"] = self._compute_volatility_regime(data_with_indicators)
                else:
                    logger.warning(f"Feature '{feature}' not found for timeframe {timeframe}")
        
        # Cache the results
        self.feature_cache[cache_key] = feature_df
        
        return feature_df
    
    def aggregate_features(self, data: Dict[str, pd.DataFrame], 
                         feature_requirements: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Aggregate features across multiple timeframes.
        
        Parameters:
        -----------
        data : Dict[str, pd.DataFrame]
            Dictionary mapping timeframe to OHLCV data
        feature_requirements : Dict[str, List[str]]
            Dictionary mapping timeframe to required features
            
        Returns:
        --------
        pd.DataFrame
            Aggregated features aligned to the finest timeframe
        """
        if not data:
            raise ValueError("No data provided for feature aggregation")
        
        # Find the base (finest) timeframe
        base_timeframe = self._find_base_timeframe(list(data.keys()))
        base_data = data[base_timeframe]
        
        # Start with base timeframe features
        all_features = pd.DataFrame(index=base_data.index)
        
        # Process each timeframe
        for timeframe, features in feature_requirements.items():
            if timeframe not in data:
                logger.warning(f"Timeframe {timeframe} not in provided data")
                continue
            
            # Compute features for this timeframe
            tf_features = self.compute_features_for_timeframe(
                data[timeframe], timeframe, features
            )
            
            # Align to base timeframe
            if timeframe != base_timeframe:
                tf_features = self._align_features_to_base(
                    tf_features, base_data.index
                )
            
            # Add to aggregated features
            all_features = all_features.join(tf_features, how='outer')
        
        # Forward fill any NaN values from alignment
        all_features = all_features.fillna(method='ffill')
        
        # Drop any remaining NaN rows
        all_features = all_features.dropna()
        
        logger.info(f"Aggregated {len(all_features.columns)} features across "
                   f"{len(feature_requirements)} timeframes")
        
        return all_features
    
    def create_momentum_features(self, data: pd.DataFrame, 
                               lookback_periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create specialized momentum features for momentum strategies.
        
        Parameters:
        -----------
        data : pd.DataFrame
            OHLCV data
        lookback_periods : List[int]
            Periods for momentum calculation
            
        Returns:
        --------
        pd.DataFrame
            Momentum features
        """
        momentum_features = pd.DataFrame(index=data.index)
        
        for period in lookback_periods:
            # Price momentum
            momentum_features[f'price_momentum_{period}'] = (
                data['close'] / data['close'].shift(period) - 1
            )
            
            # Volume momentum
            momentum_features[f'volume_momentum_{period}'] = (
                data['volume'] / data['volume'].rolling(period).mean() - 1
            )
            
            # Momentum acceleration
            momentum_features[f'momentum_accel_{period}'] = (
                momentum_features[f'price_momentum_{period}'] - 
                momentum_features[f'price_momentum_{period}'].shift(period)
            )
        
        # Composite momentum score
        momentum_features['momentum_composite'] = (
            momentum_features[[col for col in momentum_features.columns 
                             if 'price_momentum' in col]].mean(axis=1)
        )
        
        return momentum_features
    
    def _find_base_timeframe(self, timeframes: List[str]) -> str:
        """
        Find the finest (base) timeframe from a list.
        
        Parameters:
        -----------
        timeframes : List[str]
            List of timeframe strings
            
        Returns:
        --------
        str
            The finest timeframe
        """
        # Convert to pandas frequencies and sort
        freq_order = {'T': 1, 'min': 1, 'h': 60, 'H': 60, 'D': 1440, 'W': 10080}
        
        def get_minutes(tf):
            # Extract number and unit
            import re
            match = re.match(r'(\d*)([a-zA-Z]+)', tf)
            if match:
                num = int(match.group(1)) if match.group(1) else 1
                unit = match.group(2)
                return num * freq_order.get(unit, 1440)
            return 1440  # Default to daily
        
        # Sort by minutes (ascending)
        sorted_tf = sorted(timeframes, key=get_minutes)
        return sorted_tf[0]
    
    def _align_features_to_base(self, features: pd.DataFrame, 
                               base_index: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Align features from a coarser timeframe to a finer base timeframe.
        
        Parameters:
        -----------
        features : pd.DataFrame
            Features to align
        base_index : pd.DatetimeIndex
            Target index to align to
            
        Returns:
        --------
        pd.DataFrame
            Aligned features
        """
        # Reindex and forward fill
        aligned = features.reindex(base_index, method='ffill')
        
        return aligned
    
    def _compute_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute trend strength indicator.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with indicators
            
        Returns:
        --------
        pd.Series
            Trend strength values
        """
        # Use ADX if available
        if 'adx' in data.columns:
            return data['adx']
        
        # Otherwise compute simple trend strength
        sma_20 = data['close'].rolling(20).mean()
        sma_50 = data['close'].rolling(50).mean()
        
        trend_strength = abs(sma_20 - sma_50) / sma_50
        return trend_strength
    
    def _compute_momentum_score(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute composite momentum score.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with indicators
            
        Returns:
        --------
        pd.Series
            Momentum score values
        """
        scores = []
        
        # RSI momentum
        if 'rsi' in data.columns:
            rsi_score = (data['rsi'] - 50) / 50
            scores.append(rsi_score)
        
        # Price momentum
        price_mom = data['close'].pct_change(10)
        scores.append(price_mom)
        
        # MACD momentum
        if 'macd' in data.columns and 'macd_signal' in data.columns:
            macd_score = (data['macd'] - data['macd_signal']) / data['close']
            scores.append(macd_score)
        
        # Average all scores
        if scores:
            momentum_score = pd.concat(scores, axis=1).mean(axis=1)
        else:
            momentum_score = pd.Series(0, index=data.index)
        
        return momentum_score
    
    def _compute_volatility_regime(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute volatility regime classification.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data with indicators
            
        Returns:
        --------
        pd.Series
            Volatility regime (0=low, 1=medium, 2=high)
        """
        # Use ATR if available
        if 'atr' in data.columns:
            volatility = data['atr'] / data['close']
        else:
            # Use standard deviation of returns
            volatility = data['returns'].rolling(20).std()
        
        # Classify into regimes using rolling percentiles
        vol_percentile_33 = volatility.rolling(252).quantile(0.33)
        vol_percentile_67 = volatility.rolling(252).quantile(0.67)
        
        regime = pd.Series(1, index=data.index)  # Default medium
        regime[volatility <= vol_percentile_33] = 0  # Low
        regime[volatility >= vol_percentile_67] = 2  # High
        
        return regime


def create_multi_timeframe_features(data_dict: Dict[str, pd.DataFrame],
                                  strategy_requirements: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Convenience function to create multi-timeframe features.
    
    Parameters:
    -----------
    data_dict : Dict[str, pd.DataFrame]
        Dictionary mapping timeframe to OHLCV data
    strategy_requirements : Dict[str, List[str]]
        Dictionary mapping timeframe to required features
        
    Returns:
    --------
    pd.DataFrame
        Aggregated multi-timeframe features
    """
    aggregator = MultiTimeframeAggregator()
    return aggregator.aggregate_features(data_dict, strategy_requirements)