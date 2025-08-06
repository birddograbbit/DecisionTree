"""
Strategy Registry for centralized strategy management.

This module provides a registry pattern for managing all available trading strategies,
enabling easy addition of new strategies and dynamic strategy selection.
"""

import logging
import numpy as np
from typing import Dict, Type, Optional, List
from src.strategies.base_strategy import BaseStrategy
from src.strategies.trend_following import TrendFollowingStrategy
from src.strategies.regime_adaptive_strategy import RegimeAdaptiveStrategy
from src.strategies.adapters import BBRSIADXAdapter, TEMAAdapter, QuodAdapter

# Configure logging
logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    Central registry for all available trading strategies.
    
    This class implements the registry pattern to manage strategy implementations,
    providing a clean interface for strategy discovery and instantiation.
    """
    
    # Class-level registry of available strategies
    _strategies: Dict[str, Type[BaseStrategy]] = {
        'trend_following': TrendFollowingStrategy,
        'regime_adaptive': RegimeAdaptiveStrategy,
        # Momentum strategies
        'bb_rsi_adx': BBRSIADXAdapter,
        'tema': TEMAAdapter,
        'quod': QuodAdapter,
    }
    
    # Aliases for backward compatibility and convenience
    _aliases: Dict[str, str] = {
        'default': 'trend_following',
        'trend': 'trend_following',
        'regime': 'regime_adaptive',
        'adaptive': 'regime_adaptive',
    }
    
    # Performance tracking for strategies (optional)
    _performance_tracking: Dict[str, List[float]] = {}
    
    @classmethod
    def register_strategy(cls, name: str, strategy_class: Type[BaseStrategy], 
                         aliases: Optional[List[str]] = None) -> None:
        """
        Register a new strategy with the registry.
        
        Parameters:
        -----------
        name : str
            Unique name for the strategy
        strategy_class : Type[BaseStrategy]
            Strategy class that extends BaseStrategy
        aliases : List[str], optional
            Alternative names for the strategy
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"Strategy {strategy_class} must extend BaseStrategy")
        
        if name in cls._strategies:
            logger.warning(f"Overwriting existing strategy '{name}'")
        
        cls._strategies[name] = strategy_class
        logger.info(f"Registered strategy '{name}': {strategy_class.__name__}")
        
        # Register aliases if provided
        if aliases:
            for alias in aliases:
                cls._aliases[alias] = name
                logger.debug(f"Added alias '{alias}' for strategy '{name}'")
    
    @classmethod
    def get_strategy(cls, name: str, config: Optional[Dict] = None) -> BaseStrategy:
        """
        Get an instance of a strategy by name.
        
        Parameters:
        -----------
        name : str
            Strategy name or alias
        config : dict, optional
            Configuration to pass to strategy initialization
            
        Returns:
        --------
        BaseStrategy
            Initialized strategy instance
        """
        # Resolve aliases
        strategy_name = cls._aliases.get(name, name)
        
        if strategy_name not in cls._strategies:
            available = cls.list_strategies()
            raise ValueError(
                f"Unknown strategy: '{name}'. Available strategies: {available}"
            )
        
        strategy_class = cls._strategies[strategy_name]
        strategy_instance = strategy_class()
        
        # Initialize with config if provided
        if config:
            strategy_instance.initialize(config)
        
        logger.info(f"Created strategy instance: {strategy_name}")
        return strategy_instance
    
    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        Get list of all available strategy names.
        
        Returns:
        --------
        List[str]
            List of registered strategy names
        """
        return list(cls._strategies.keys())
    
    @classmethod
    def list_aliases(cls) -> Dict[str, str]:
        """
        Get all strategy aliases.
        
        Returns:
        --------
        Dict[str, str]
            Mapping of aliases to strategy names
        """
        return cls._aliases.copy()
    
    @classmethod
    def get_strategy_info(cls, name: str) -> Dict[str, any]:
        """
        Get detailed information about a strategy.
        
        Parameters:
        -----------
        name : str
            Strategy name or alias
            
        Returns:
        --------
        Dict[str, any]
            Strategy information including class, features, timeframes
        """
        # Resolve aliases
        strategy_name = cls._aliases.get(name, name)
        
        if strategy_name not in cls._strategies:
            raise ValueError(f"Unknown strategy: '{name}'")
        
        strategy_class = cls._strategies[strategy_name]
        
        # Create temporary instance to get requirements
        temp_instance = strategy_class()
        
        info = {
            'name': strategy_name,
            'class': strategy_class.__name__,
            'module': strategy_class.__module__,
            'docstring': strategy_class.__doc__,
            'required_features': temp_instance.get_required_features(),
            'required_timeframes': temp_instance.get_required_timeframes(),
            'order_config': temp_instance.get_order_management_config(),
        }
        
        return info
    
    @classmethod
    def validate_strategy_config(cls, name: str, config: Dict) -> bool:
        """
        Validate configuration for a strategy.
        
        Parameters:
        -----------
        name : str
            Strategy name or alias
        config : dict
            Configuration to validate
            
        Returns:
        --------
        bool
            True if configuration is valid
        """
        try:
            # Try to create instance with config
            strategy = cls.get_strategy(name, config)
            
            # Check if data requirements can be met
            if 'data_timeframes' in config:
                return strategy.validate_data_requirements(config['data_timeframes'])
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    @classmethod
    def create_strategy(cls, name: str, config: Optional[Dict] = None) -> BaseStrategy:
        """
        Create a strategy instance (alias for get_strategy).
        
        Parameters:
        -----------
        name : str
            Strategy name or alias
        config : dict, optional
            Configuration for strategy initialization
            
        Returns:
        --------
        BaseStrategy
            Initialized strategy instance
        """
        return cls.get_strategy(name, config)
    
    @classmethod
    def track_performance(cls, strategy_name: str, returns: List[float]) -> None:
        """
        Track performance data for a strategy.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
        returns : List[float]
            Returns to track
        """
        if strategy_name not in cls._performance_tracking:
            cls._performance_tracking[strategy_name] = []
        
        cls._performance_tracking[strategy_name].extend(returns)
        
        # Keep only recent history (last 1000 data points)
        if len(cls._performance_tracking[strategy_name]) > 1000:
            cls._performance_tracking[strategy_name] = cls._performance_tracking[strategy_name][-1000:]
    
    @classmethod
    def get_performance_stats(cls, strategy_name: str, window: int = 100) -> Dict[str, float]:
        """
        Get performance statistics for a strategy.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
        window : int
            Number of recent returns to analyze
            
        Returns:
        --------
        Dict[str, float]
            Performance statistics
        """
        if strategy_name not in cls._performance_tracking:
            return {}
        
        returns = cls._performance_tracking[strategy_name]
        if len(returns) < window:
            return {'data_points': len(returns), 'insufficient_data': True}
        
        recent_returns = returns[-window:]
        
        stats = {
            'mean_return': np.mean(recent_returns),
            'std_return': np.std(recent_returns),
            'sharpe_ratio': np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252) if np.std(recent_returns) > 0 else 0,
            'win_rate': sum(1 for r in recent_returns if r > 0) / len(recent_returns),
            'max_return': max(recent_returns),
            'min_return': min(recent_returns),
            'data_points': len(recent_returns)
        }
        
        return stats


# Convenience function for backward compatibility
def get_strategy(name: str, config: Optional[Dict] = None) -> BaseStrategy:
    """
    Convenience function to get a strategy instance.
    
    Parameters:
    -----------
    name : str
        Strategy name or alias
    config : dict, optional
        Configuration for strategy initialization
        
    Returns:
    --------
    BaseStrategy
        Initialized strategy instance
    """
    return StrategyRegistry.get_strategy(name, config)