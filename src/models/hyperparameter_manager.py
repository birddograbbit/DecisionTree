# src/models/hyperparameter_manager.py

"""
Hyperparameter Manager Module for automated hyperparameter loading, persistence, and optimization.

This module provides functionality to:
1. Automatically load the best hyperparameters for each model
2. Save optimized hyperparameters to versioned files
3. Create model instances with optimized parameters
4. Support regime-specific hyperparameter optimization

Part of Phase 1.5 implementation - Hyperparameter Optimization Integration
"""

import os
import json
import pickle
import datetime
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import config

# Configure logging
logger = logging.getLogger(__name__)

class HyperparameterManager:
    """
    Manage hyperparameters for machine learning models in the trading system.
    
    This class handles loading, saving, and optimizing hyperparameters for
    different model types, including support for regime-specific optimization.
    """
    
    def __init__(self, base_path=config.HYPERPARAMS_DIR):
        """
        Initialize HyperparameterManager.
        
        Parameters:
        -----------
        base_path : str
            Base directory path for saving/loading hyperparameters
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Directory for versioned hyperparameters
        self.versioned_path = self.base_path / 'versioned'
        self.versioned_path.mkdir(exist_ok=True)
        
        # Directory for regime-specific hyperparameters
        self.regime_path = self.base_path / 'regimes'
        self.regime_path.mkdir(exist_ok=True)
    
    def get_best_params(self, model_type, regime=None):
        """
        Get the best hyperparameters for a specific model type.
        
        Parameters:
        -----------
        model_type : str
            Type of model ('decision_tree', 'random_forest', 'xgboost', 'stacking')
        regime : str or None
            Market regime (if None, use general hyperparameters)
            
        Returns:
        --------
        dict
            Best hyperparameters for the model
        """
        # First try regime-specific hyperparameters if requested
        if regime is not None:
            regime_params = self._load_regime_params(model_type, regime)
            if regime_params is not None:
                logger.info(f"Loaded regime-specific hyperparameters for {model_type} in {regime} regime")
                return regime_params
            
        # If no regime-specific params found, try general hyperparameters
        general_params = self._load_general_params(model_type)
        if general_params is not None:
            logger.info(f"Loaded general hyperparameters for {model_type}")
            return general_params
            
        # If no saved hyperparameters found, use defaults from ModelFactory
        from .model_factory import ModelFactory
        logger.warning(f"No saved hyperparameters found for {model_type}. Using defaults.")
        return ModelFactory.get_default_params(model_type)
    
    def _load_general_params(self, model_type):
        """
        Load general hyperparameters for a model type.
        
        Parameters:
        -----------
        model_type : str
            Type of model
            
        Returns:
        --------
        dict or None
            Hyperparameters if found, None otherwise
        """
        # Check for latest versioned parameters
        versions = self._get_versioned_files(model_type)
        if versions:
            latest_version = sorted(versions)[-1]
            return self._load_params_file(self.versioned_path / latest_version)
        
        # If no versioned parameters, try the base file
        base_file = self.base_path / f"{model_type}_hyperparameters.pkl"
        if base_file.exists():
            return self._load_params_file(base_file)
        
        return None
    
    def _load_regime_params(self, model_type, regime):
        """
        Load regime-specific hyperparameters.
        
        Parameters:
        -----------
        model_type : str
            Type of model
        regime : str
            Market regime
            
        Returns:
        --------
        dict or None
            Hyperparameters if found, None otherwise
        """
        regime_file = self.regime_path / f"{model_type}_{regime}_hyperparameters.pkl"
        if regime_file.exists():
            return self._load_params_file(regime_file)
        
        return None
    
    def _get_versioned_files(self, model_type):
        """
        Get all versioned parameter files for a model type.
        
        Parameters:
        -----------
        model_type : str
            Type of model
            
        Returns:
        --------
        list
            List of versioned parameter filenames
        """
        if not self.versioned_path.exists():
            return []
        
        # Find all versioned parameter files
        prefix = f"{model_type}_hyperparameters_v"
        return [f for f in os.listdir(self.versioned_path) 
                if f.startswith(prefix) and f.endswith('.pkl')]
    
    def _load_params_file(self, file_path):
        """
        Load parameters from a file.
        
        Parameters:
        -----------
        file_path : str or Path
            Path to parameter file
            
        Returns:
        --------
        dict or None
            Loaded parameters
        """
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except (IOError, pickle.PickleError) as e:
            logger.error(f"Error loading parameters from {file_path}: {e}")
            return None
    
    def save_params(self, params, model_type, regime=None, create_version=True):
        """
        Save hyperparameters.
        
        Parameters:
        -----------
        params : dict
            Hyperparameters to save
        model_type : str
            Type of model
        regime : str or None
            Market regime (if None, save as general hyperparameters)
        create_version : bool
            Whether to create a versioned copy
            
        Returns:
        --------
        str
            Path to saved file
        """
        # Determine save path
        if regime is not None:
            # Save regime-specific parameters
            save_path = self.regime_path / f"{model_type}_{regime}_hyperparameters.pkl"
            logger.info(f"Saving regime-specific hyperparameters for {model_type} in {regime} regime")
        else:
            # Save general parameters
            save_path = self.base_path / f"{model_type}_hyperparameters.pkl"
            logger.info(f"Saving general hyperparameters for {model_type}")
        
        # Save parameters
        with open(save_path, 'wb') as f:
            pickle.dump(params, f)
        
        # Create versioned copy if requested
        if create_version and regime is None:
            version = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            version_path = self.versioned_path / f"{model_type}_hyperparameters_v{version}.pkl"
            
            with open(version_path, 'wb') as f:
                pickle.dump(params, f)
            
            logger.info(f"Created versioned hyperparameters: {version_path}")
            
            # Save metadata
            self._save_metadata(model_type, version, params)
        
        return str(save_path)
    
    def _save_metadata(self, model_type, version, params):
        """
        Save metadata for versioned hyperparameters.
        
        Parameters:
        -----------
        model_type : str
            Type of model
        version : str
            Version string
        params : dict
            Hyperparameters
        """
        # Create metadata
        metadata = {
            'model_type': model_type,
            'version': version,
            'created_at': datetime.datetime.now().isoformat(),
            'params': {k: str(v) for k, v in params.items()}  # Convert to strings for JSON
        }
        
        # Save metadata
        metadata_path = self.versioned_path / f"{model_type}_hyperparameters_v{version}_meta.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def optimize_hyperparameters(self, model_type, X, y, n_trials=None, regime=None):
        """
        Optimize hyperparameters for a model type.
        
        Parameters:
        -----------
        model_type : str
            Type of model
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target vector
        n_trials : int or None
            Number of optimization trials (if None, use default from config)
        regime : str or None
            Market regime (if provided, optimize for this specific regime)
            
        Returns:
        --------
        dict
            Optimized hyperparameters
        """
        # Import here to avoid circular imports
        import config
        from .hyperparameter_optimization import optimize_hyperparameters
        
        # Use default from config if n_trials not specified
        if n_trials is None:
            n_trials = config.OPTUNA_TRIALS
        
        logger.info(f"Optimizing hyperparameters for {model_type}" + 
                   (f" in {regime} regime" if regime else ""))
        
        # Optimize hyperparameters
        best_params = optimize_hyperparameters(
            model_type=model_type,
            X=X,
            y=y,
            n_trials=n_trials,
            n_splits=config.TIMESERIES_CV_SPLITS,
            random_state=config.RANDOM_STATE
        )
        
        # Save optimized hyperparameters
        self.save_params(best_params, model_type, regime)
        
        return best_params
    
    def create_optimized_model(self, model_type, X=None, y=None, regime=None, 
                            force_optimization=False, n_trials=None):
        """
        Create a model with optimized hyperparameters.
        
        Parameters:
        -----------
        model_type : str
            Type of model
        X : pd.DataFrame or None
            Feature matrix (required if force_optimization=True)
        y : pd.Series or None
            Target vector (required if force_optimization=True)
        regime : str or None
            Market regime
        force_optimization : bool
            Whether to force optimization even if saved parameters exist
        n_trials : int or None
            Number of optimization trials
            
        Returns:
        --------
        BaseModel
            Model instance with optimized hyperparameters
        """
        # Import here to avoid circular imports
        from .model_factory import ModelFactory
        
        # If forcing optimization and data is provided, optimize
        if force_optimization and X is not None and y is not None:
            best_params = self.optimize_hyperparameters(model_type, X, y, n_trials, regime)
        else:
            # Otherwise, load best parameters
            best_params = self.get_best_params(model_type, regime)
        
        # Create model with best parameters
        return ModelFactory.create_model(model_type, **best_params)
    
    def get_regime_specific_models(self, X, y, model_type, regimes, 
                                    force_optimization=False, n_trials=None):
        """
        Create regime-specific models with optimized hyperparameters.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Full feature matrix
        y : pd.Series
            Full target vector
        model_type : str
            Type of model
        regimes : dict
            Dictionary mapping data indices to regime labels
        force_optimization : bool
            Whether to force optimization even if saved parameters exist
        n_trials : int or None
            Number of optimization trials
            
        Returns:
        --------
        dict
            Dictionary mapping regimes to optimized models
        """
        # Create dictionary to hold regime-specific models
        regime_models = {}
        
        # Get unique regimes
        unique_regimes = set(regimes.values())
        
        for regime in unique_regimes:
            # Get indices for this regime
            regime_indices = [idx for idx, r in regimes.items() if r == regime and idx < len(X)]
            
            # Skip if too few samples
            if len(regime_indices) < 100:  # Minimum samples threshold
                logger.warning(f"Too few samples ({len(regime_indices)}) for regime {regime}. "
                              f"Using general model instead.")
                regime_models[regime] = self.create_optimized_model(model_type)
                continue
            
            # Extract data for this regime using positional indices
            X_regime = X.iloc[regime_indices]
            y_regime = y.iloc[regime_indices]
            
            # Create optimized model for this regime
            regime_models[regime] = self.create_optimized_model(
                model_type, X_regime, y_regime, regime, 
                force_optimization, n_trials
            )
            
            logger.info(f"Created regime-specific model for {regime} with {len(X_regime)} samples")
        
        return regime_models
