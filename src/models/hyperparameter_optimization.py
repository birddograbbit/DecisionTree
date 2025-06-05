# src/models/hyperparameter_optimization.py

"""
Module for hyperparameter optimization using Optuna.

This module provides optimization functions for different model types.
For saving and loading hyperparameters, use HyperparameterManager from 
src.models.hyperparameter_manager instead of the deprecated functions
that were previously in this module.
"""

import os
import pickle
import numpy as np
import pandas as pd
import config
import warnings
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
import config

def optimize_decision_tree(X, y, n_trials=100, n_splits=5, random_state=42):
    """
    Optimize hyperparameters for Decision Tree model using Optuna.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target values
    n_trials : int
        Number of optimization trials (default: 100)
    n_splits : int
        Number of splits for TimeSeriesSplit (default: 5)
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    dict
        Best hyperparameters
    """
    # Define the objective function for Optuna
    def objective(trial):
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        
        # Define hyperparameters to optimize
        params = {
            'max_depth': trial.suggest_int('max_depth', 2, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
        }
        
        # Create model with suggested hyperparameters
        model = DecisionTreeClassifier(
            random_state=random_state,
            **params
        )
        
        # Create pipeline with scaling
        pipeline = make_pipeline(StandardScaler(), model)
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Evaluate model with cross-validation
        scores = cross_val_score(
            pipeline, X, y, 
            cv=tscv, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Return mean score
        return scores.mean()
    
    # Create study with TPE sampler
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state)
    )
    
    # Optimize hyperparameters
    study.optimize(objective, n_trials=n_trials)
    
    # Return best hyperparameters
    return study.best_params

def optimize_random_forest(X, y, n_trials=100, n_splits=5, random_state=42):
    """
    Optimize hyperparameters for Random Forest model using Optuna.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target values
    n_trials : int
        Number of optimization trials (default: 100)
    n_splits : int
        Number of splits for TimeSeriesSplit (default: 5)
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    dict
        Best hyperparameters
    """
    # Define the objective function for Optuna
    def objective(trial):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        
        # Define hyperparameters to optimize
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
        }
        
        # Create model with suggested hyperparameters
        model = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1,
            **params
        )
        
        # Create pipeline with scaling
        pipeline = make_pipeline(StandardScaler(), model)
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Evaluate model with cross-validation
        scores = cross_val_score(
            pipeline, X, y, 
            cv=tscv, 
            scoring='accuracy',
            n_jobs=-1
        )
        
        # Return mean score
        return scores.mean()
    
    # Create study with TPE sampler
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state)
    )
    
    # Optimize hyperparameters
    study.optimize(objective, n_trials=n_trials)
    
    # Return best hyperparameters
    return study.best_params

def optimize_xgboost(X, y, n_trials=100, n_splits=5, random_state=42):
    """
    Optimize hyperparameters for XGBoost model using Optuna.
    
    This function now works with the simplified XGBoost implementation
    that uses standard XGBClassifier without focal loss complexity.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target values
    n_trials : int
        Number of optimization trials (default: 100)
    n_splits : int
        Number of splits for TimeSeriesSplit (default: 5)
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    dict
        Best hyperparameters
    """
    # Define the objective function for Optuna
    def objective(trial):
        try:
            # Import required modules
            from ..models.xgboost_model import XGBoostModel
            
            # Define hyperparameters to optimize (simplified - no focal loss)
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
            }
            
            # Create custom evaluation function using the simplified XGBoostModel
            def custom_cv_score_with_model(X, y, cv):
                scores = []
                for train_idx, test_idx in cv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Create XGBoostModel with suggested hyperparameters
                    model = XGBoostModel(
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'],
                        learning_rate=params['learning_rate'],
                        subsample=params['subsample'],
                        colsample_bytree=params['colsample_bytree'],
                        gamma=params['gamma'],
                        min_child_weight=params['min_child_weight'],
                        reg_alpha=params['reg_alpha'],
                        reg_lambda=params['reg_lambda'],
                        scale_pos_weight=params['scale_pos_weight'],
                        random_state=random_state,
                        class_weight=params['class_weight']
                    )
                    
                    # Train the model using the same method as in production
                    model.train(X_train, y_train)
                    
                    # Get predictions using the same method as in production
                    y_pred_proba = model.predict(X_test)
                    
                    # Convert to binary predictions using threshold 0.5
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    
                    # Calculate accuracy
                    score = (y_pred == y_test).mean()
                    scores.append(score)
                
                return np.mean(scores)
            
            # Use TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Perform custom cross-validation using the actual XGBoostModel
            score = custom_cv_score_with_model(X, y, tscv)
            
            # Return the mean score
            return score
            
        except ImportError as e:
            # If XGBoost is not available, return a bad score
            print(f"Import error in XGBoost optimization: {e}")
            return 0.0
        except Exception as e:
            # Log any errors for debugging
            print(f"Error in XGBoost optimization: {e}")
            return 0.0
    
    # Create study with TPE sampler
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state)
    )
    
    # Optimize hyperparameters
    study.optimize(objective, n_trials=n_trials)
    
    # Return best hyperparameters
    return study.best_params

def get_sample_weights(y, class_weight='balanced'):
    """
    Calculate sample weights for class imbalance.
    
    Parameters:
    -----------
    y : pd.Series or np.ndarray
        Target values
    class_weight : str or dict
        Class weight strategy (default: 'balanced')
        
    Returns:
    --------
    np.ndarray
        Sample weights
    """
    if class_weight == 'balanced':
        # Calculate balanced class weights
        class_counts = np.bincount(y)
        total_samples = len(y)
        weight_for_0 = total_samples / (2 * class_counts[0])
        weight_for_1 = total_samples / (2 * class_counts[1])
        
        # Set sample weights
        sample_weights = np.ones(len(y))
        sample_weights[y == 0] = weight_for_0
        sample_weights[y == 1] = weight_for_1
        
        return sample_weights
    
    elif isinstance(class_weight, dict):
        # Convert class weight dictionary to sample weights
        sample_weights = np.ones(len(y))
        for class_val, weight in class_weight.items():
            sample_weights[y == class_val] = weight
            
        return sample_weights
    
    else:
        # No class weights
        return np.ones(len(y))

def optimize_hyperparameters(model_type, X, y, n_trials=100, n_splits=5, random_state=42):
    """
    Optimize hyperparameters for a given model type using Optuna.
    
    Parameters:
    -----------
    model_type : str
        Model type ('decision_tree', 'random_forest', 'xgboost')
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target values
    n_trials : int
        Number of optimization trials (default: 100)
    n_splits : int
        Number of splits for TimeSeriesSplit (default: 5)
    random_state : int
        Random seed for reproducibility (default: 42)
        
    Returns:
    --------
    dict
        Best hyperparameters
    """
    if model_type == 'decision_tree':
        return optimize_decision_tree(X, y, n_trials, n_splits, random_state)
    elif model_type == 'random_forest':
        return optimize_random_forest(X, y, n_trials, n_splits, random_state)
    elif model_type == 'xgboost':
        return optimize_xgboost(X, y, n_trials, n_splits, random_state)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

# DEPRECATED FUNCTIONS - Use HyperparameterManager instead
# =========================================================

def save_hyperparameters(params, model_type, path=config.HYPERPARAMS_DIR):
    """
    DEPRECATED: Save hyperparameters to disk.
    
    This function is deprecated. Use HyperparameterManager.save_params() instead:
    
    from src.models.hyperparameter_manager import HyperparameterManager
    manager = HyperparameterManager()
    manager.save_params(params, model_type)
    
    Parameters:
    -----------
    params : dict
        Hyperparameters
    model_type : str
        Model type ('decision_tree', 'random_forest', 'xgboost')
    path : str
        Path to save hyperparameters (default: config.HYPERPARAMS_DIR)
    """
    warnings.warn(
        "save_hyperparameters() is deprecated. Use HyperparameterManager.save_params() instead. "
        "Import with: from src.models.hyperparameter_manager import HyperparameterManager",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Fallback to basic functionality for backward compatibility
    os.makedirs(path, exist_ok=True)
    filename = os.path.join(path, f'{model_type}_hyperparameters.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
    print(f"Hyperparameters saved to {filename}")
    print("WARNING: Consider migrating to HyperparameterManager for better functionality")

def load_hyperparameters(model_type, path=config.HYPERPARAMS_DIR):
    """
    DEPRECATED: Load hyperparameters from disk.
    
    This function is deprecated. Use HyperparameterManager.get_best_params() instead:
    
    from src.models.hyperparameter_manager import HyperparameterManager
    manager = HyperparameterManager()
    params = manager.get_best_params(model_type)
    
    Parameters:
    -----------
    model_type : str
        Model type ('decision_tree', 'random_forest', 'xgboost')
    path : str
        Path to load hyperparameters from (default: config.HYPERPARAMS_DIR)
        
    Returns:
    --------
    dict
        Hyperparameters
    """
    warnings.warn(
        "load_hyperparameters() is deprecated. Use HyperparameterManager.get_best_params() instead. "
        "Import with: from src.models.hyperparameter_manager import HyperparameterManager",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Fallback to basic functionality for backward compatibility
    filename = os.path.join(path, f'{model_type}_hyperparameters.pkl')
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Hyperparameters file not found: {filename}")
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    print("WARNING: Consider migrating to HyperparameterManager for better functionality")
    return params
