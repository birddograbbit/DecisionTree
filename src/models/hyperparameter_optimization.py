"""
Hyperparameter optimization using Optuna.

This module provides functions for optimizing model hyperparameters
using Optuna with time series cross-validation to prevent data leakage.
"""

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import logging
import joblib
import os

from src.models.model_factory import ModelFactory

# Set up logging
logger = logging.getLogger(__name__)

def define_param_space(trial, model_type):
    """
    Define the hyperparameter search space for different model types.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
    model_type : str
        Type of model ('decision_tree', 'random_forest', 'xgboost', 'stacking')
        
    Returns:
    --------
    dict
        Hyperparameter configuration for the trial
    """
    # Common parameters for all model types
    params = {
        'calibrate': True  # Always use calibration for tree-based models
    }
    
    if model_type == 'decision_tree':
        params.update({
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
        })
    
    elif model_type == 'random_forest':
        params.update({
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
        })
    
    elif model_type == 'xgboost':
        params.update({
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.5, 10.0)  # For class imbalance
        })
    
    elif model_type == 'stacking':
        # For stacking model, we optimize meta-model and cross-validation settings
        # Base models will use their optimized configurations
        params.update({
            'cv': trial.suggest_int('cv', 3, 10),
            'use_features': trial.suggest_categorical('use_features', [True, False]),
            'meta_model_type': trial.suggest_categorical('meta_model_type', ['logistic_regression', 'random_forest']),
        })
        
        # Meta-model specific parameters
        if params['meta_model_type'] == 'logistic_regression':
            params['meta_model_params'] = {
                'C': trial.suggest_float('meta_C', 0.001, 10.0, log=True),
                'max_iter': 1000,  # Set high enough to avoid convergence warnings
                'solver': trial.suggest_categorical('meta_solver', ['liblinear', 'saga']),
                'class_weight': trial.suggest_categorical('meta_class_weight', ['balanced', None])
            }
        else:  # Random forest meta-model
            params['meta_model_params'] = {
                'n_estimators': trial.suggest_int('meta_n_estimators', 50, 200),
                'max_depth': trial.suggest_int('meta_max_depth', 2, 7),
                'class_weight': trial.suggest_categorical('meta_class_weight', ['balanced', None])
            }
    
    return params

def objective(trial, X, y, model_type, n_splits=5, scoring='accuracy'):
    """
    Objective function for Optuna optimization.
    
    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target values
    model_type : str
        Type of model to optimize
    n_splits : int, default=5
        Number of splits for time series cross-validation
    scoring : str, default='accuracy'
        Scoring metric to use for optimization ('accuracy', 'f1', 'roc_auc')
        
    Returns:
    --------
    float
        Average score across all splits
    """
    # Define parameters based on model type
    params = define_param_space(trial, model_type)
    
    # Initialize time series cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Store scores for each fold
    scores = []
    
    # Perform cross-validation
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create and train model pipeline
        model = ModelFactory.create_model(model_type, **params)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_val)
        y_pred_proba = None
        if hasattr(pipeline, 'predict_proba'):
            try:
                y_pred_proba = pipeline.predict_proba(X_val)[:, 1]
            except:
                # Some models might not support predict_proba
                pass
        
        # Calculate score based on the specified metric
        if scoring == 'accuracy':
            fold_score = accuracy_score(y_val, y_pred)
        elif scoring == 'f1':
            fold_score = f1_score(y_val, y_pred)
        elif scoring == 'roc_auc':
            if y_pred_proba is not None:
                fold_score = roc_auc_score(y_val, y_pred_proba)
            else:
                fold_score = 0.5  # Default for models without predict_proba
        else:
            raise ValueError(f"Unknown scoring metric: {scoring}")
        
        scores.append(fold_score)
    
    # Return the average score across all folds
    return np.mean(scores)

def optimize_hyperparameters(X, y, model_type, n_trials=100, n_splits=5, scoring='accuracy', output_dir='optimization_results'):
    """
    Optimize hyperparameters for a specific model type.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target values
    model_type : str
        Type of model to optimize
    n_trials : int, default=100
        Number of trials for optimization
    n_splits : int, default=5
        Number of splits for time series cross-validation
    scoring : str, default='accuracy'
        Scoring metric to use for optimization ('accuracy', 'f1', 'roc_auc')
    output_dir : str, default='optimization_results'
        Directory to save optimization results
        
    Returns:
    --------
    dict
        Best hyperparameters found during optimization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, f'{model_type}_optimization.log')
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    logger.info(f"Starting hyperparameter optimization for {model_type} using {scoring} metric")
    
    # Define the study
    study_name = f"{model_type}_{scoring}_optimization"
    study = optuna.create_study(direction='maximize', study_name=study_name)
    
    # Start optimization
    try:
        study.optimize(
            lambda trial: objective(trial, X, y, model_type, n_splits, scoring), 
            n_trials=n_trials
        )
        
        # Log best results
        logger.info(f"Best {scoring}: {study.best_value:.4f}")
        logger.info(f"Best parameters: {study.best_params}")
        
        # Save study results
        study_path = os.path.join(output_dir, f'{model_type}_study.pkl')
        joblib.dump(study, study_path)
        
        # Create best model with optimal parameters
        best_params = study.best_params
        best_model = ModelFactory.create_model(model_type, **best_params)
        
        # Save best model
        model_path = os.path.join(output_dir, f'{model_type}_best_model.pkl')
        joblib.dump(best_model, model_path)
        
        # Save optimization results summary
        summary_path = os.path.join(output_dir, f'{model_type}_optimization_summary.txt')
        with open(summary_path, 'w') as f:
            f.write(f"Optimization results for {model_type}\n")
            f.write(f"Metric: {scoring}\n")
            f.write(f"Best {scoring}: {study.best_value:.4f}\n\n")
            f.write("Best parameters:\n")
            for param, value in study.best_params.items():
                f.write(f"{param}: {value}\n")
        
        return study.best_params
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        raise
    finally:
        logger.removeHandler(file_handler)

def optimize_all_models(X, y, n_trials=100, n_splits=5, scoring='accuracy', output_dir='optimization_results'):
    """
    Optimize hyperparameters for all available models.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target values
    n_trials : int, default=100
        Number of trials for optimization
    n_splits : int, default=5
        Number of splits for time series cross-validation
    scoring : str, default='accuracy'
        Scoring metric to use for optimization ('accuracy', 'f1', 'roc_auc')
    output_dir : str, default='optimization_results'
        Directory to save optimization results
        
    Returns:
    --------
    dict
        Best hyperparameters for each model type
    """
    # Get available models
    model_types = ModelFactory.get_available_models()
    
    # Store best parameters for each model
    best_params = {}
    
    # Optimize each model
    for model_type in model_types:
        logger.info(f"Optimizing {model_type}...")
        try:
            model_best_params = optimize_hyperparameters(
                X, y, model_type, n_trials, n_splits, scoring, output_dir
            )
            best_params[model_type] = model_best_params
        except Exception as e:
            logger.error(f"Error optimizing {model_type}: {e}")
    
    # Save all best parameters to a single file
    all_params_path = os.path.join(output_dir, 'all_models_best_params.joblib')
    joblib.dump(best_params, all_params_path)
    
    return best_params
