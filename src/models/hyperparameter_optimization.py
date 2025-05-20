# src/models/hyperparameter_optimization.py

"""
Module for hyperparameter optimization using Optuna.
"""

import os
import pickle
import numpy as np
import pandas as pd
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
            import xgboost as xgb
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import cross_val_score
            
            # Define hyperparameters to optimize
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
            }
            
            # Special case: Focal loss or class weights
            use_focal_loss = trial.suggest_categorical('use_focal_loss', [True, False])
            
            if use_focal_loss:
                focal_gamma = trial.suggest_float('focal_gamma', 0.5, 5.0)
                focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.9)
                class_weight = None
                # Add focal loss parameters to the params
                params['use_focal_loss'] = True
                params['focal_gamma'] = focal_gamma
                params['focal_alpha'] = focal_alpha
            else:
                focal_gamma = None
                focal_alpha = None
                class_weight = trial.suggest_categorical('class_weight', ['balanced', None])
                # Add class weight to params
                params['use_focal_loss'] = False
                params['class_weight'] = class_weight
            
            # Create custom evaluation function for cross-validation
            def custom_cv_score(estimator, X, y, cv):
                # Create simple dataset splits
                scores = []
                for train_idx, test_idx in cv.split(X):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Use the standard XGBClassifier directly instead of our custom model
                    # This ensures compatibility with scikit-learn's pipelines
                    model_params = {
                        'n_estimators': params['n_estimators'],
                        'max_depth': params['max_depth'],
                        'learning_rate': params['learning_rate'],
                        'subsample': params['subsample'],
                        'colsample_bytree': params['colsample_bytree'],
                        'gamma': params['gamma'],
                        'min_child_weight': params['min_child_weight'],
                        'reg_alpha': params['reg_alpha'],
                        'reg_lambda': params['reg_lambda'],
                        'scale_pos_weight': params['scale_pos_weight'],
                        'random_state': random_state,
                        'n_jobs': -1
                    }
                    
                    # Always initialize sample_weights to None
                    sample_weights = None
                    
                    # Handle class weight or focal loss
                    if use_focal_loss:
                        # For focal loss, use scale_pos_weight parameter
                        # This is a simplification since XGBoost doesn't directly support focal loss
                        # But scale_pos_weight helps with class imbalance
                        pass
                    else:
                        if class_weight == 'balanced':
                            # Calculate balanced class weights
                            class_counts = np.bincount(y_train)
                            total_samples = len(y_train)
                            weight_for_0 = total_samples / (2 * class_counts[0])
                            weight_for_1 = total_samples / (2 * class_counts[1])
                            
                            # Set sample weights
                            sample_weights = np.ones(len(y_train))
                            sample_weights[y_train == 0] = weight_for_0
                            sample_weights[y_train == 1] = weight_for_1
                        elif isinstance(class_weight, dict):
                            # Convert class weight dictionary to sample weights
                            sample_weights = np.ones(len(y_train))
                            for class_val, weight in class_weight.items():
                                sample_weights[y_train == class_val] = weight
                    
                    # Create model
                    xgb_model = xgb.XGBClassifier(**model_params)
                    
                    # Create and fit pipeline
                    pipeline = make_pipeline(StandardScaler(), xgb_model)
                    
                    if sample_weights is not None:
                        pipeline.fit(X_train, y_train, xgbclassifier__sample_weight=sample_weights)
                    else:
                        pipeline.fit(X_train, y_train)
                    
                    # Predict probabilities
                    y_pred_proba = pipeline.predict_proba(X_test)
                    
                    # Convert to binary predictions using threshold 0.5
                    y_pred = (y_pred_proba[:, 1] > 0.5).astype(int)
                    
                    # Calculate accuracy
                    score = (y_pred == y_test).mean()
                    scores.append(score)
                
                return np.mean(scores)
            
            # Use TimeSeriesSplit for cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            # Perform custom cross-validation
            score = custom_cv_score(None, X, y, tscv)
            
            # Return the mean score
            return score
            
        except ImportError:
            # If XGBoost is not available, return a bad score
            return 0.0
        except Exception as e:
            # Log any errors for debugging
            print(f"Error in optimization: {e}")
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

def save_hyperparameters(params, model_type, path=config.HYPERPARAMS_DIR):
    """
    Save hyperparameters to disk.
    
    Parameters:
    -----------
    params : dict
        Hyperparameters
    model_type : str
        Model type ('decision_tree', 'random_forest', 'xgboost')
    path : str
        Path to save hyperparameters (default: config.HYPERPARAMS_DIR)
    """
    # Create directory if it doesn't exist
    os.makedirs(path, exist_ok=True)
    
    # Create filename
    filename = os.path.join(path, f'{model_type}_hyperparameters.pkl')
    
    # Save hyperparameters
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
    
    print(f"Hyperparameters saved to {filename}")

def load_hyperparameters(model_type, path=config.HYPERPARAMS_DIR):
    """
    Load hyperparameters from disk.
    
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
    # Create filename
    filename = os.path.join(path, f'{model_type}_hyperparameters.pkl')
    
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Hyperparameters file not found: {filename}")
    
    # Load hyperparameters
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    
    return params