"""
Model engine for training, evaluating, and managing models.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.models.model_factory import ModelFactory

class ModelEngine:
    """
    Engine for training and managing models.
    
    This class handles model creation, training, evaluation, and persistence,
    abstracting away the details of specific model implementations.
    """

    def __init__(self, model_type="random_forest", model_params=None, model_object=None):
        """
        Initialize the model engine.
        
        Parameters:
        -----------
        model_type : str, default="random_forest"
            Type of model to use
        model_params : dict, default=None
            Parameters to pass to the model constructor
        model_object : BaseModel, default=None
            Pre-instantiated model object (alternative to model_type)
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.feature_names = None
        self.metrics = {}
        
        # Use provided model or create one
        if model_object is not None:
            self.model = model_object
        else:
            self.model = ModelFactory.create_model(model_type, **self.model_params)
    
    def train(self, X, y, cross_validation=True, cv=5, perform_hpo=False, 
              hpo_param_grid=None, hpo_cv=3, hpo_scoring='accuracy'):
        """
        Train the model on given data.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
        y : pd.Series or np.ndarray
            Target values
        cross_validation : bool, default=True
            Whether to perform cross-validation
        cv : int, default=5
            Number of cross-validation folds
        perform_hpo : bool, default=False
            Whether to perform hyperparameter optimization
        hpo_param_grid : dict, default=None
            Parameter grid for hyperparameter optimization
        hpo_cv : int, default=3
            Number of cross-validation folds for hyperparameter optimization
        hpo_scoring : str, default='accuracy'
            Scoring metric for hyperparameter optimization
            
        Returns:
        --------
        self
            For method chaining
        """
        # Store feature names if available
        self.feature_names = X.columns if hasattr(X, 'columns') else None
        
        # Convert to numpy arrays if needed
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        # Perform hyperparameter optimization if requested
        if perform_hpo and hpo_param_grid is not None:
            self._perform_hpo(X_array, y_array, hpo_param_grid, hpo_cv, hpo_scoring)
        
        # Train model
        self.model.train(X_array, y_array)
        
        # Perform cross-validation if requested
        if cross_validation:
            cv_scores = self._cross_validate(X_array, y_array, cv)
            self.metrics['cv_scores'] = cv_scores
            self.metrics['cv_mean'] = np.mean(cv_scores)
            self.metrics['cv_std'] = np.std(cv_scores)
            
            print(f"Cross-validation accuracy: {self.metrics['cv_mean']:.4f} ± {self.metrics['cv_std']:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Generate predictions for given features.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
            
        Returns:
        --------
        np.ndarray
            Predicted probabilities
        """
        # Convert to numpy array if needed
        X_array = X.values if hasattr(X, 'values') else X
        
        # Make predictions
        return self.model.predict(X_array)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance on given data.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
        y : pd.Series or np.ndarray
            Target values
            
        Returns:
        --------
        dict
            Performance metrics
        """
        # Convert to numpy arrays if needed
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        # Make predictions
        y_pred_proba = self.predict(X_array)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {}
        metrics['accuracy'] = accuracy_score(y_array, y_pred)
        metrics['precision'] = precision_score(y_array, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_array, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_array, y_pred, zero_division=0)
        
        # Only calculate AUC if we have binary predictions
        if len(np.unique(y_array)) == 2:
            metrics['auc'] = roc_auc_score(y_array, y_pred_proba)
        else:
            metrics['auc'] = None
        
        # Store metrics
        self.metrics.update(metrics)
        
        return metrics
    
    def get_feature_importance(self):
        """
        Get feature importances from the model.
        
        Returns:
        --------
        pd.Series or dict
            Feature importances
        """
        # Get raw feature importances
        importances = self.model.get_feature_importance()
        
        # If feature names are available, create a Series
        if self.feature_names is not None and isinstance(importances, np.ndarray):
            return pd.Series(importances, index=self.feature_names)
        
        return importances
    
    def save(self, path):
        """
        Save model to disk.
        
        Parameters:
        -----------
        path : str
            Path to save model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """
        Load model from disk.
        
        Parameters:
        -----------
        path : str
            Path to saved model
            
        Returns:
        --------
        self
            For method chaining
        """
        # Get model class for loading
        model_class = type(self.model)
        
        # Load model
        self.model = model_class.load(path)
        print(f"Model loaded from {path}")
        
        return self
    
    def _cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        cv : int, default=5
            Number of cross-validation folds
            
        Returns:
        --------
        np.ndarray
            Cross-validation scores
        """
        # Get sklearn model if available
        if hasattr(self.model, 'model'):
            sklearn_model = self.model.model
        else:
            # Create a temporary sklearn-compatible wrapper
            from sklearn.base import BaseEstimator, ClassifierMixin
            
            class ModelWrapper(BaseEstimator, ClassifierMixin):
                def __init__(self, model):
                    self.model = model
                
                def fit(self, X, y):
                    self.model.train(X, y)
                    return self
                
                def predict(self, X):
                    return (self.model.predict(X) >= 0.5).astype(int)
            
            sklearn_model = ModelWrapper(self.model)
        
        # Perform cross-validation while ignoring runtime warnings that can
        # occur with small sample sizes or constant features
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            scores = cross_val_score(sklearn_model, X, y, cv=cv, scoring='accuracy')
        return scores
    
    def _perform_hpo(self, X, y, param_grid, cv=3, scoring='accuracy'):
        """
        Perform hyperparameter optimization.
        
        Parameters:
        -----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        param_grid : dict
            Parameter grid for hyperparameter optimization
        cv : int, default=3
            Number of cross-validation folds
        scoring : str, default='accuracy'
            Scoring metric
        """
        # This is a placeholder for HPO implementation
        # In a real implementation, this would use GridSearchCV or another HPO method
        print("Hyperparameter optimization not implemented yet.")
        pass
