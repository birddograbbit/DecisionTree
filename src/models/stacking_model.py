"""
Stacking model implementation for combining multiple models.

This module provides a flexible stacking ensemble approach that can work with
any combination of base models and meta-models to potentially improve prediction
performance over any single model.
"""

import pickle
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel

class StackingModel(BaseModel):
    """
    Stacking ensemble model that combines multiple base models.
    
    This model implements stacking, a technique where multiple models (base models)
    are trained on the same data, and their predictions are used as features to train
    a meta-model, which makes the final prediction.
    """
    
    # Add a property to make it compatible with ModelEngine._cross_validate
    @property
    def model(self):
        """
        Property that returns the meta-model as a scikit-learn estimator.
        
        Returns:
        --------
        estimator
            The scikit-learn estimator used for meta-modeling
        """
        # Return meta_model_sklearn if available, otherwise a default LogisticRegression
        if hasattr(self, 'meta_model_sklearn') and self.meta_model_sklearn is not None:
            return self.meta_model_sklearn
        else:
            # Create a default meta-learner pipeline
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
            )
    
    def __init__(self, base_models=None, meta_model=None, cv=5, use_features=False, 
                 meta_model_sklearn=None, meta_model_type='logistic_regression', meta_model_params=None):
        """
        Initialize the stacking model.
        
        Parameters:
        -----------
        base_models : list of BaseModel instances, default=None
            List of trained or untrained base models
        meta_model : BaseModel instance, default=None
            Meta-model to use (from BaseModel interface)
        cv : int, default=5
            Number of cross-validation folds for creating meta-features
        use_features : bool, default=False
            Whether to include original features along with model predictions
            for meta-model training
        meta_model_sklearn : scikit-learn model instance, default=None
            Direct scikit-learn model to use as meta-model (alternative to meta_model)
        meta_model_type : str, default='logistic_regression'
            Type of scikit-learn meta-model to use if meta_model and meta_model_sklearn are None
            Options: 'logistic_regression', 'random_forest', 'svm'
        meta_model_params : dict, default=None
            Parameters for scikit-learn meta-model if using meta_model_type
        """
        # Store configuration
        self.base_models = base_models or []
        self.meta_model = meta_model
        self.cv = cv
        self.use_features = use_features
        self.feature_names = None
        self.is_trained = False
        
        # ---- THE ONLY place we build the meta learner ----
        if self.meta_model is None:
            self.meta_model_params = meta_model_params or {}
            
            if meta_model_type == 'logistic_regression':
                params = {'C': 1.0, 'random_state': 42}
                params.update(self.meta_model_params)
                self.meta_model_sklearn = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1, **params)
                )
            elif meta_model_type == 'random_forest':
                from sklearn.ensemble import RandomForestClassifier
                params = {'n_estimators': 100, 'max_depth': 3, 'random_state': 42}
                params.update(self.meta_model_params)
                self.meta_model_sklearn = RandomForestClassifier(**params)
            elif meta_model_type == 'svm':
                from sklearn.svm import SVC
                params = {'probability': True, 'random_state': 42}
                params.update(self.meta_model_params)
                self.meta_model_sklearn = SVC(**params)
            else:
                raise ValueError(f"Unsupported meta_model_type: {meta_model_type}")
        else:
            self.meta_model_sklearn = meta_model_sklearn
                
        self.meta_model_type = meta_model_type
        
    def train(self, X, y):
        """
        Train the stacking model.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
        y : pd.Series or np.ndarray
            Target values
            
        Returns:
        --------
        self
            For method chaining
        """
        # Store feature names if available
        self.feature_names = X.columns if hasattr(X, 'columns') else None
        
        # Convert inputs to numpy arrays for easier handling
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        # Initialize array to store meta-features
        meta_features = np.zeros((X_array.shape[0], len(self.base_models)))
        
        # Create cross-validation folds
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        # Generate out-of-fold predictions for each base model
        for i, model in enumerate(self.base_models):
            # Initialize array to store oof predictions
            oof_preds = np.zeros(X_array.shape[0])
            
            # Generate oof predictions via cross-validation
            for train_idx, val_idx in kf.split(X_array):
                # Get train/validation split
                X_train, X_val = X_array[train_idx], X_array[val_idx]
                y_train = y_array[train_idx]
                
                # Train model on training fold
                model.train(X_train, y_train)
                
                # Make predictions on validation fold
                oof_preds[val_idx] = model.predict(X_val)
            
            # Store oof predictions as meta-features
            meta_features[:, i] = oof_preds
        
        # Now train base models on the entire dataset
        for model in self.base_models:
            model.train(X_array, y_array)
        
        # Prepare meta-features for meta-model
        if self.use_features:
            # Include original features
            meta_X = np.hstack([meta_features, X_array])
        else:
            # Use only model predictions
            meta_X = meta_features
        
        # Train meta-model
        if self.meta_model is not None:
            self.meta_model.train(meta_X, y_array)
        elif self.meta_model_sklearn is not None:
            # Use the meta-model created in __init__ - do NOT create a new one here
            self.meta_model_sklearn.fit(meta_X, y_array)
        else:
            raise ValueError("No meta-model available for training - this should never happen")
        
        self.is_trained = True
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
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        # Convert input to numpy array
        X_array = X.values if hasattr(X, 'values') else X
        
        # Generate base model predictions
        base_preds = np.zeros((X_array.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            base_preds[:, i] = model.predict(X_array)
        
        # Prepare features for meta-model
        if self.use_features:
            # Include original features
            meta_X = np.hstack([base_preds, X_array])
        else:
            # Use only model predictions
            meta_X = base_preds
        
        # Generate meta-model predictions
        if self.meta_model is not None:
            return self.meta_model.predict(meta_X)
        elif self.meta_model_sklearn is not None:
            if hasattr(self.meta_model_sklearn, 'predict_proba'):
                return self.meta_model_sklearn.predict_proba(meta_X)[:, 1]
            else:
                return self.meta_model_sklearn.predict(meta_X)
        else:
            raise ValueError("No meta-model available for prediction")
        
    def get_feature_importance(self):
        """
        Return feature importance scores.
        
        Returns:
        --------
        dict
            Feature importance scores for each base model and meta-model
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet.")
        
        # Get base model importances
        base_importances = []
        for i, model in enumerate(self.base_models):
            model_name = f"base_model_{i}"
            base_importances.append({
                'name': model_name,
                'importance': model.get_feature_importance()
            })
        
        # Get meta-model importances
        if self.meta_model is not None:
            meta_importance = self.meta_model.get_feature_importance()
        elif self.meta_model_sklearn is not None:
            # Try to get importances from sklearn model
            if hasattr(self.meta_model_sklearn, 'feature_importances_'):
                meta_importance = {
                    f'meta_feature_{i}': imp for i, imp in 
                    enumerate(self.meta_model_sklearn.feature_importances_)
                }
            elif hasattr(self.meta_model_sklearn, 'coef_'):
                meta_importance = {
                    f'meta_feature_{i}': abs(imp) for i, imp in 
                    enumerate(self.meta_model_sklearn.coef_[0])
                }
            else:
                meta_importance = {
                    f'meta_feature_{i}': 1.0/len(self.base_models) 
                    for i in range(len(self.base_models))
                }
        else:
            meta_importance = {}
        
        # Return combined importances
        return {
            'base_models': base_importances,
            'meta_model': meta_importance
        }
        
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
        
        # Save the entire model instance
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Stacking model saved to {path}")
        
    @classmethod
    def load(cls, path):
        """
        Load model from disk.
        
        Parameters:
        -----------
        path : str
            Path to saved model
            
        Returns:
        --------
        StackingModel
            Loaded model instance
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        # Ensure the loaded object is a StackingModel
        if not isinstance(model, cls):
            raise TypeError(f"Loaded model is not a {cls.__name__}")
        
        return model
    
    def __str__(self):
        """String representation of the model."""
        if self.meta_model is not None:
            meta_model_str = type(self.meta_model).__name__
        elif self.meta_model_sklearn is not None:
            meta_model_str = type(self.meta_model_sklearn).__name__
        else:
            meta_model_str = "None"
            
        return f"StackingModel(base_models={len(self.base_models)}, meta_model={meta_model_str}, cv={self.cv})"
