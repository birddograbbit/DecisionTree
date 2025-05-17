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
    
    This model implements stacking, a technique that combines multiple base models
    by training a meta-learner on the predictions of the base models.
    """
    
    def __init__(self, base_models=None, n_folds=5):
        """
        Initialize the stacking model.
        
        Parameters:
        -----------
        base_models : list, default=None
            List of model objects
            If None, will use default models
        n_folds : int, default=5
            Number of folds for cross-validation when generating meta-features
        """
        super().__init__()
        self.base_models = base_models if base_models is not None else []
        self.n_folds = n_folds
        self.meta_learner = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver='lbfgs')
        )
        
    def train(self, X, y):
        """
        Train the stacking model.
        
        This involves:
        1. Training each base model on the entire dataset
        2. Generating meta-features via cross-validation
        3. Training the meta-learner on the meta-features
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Target values
            
        Returns:
        --------
        self
        """
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Train each base model on the entire dataset
        for model in self.base_models:
            model.train(X, y)
        
        # Generate meta-features via cross-validation
        meta_features = self._generate_meta_features(X, y)
        
        # Train meta-learner on meta-features
        self.meta_learner.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        """
        Generate predictions from the stacking model.
        
        Parameters:
        -----------
        X : array-like
            Features
            
        Returns:
        --------
        array-like
            Predicted probabilities
        """
        # Convert to numpy array
        X = np.array(X)
        
        # Generate predictions from base models
        base_preds = self._predict_base_models(X)
        
        # Generate final predictions using meta-learner
        return self.meta_learner.predict_proba(base_preds)[:, 1]
    
    def predict_proba(self, X):
        """
        Generate probability predictions.
        
        Parameters:
        -----------
        X : array-like
            Features
            
        Returns:
        --------
        array-like
            Predicted probabilities
        """
        return self.predict(X)
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        StackingModel
            Loaded model
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def _generate_meta_features(self, X, y):
        """
        Generate meta-features via cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target values
            
        Returns:
        --------
        array-like
            Meta-features
        """
        # Initialize meta-features array
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        # Use KFold cross-validation to generate meta-features
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # For each fold
        for train_idx, val_idx in kf.split(X):
            # Train each base model on the training set
            for i, model in enumerate(self.base_models):
                model_copy = model.__class__(**model.__dict__)
                model_copy.train(X[train_idx], y[train_idx])
                
                # Generate predictions on the validation set
                meta_features[val_idx, i] = model_copy.predict(X[val_idx])
        
        return meta_features
    
    def _predict_base_models(self, X):
        """
        Generate predictions from all base models.
        
        Parameters:
        -----------
        X : array-like
            Features
            
        Returns:
        --------
        array-like
            Base model predictions
        """
        # Initialize predictions array
        preds = np.zeros((X.shape[0], len(self.base_models)))
        
        # Generate predictions from each base model
        for i, model in enumerate(self.base_models):
            preds[:, i] = model.predict(X)
        
        return preds