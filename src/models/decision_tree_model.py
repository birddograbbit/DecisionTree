"""
Decision Tree model implementation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from .base_model import BaseModel

class DecisionTreeModel(BaseModel):
    """
    Decision Tree implementation of the BaseModel interface.
    
    This class wraps sklearn's DecisionTreeClassifier to conform
    to our BaseModel interface.
    """

    @property
    def model(self):
        """
        Property that returns the base model.
        Maintained for backward compatibility with the model_engine.
        
        Returns:
        --------
        sklearn.tree.DecisionTreeClassifier
            The base DecisionTreeClassifier
        """
        return self._base

    def __init__(self, calibrate=False, max_depth=5, min_samples_split=2, min_samples_leaf=1, 
                 max_features=None, criterion='gini', random_state=42):
        """
        Initialize the Decision Tree model.
        
        Parameters:
        -----------
        calibrate : bool, default=False
            Whether to apply probability calibration to the model
        max_depth : int, default=5
            Maximum depth of the tree
        min_samples_split : int, default=2
            Minimum samples required to split a node
        min_samples_leaf : int, default=1
            Minimum samples required at a leaf node
        max_features : str or int, default=None
            Number of features to consider when looking for the best split
        criterion : str, default='gini'
            Function to measure the quality of a split
        random_state : int, default=42
            Random state for reproducibility
        """
        super().__init__()
        self.params = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'criterion': criterion,
            'random_state': random_state
        }
        self.calibrate = calibrate
        self._base = DecisionTreeClassifier(**self.params)
        self._clf = None
        
    def train(self, X, y):
        """
        Train the Decision Tree model.
        
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
        if self.calibrate:
            # first train the base estimator
            self._base.fit(X, y)
            # wrap it in isotonic or sigmoid calibration
            self._clf = CalibratedClassifierCV(
                self._base, method="isotonic", cv=5
            )
            self._clf.fit(X, y)
        else:
            self._base.fit(X, y)
            self._clf = self._base
        return self
    
    def predict(self, X):
        """
        Generate binary predictions.
        
        Parameters:
        -----------
        X : array-like
            Features
            
        Returns:
        --------
        array-like
            Predicted class probabilities for the positive class
        """
        return self.predict_proba(X)
    
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
            Predicted probabilities for the positive class
        """
        # always route through the calibrated object
        return self._clf.predict_proba(X)[:, 1]
    
    def get_feature_importances(self):
        """
        Get feature importances.
        
        Returns:
        --------
        array-like
            Feature importances
        """
        return self._base.feature_importances_
    
    def save(self, filepath):
        """
        Save the model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
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
        DecisionTreeModel
            Loaded model
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)