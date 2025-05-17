"""
Random Forest model implementation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """
    Random Forest implementation of the BaseModel interface.
    
    This class wraps sklearn's RandomForestClassifier to conform
    to our BaseModel interface.
    """

    @property
    def model(self):
        """
        Property that returns the base model.
        Maintained for backward compatibility with the model_engine.
        
        Returns:
        --------
        sklearn.ensemble.RandomForestClassifier
            The base random forest classifier
        """
        return self._base

    def __init__(self, calibrate=False, n_estimators=100, max_depth=5, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', criterion='gini', 
                 random_state=42, n_jobs=-1, **kwargs):
        """
        Initialize the Random Forest model.
        
        Parameters:
        -----------
        calibrate : bool, default=False
            Whether to use probability calibration
        n_estimators : int, default=100
            Number of trees in the forest
        max_depth : int or None, default=5
            Maximum depth of the trees
        min_samples_split : int, default=2
            Minimum samples required to split an internal node
        min_samples_leaf : int, default=1
            Minimum samples required at a leaf node
        max_features : int, float, str, or None, default='sqrt'
            Number of features to consider for best split
        criterion : str, default='gini'
            Function to measure the quality of a split
        random_state : int, default=42
            Random seed for reproducibility
        n_jobs : int, default=-1
            Number of jobs to run in parallel (-1 means using all processors)
        **kwargs : dict
            Additional keyword arguments to pass to RandomForestClassifier
        """
        # Combine default/explicit parameters with any overriding kwargs
        # kwargs take precedence
        current_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'criterion': criterion,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        current_params.update(kwargs) # Update with any kwargs, including new ones like class_weight

        self.params = current_params # Store all actual params used
        self.calibrate = calibrate
        self._base = RandomForestClassifier(**self.params)
        self._clf = None
        self.feature_names = None

    def train(self, X, y):
        """
        Train the model on given data.
        
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
        # Store feature names if available (for feature importance)
        self.feature_names = X.columns if hasattr(X, 'columns') else None
        
        # Train the model
        if self.calibrate:
            # First train the base estimator
            self._base.fit(X, y)
            # Wrap it in isotonic calibration
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
        Generate predictions for given features.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
            
        Returns:
        --------
        np.ndarray
            Predicted probabilities for positive class (class 1)
        """
        return self._clf.predict_proba(X)[:, 1]  # Probability of positive class

    def get_feature_importance(self):
        """
        Return feature importance scores.
        
        Returns:
        --------
        dict or np.ndarray
            Feature importance scores
        """
        if self.feature_names is None:
            return self._base.feature_importances_
        else:
            # Return dictionary mapping feature names to importance scores
            return dict(zip(self.feature_names, self._base.feature_importances_))

    def save(self, path):
        """
        Save model to disk.
        
        Parameters:
        -----------
        path : str
            Path to save model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save the entire model instance
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Model saved to {path}")

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
        RandomForestModel
            Loaded model instance
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        # Ensure the loaded object is a RandomForestModel
        if not isinstance(model, cls):
            raise TypeError(f"Loaded model is not a {cls.__name__}")
        
        return model
        
    def __str__(self):
        """String representation of the model."""
        calib_str = ", calibrated" if self.calibrate else ""
        return f"RandomForestModel(n_estimators={self.params['n_estimators']}, " \
               f"max_depth={self.params['max_depth']}{calib_str})"
