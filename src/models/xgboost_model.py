"""
XGBoost model implementation.
"""

import os
import pickle
import numpy as np
import pandas as pd
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost is not installed. XGBoostModel will not be available.")
    print("To install XGBoost: pip install xgboost")

from .base_model import BaseModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class XGBoostModel(BaseModel):
    """
    XGBoost implementation of the BaseModel interface.
    
    This class wraps XGBoost's XGBClassifier to conform
    to our BaseModel interface.
    """

    @property
    def model(self):
        """Return the underlying XGBClassifier."""
        return self._base

    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, gamma=0, 
                 objective='binary:logistic', random_state=42, n_jobs=-1):
        """
        Initialize the XGBoost model.
        
        Parameters:
        -----------
        n_estimators : int, default=100
            Number of gradient boosted trees
        max_depth : int, default=5
            Maximum tree depth for base learners
        learning_rate : float, default=0.1
            Boosting learning rate
        subsample : float, default=0.8
            Subsample ratio of the training instances
        colsample_bytree : float, default=0.8
            Subsample ratio of columns when constructing each tree
        gamma : float, default=0
            Minimum loss reduction required to make a further partition
        objective : str, default='binary:logistic'
            Learning task objective
        random_state : int, default=42
            Random seed for reproducibility
        n_jobs : int, default=-1
            Number of parallel threads used to run xgboost
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Cannot create XGBoostModel.")
            
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'gamma': gamma,
            'objective': objective,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        self._base = xgb.XGBClassifier(**self.params)
        self._clf = None  # Will hold the pipeline
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
        
        # Build pipeline with scaling
        self._clf = make_pipeline(StandardScaler(), self._base)
        self._clf.fit(X, y)
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
        XGBoostModel
            Loaded model instance
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Cannot load XGBoostModel.")
            
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        # Ensure the loaded object is an XGBoostModel
        if not isinstance(model, cls):
            raise TypeError(f"Loaded model is not a {cls.__name__}")
        
        return model
        
    def __str__(self):
        """String representation of the model."""
        return f"XGBoostModel(n_estimators={self.params['n_estimators']}, " \
               f"max_depth={self.params['max_depth']}, " \
               f"learning_rate={self.params['learning_rate']})"

