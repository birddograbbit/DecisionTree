"""
XGBoost model implementation - Simplified version.

This simplified implementation removes the complex ModelAdapter and FocalLoss
classes in favor of a straightforward XGBClassifier approach with built-in
class balancing capabilities.
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
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV


class XGBoostModel(BaseModel, BaseEstimator, ClassifierMixin):
    """
    Simplified XGBoost implementation of the BaseModel interface.
    
    This implementation uses standard XGBClassifier with built-in class balancing
    instead of complex custom objective functions, while maintaining full
    scikit-learn compatibility.
    """

    @property
    def model(self):
        """Return the underlying XGBClassifier."""
        return self._base

    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, gamma=0, 
                 objective='binary:logistic', random_state=42, n_jobs=-1,
                 class_weight=None, min_child_weight=1, reg_alpha=0, reg_lambda=1, 
                 scale_pos_weight=1, **kwargs):
        """
        Initialize the simplified XGBoost model.
        
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
        class_weight : dict or 'balanced', default=None
            Weights associated with classes
            If 'balanced', scale_pos_weight is automatically calculated
        min_child_weight : float, default=1
            Minimum sum of instance weight needed in a child
        reg_alpha : float, default=0
            L1 regularization term on weights
        reg_lambda : float, default=1
            L2 regularization term on weights
        scale_pos_weight : float, default=1
            Controls the balance of positive and negative weights
        kwargs : dict
            Additional parameters to pass to XGBClassifier
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
            'n_jobs': n_jobs,
            'min_child_weight': min_child_weight,
            'reg_alpha': reg_alpha,
            'reg_lambda': reg_lambda,
            'scale_pos_weight': scale_pos_weight
        }
        
        # Add any additional parameters
        for key, value in kwargs.items():
            self.params[key] = value
        
        # Store class weight handling preference
        self.class_weight = class_weight
        
        # Initialize base model
        self._base = xgb.XGBClassifier(**self.params)
        self._clf = None  # Will hold the pipeline
        self.feature_names = None
        
        # For scikit-learn compatibility, store all parameters as attributes
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.objective = objective
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight

    def fit(self, X, y, sample_weight=None):
        """
        Fit the model to the data (scikit-learn compatible method).
        
        This is a wrapper around the train method to make it compatible
        with scikit-learn's estimator interface.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
        y : pd.Series or np.ndarray
            Target values
        sample_weight : array-like or None, default=None
            Sample weights
            
        Returns:
        --------
        self
            For method chaining
        """
        return self.train(X, y)
        
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
        
        # Handle class imbalance with built-in XGBoost mechanisms
        if self.class_weight == 'balanced':
            # Calculate balanced scale_pos_weight
            class_counts = np.bincount(y)
            if len(class_counts) >= 2:
                scale_pos_weight = class_counts[0] / class_counts[1]
                # Update the base model parameter
                self._base.set_params(scale_pos_weight=scale_pos_weight)
                print(f"Automatically set scale_pos_weight to {scale_pos_weight:.3f} for balanced classes")
        
        elif isinstance(self.class_weight, dict):
            # Convert class weight dictionary to scale_pos_weight if possible
            if 0 in self.class_weight and 1 in self.class_weight:
                scale_pos_weight = self.class_weight[1] / self.class_weight[0]
                self._base.set_params(scale_pos_weight=scale_pos_weight)
                print(f"Set scale_pos_weight to {scale_pos_weight:.3f} based on class_weight dict")
        
        # Create pipeline with scaling and XGBoost
        self._clf = make_pipeline(StandardScaler(), self._base)
        
        # Train the model
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
        if self._clf is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Return probabilities for positive class
        return self._clf.predict_proba(X)[:, 1]
        
    def predict_proba(self, X):
        """
        Generate probability predictions (scikit-learn compatible method).
        
        Parameters:
        -----------
        X : pd.DataFrame or np.ndarray
            Feature matrix
            
        Returns:
        --------
        np.ndarray
            Array with probabilities for both classes
        """
        if self._clf is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        return self._clf.predict_proba(X)

    def get_feature_importance(self):
        """
        Return feature importance scores.
        
        Returns:
        --------
        dict or np.ndarray
            Feature importance scores
        """
        if self._clf is None:
            raise ValueError("Model has not been trained yet. Call train() first.")

        # Get the XGBoost estimator from the pipeline
        xgb_estimator = None
        if isinstance(self._clf, Pipeline):
            xgb_estimator = self._clf.steps[-1][1]
        elif isinstance(self._clf, CalibratedClassifierCV):
            if hasattr(self._clf, 'calibrated_classifiers_') and self._clf.calibrated_classifiers_:
                base = self._clf.calibrated_classifiers_[0].estimator
                if isinstance(base, Pipeline):
                    xgb_estimator = base.steps[-1][1]
                else:
                    xgb_estimator = base
            else:
                xgb_estimator = self._clf.estimator
                if isinstance(xgb_estimator, Pipeline):
                    xgb_estimator = xgb_estimator.steps[-1][1]
        else:
            xgb_estimator = self._base

        if xgb_estimator is None:
            raise ValueError("Could not find XGBoost estimator in pipeline.")

        # Get feature importance
        if hasattr(xgb_estimator, 'feature_importances_'):
            importances = xgb_estimator.feature_importances_
        else:
            raise ValueError("XGBoost estimator does not have feature_importances_ attribute.")

        # Return as dictionary with feature names if available
        if self.feature_names is not None:
            return dict(zip(self.feature_names, importances))
        else:
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
        base_str = f"XGBoostModel(n_estimators={self.params['n_estimators']}, " \
               f"max_depth={self.params['max_depth']}, " \
               f"learning_rate={self.params['learning_rate']})"
               
        if self.class_weight:
            base_str += f", class_weight={self.class_weight}"
            
        return base_str