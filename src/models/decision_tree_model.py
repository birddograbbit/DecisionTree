"""
Decision Tree model implementation.
"""

import os
import pickle
import numpy as np
import pandas as pd
import logging
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from .base_model import BaseModel

logger = logging.getLogger(__name__)

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
            The base decision tree classifier
        """
        return self._base

    def __init__(self, calibrate=False, max_depth=6, min_samples_split=100, min_samples_leaf=20,
                 max_features=None, criterion='gini', random_state=42,
                 ccp_alpha=0.0001, class_weight='balanced'):
        """
        Initialize the Decision Tree model.
        
        Parameters:
        -----------
        calibrate : bool, default=False
            Whether to use probability calibration
        max_depth : int or None, default=6
            Maximum depth of the tree
        min_samples_split : int, default=100
            Minimum samples required to split an internal node
        min_samples_leaf : int, default=20
            Minimum samples required at a leaf node
        max_features : int, float, str, or None, default=None
            Number of features to consider for best split
        criterion : str, default='gini'
            Function to measure the quality of a split
        random_state : int, default=42
            Random seed for reproducibility
        ccp_alpha : float, default=0.0001
            Complexity parameter used for Minimal Cost-Complexity Pruning
        class_weight : dict or 'balanced', default='balanced'
            Weights associated with classes to handle imbalance
        """
        self.params = {
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'criterion': criterion,
            'random_state': random_state,
            'ccp_alpha': ccp_alpha,
            'class_weight': class_weight
        }
        self.calibrate = calibrate
        self._base = DecisionTreeClassifier(**self.params)
        self._clf = None  # Will hold CalibratedClassifierCV or DecisionTreeClassifier
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

        if self.calibrate:
            # Calibrate probabilities using sigmoid (Platt scaling) with time-aware CV
            cv_split = TimeSeriesSplit(n_splits=5)
            self._clf = CalibratedClassifierCV(
                self._base, method="sigmoid", cv=cv_split
            )
            self._clf.fit(X, y)
            estimator = self._clf.calibrated_classifiers_[0].estimator
        else:
            self._clf = self._base
            self._clf.fit(X, y)
            estimator = self._clf

        if hasattr(estimator, "get_depth"):
            logger.info(
                "Trained decision tree depth: %d, leaves: %d",
                estimator.get_depth(),
                estimator.get_n_leaves(),
            )

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
        # Determine the fitted estimator
        if isinstance(self._clf, CalibratedClassifierCV):
            if hasattr(self._clf, 'calibrated_classifiers_') and self._clf.calibrated_classifiers_:
                estimator = self._clf.calibrated_classifiers_[0].estimator
            else:
                estimator = self._clf.estimator
        else:
            estimator = self._clf

        if estimator is None or not hasattr(estimator, "feature_importances_"):
            raise ValueError("Model has not been trained yet or does not expose feature_importances_.")

        importances = estimator.feature_importances_

        if self.feature_names is None:
            return importances
        else:
            # Return dictionary mapping feature names to importance scores
            return dict(zip(self.feature_names, importances))

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
        DecisionTreeModel
            Loaded model instance
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        # Ensure the loaded object is a DecisionTreeModel
        if not isinstance(model, cls):
            raise TypeError(f"Loaded model is not a {cls.__name__}")
        
        return model
        
    def __str__(self):
        """String representation of the model."""
        calib_str = ", calibrated" if self.calibrate else ""
        return f"DecisionTreeModel(max_depth={self.params['max_depth']}, " \
               f"min_samples_split={self.params['min_samples_split']}{calib_str})"

