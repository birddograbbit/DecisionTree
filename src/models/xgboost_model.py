"""
XGBoost model implementation with focal loss support.

This implementation supports both standard XGBoost and focal loss via imbalance-xgboost
for better handling of class imbalance in trading scenarios.
"""

import os
import pickle
import numpy as np
import pandas as pd

# Focal loss implementation (no external dependency needed)
FOCAL_LOSS_AVAILABLE = True  # We'll implement it ourselves

def focal_loss_objective(alpha=0.25, gamma=2.0):
    """
    Create a focal loss objective function for XGBoost.
    
    Focal loss helps with class imbalance by down-weighting easy examples
    and focusing on hard negatives.
    
    Parameters:
    -----------
    alpha : float
        Balancing parameter (typically set to inverse class frequency)
    gamma : float
        Focusing parameter (higher values focus more on hard examples)
        
    Returns:
    --------
    callable
        Objective function for XGBoost
    """
    def objective(y_true, y_pred):
        # Convert predictions to probabilities using sigmoid
        p = 1.0 / (1.0 + np.exp(-y_pred))
        
        # Focal loss gradient calculation
        # For y=1: gradient = alpha * (1-p)^gamma * (gamma * p * log(p) + p - 1)
        # For y=0: gradient = (1-alpha) * p^gamma * (gamma * (1-p) * log(1-p) - p)
        
        # Calculate pt (probability of true class)
        pt = p * y_true + (1 - p) * (1 - y_true)
        
        # Calculate alpha_t (class weight)
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        
        # Gradient calculation
        # Avoid log(0) by clipping probabilities
        p_clip = np.clip(p, 1e-8, 1 - 1e-8)
        
        # Standard cross-entropy gradient
        ce_grad = y_true - p
        
        # Focal term modulation
        focal_term = alpha_t * (1 - pt) ** gamma
        
        # Final gradient
        grad = -focal_term * ce_grad
        
        # Hessian (second derivative)
        # For stability, use a simplified hessian
        hess = focal_term * p * (1 - p) + 1e-8
        
        return grad, hess
    
    return objective

# Import standard XGBoost as fallback
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
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
import warnings


class XGBoostModel(BaseModel, BaseEstimator, ClassifierMixin):
    """
    XGBoost implementation with focal loss support.
    
    This implementation supports both standard XGBoost and focal loss via imbalance-xgboost
    for better handling of class imbalance. Falls back to standard XGBoost if focal loss
    is not available or not requested.
    """

    @property
    def model(self):
        """Return the underlying XGBClassifier."""
        return self._base

    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, gamma=0, 
                 objective='binary:logistic', random_state=42, n_jobs=-1,
                 class_weight=None, min_child_weight=1, reg_alpha=0, reg_lambda=1, 
                 scale_pos_weight=1, use_focal_loss=False, focal_gamma=2.0, 
                 focal_alpha=0.25, **kwargs):
        """
        Initialize the XGBoost model with optional focal loss support.
        
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
        use_focal_loss : bool, default=False
            Whether to use focal loss (requires imbalance-xgboost)
        focal_gamma : float, default=2.0
            Focal loss gamma parameter (focusing parameter)
            Higher values focus more on hard examples
        focal_alpha : float, default=0.25
            Focal loss alpha parameter (balance parameter)
            Controls class weight in focal loss
        kwargs : dict
            Additional parameters to pass to XGBClassifier
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not installed. Cannot create XGBoostModel.")
        
        # Store focal loss parameters
        self.use_focal_loss = use_focal_loss and FOCAL_LOSS_AVAILABLE
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
        # No warning needed since we implement focal loss ourselves
            
        # Setup parameters
        if self.use_focal_loss:
            # For focal loss, we'll use custom objective
            self.params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'gamma': gamma,
                'disable_default_eval_metric': True,  # Disable default metric for custom objective
                'random_state': random_state,
                'n_jobs': n_jobs,
                'min_child_weight': min_child_weight,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda
            }
            # Note: scale_pos_weight not used with focal loss
            print(f"Using focal loss with gamma={focal_gamma}, alpha={focal_alpha}")
        else:
            # Standard XGBoost parameters
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
        self._base = XGBClassifier(**self.params)
            
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
        
        # Handle class imbalance differently for focal loss vs standard XGBoost
        if self.use_focal_loss:
            # Calculate alpha based on class distribution if needed
            if self.focal_alpha == 'auto' or self.class_weight == 'balanced':
                class_counts = np.bincount(y)
                if len(class_counts) >= 2:
                    # Alpha should be minority class frequency for focal loss
                    self.focal_alpha = class_counts[1] / (class_counts[0] + class_counts[1])
                    print(f"Automatically set focal_alpha to {self.focal_alpha:.3f} based on class distribution")
            
            # Set custom objective function
            focal_obj = focal_loss_objective(alpha=self.focal_alpha, gamma=self.focal_gamma)
            self._base.set_params(objective=focal_obj)
            # Also set evaluation metric
            self._base.set_params(eval_metric='logloss')
        else:
            # Standard XGBoost class balancing
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