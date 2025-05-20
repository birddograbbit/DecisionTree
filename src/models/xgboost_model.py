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
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

class ModelAdapter(BaseEstimator, ClassifierMixin):
    """
    Adapter class to make a XGBoost Booster object compatible with scikit-learn.
    
    This class wraps a XGBoost Booster and provides the scikit-learn estimator interface,
    including methods like fit, predict, and predict_proba.
    """
    
    def __init__(self, booster=None, use_sigmoid=True, scaler=None):
        """
        Initialize the adapter.
        
        Parameters:
        -----------
        booster : xgboost.Booster or None
            Pretrained XGBoost booster (default: None)
        use_sigmoid : bool
            Whether to apply sigmoid to get probabilities (default: True)
        scaler : sklearn.preprocessing.StandardScaler or None
            Fitted scaler to preprocess input data (default: None)
        """
        self.booster = booster
        self.use_sigmoid = use_sigmoid
        self.scaler = scaler
        print("Using ModelAdapter for scikit-learn compatibility")
        
    def fit(self, X, y=None, **kwargs):
        """
        Fit method for compatibility with scikit-learn.
        
        If scaler is provided, it will be fitted on X.
        
        Parameters:
        -----------
        X : pd.DataFrame or numpy.ndarray
            Feature matrix to fit the scaler (if provided)
        y : Any
            Not used for the booster (already trained)
        kwargs : dict
            Additional arguments
            
        Returns:
        --------
        self
            For method chaining
        """
        # If scaler is provided but not fitted, fit it now
        if self.scaler is not None and not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(X)
        return self
    
    def predict_proba(self, X):
        """
        Generate probability predictions.
        
        Parameters:
        -----------
        X : pd.DataFrame or numpy.ndarray
            Feature matrix
            
        Returns:
        --------
        numpy.ndarray
            Array of shape (n_samples, 2) with probabilities for both classes
        """
        # Apply scaling if scaler is available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        # Convert to DMatrix
        dX = xgb.DMatrix(X)
        
        # Get raw predictions
        raw_preds = self.booster.predict(dX)
        
        # Apply sigmoid to get probabilities
        if self.use_sigmoid:
            probs = 1.0 / (1.0 + np.exp(-raw_preds))
        else:
            probs = raw_preds  # Assume these are already probabilities
            
        # Return probabilities for both classes
        return np.vstack((1 - probs, probs)).T
    
    def predict(self, X):
        """
        Generate class predictions.
        
        Parameters:
        -----------
        X : pd.DataFrame or numpy.ndarray
            Feature matrix
            
        Returns:
        --------
        numpy.ndarray
            Class predictions (0 or 1)
        """
        # Get probabilities for positive class
        probs = self.predict_proba(X)[:, 1]
        
        # Convert to class predictions
        return (probs > 0.5).astype(int)

class FocalLoss:
    """
    Focal Loss implementation for XGBoost.
    
    Focal Loss is a modified version of Cross-Entropy Loss that reduces
    the relative loss for well-classified examples and focuses more on
    difficult examples.
    
    Parameters:
    -----------
    gamma : float
        Focusing parameter (default: 2.0)
        Controls how much to down-weight easy examples
    alpha : float
        Class balancing parameter (default: 0.25)
        Controls the weight of the rare class
    """
    
    def __init__(self, gamma=2.0, alpha=0.25):
        self.gamma = gamma
        self.alpha = alpha
    
    def __call__(self, y_pred, y_true):
        """
        Calculate focal loss gradient and hessian.
        
        Parameters:
        -----------
        y_pred : numpy.ndarray
            Predictions (probabilities after sigmoid)
        y_true : xgboost.DMatrix
            Ground truth labels
            
        Returns:
        --------
        tuple
            (Gradient, Hessian)
        """
        # Extract true labels
        y = y_true.get_label()
        
        # Convert to numpy array
        y_pred = np.array(y_pred)
        
        # Apply sigmoid to get probabilities
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        
        # Calculate p_t
        p_t = y_pred * y + (1 - y_pred) * (1 - y)
        
        # Calculate alpha_t
        alpha_t = self.alpha * y + (1 - self.alpha) * (1 - y)
        
        # Calculate focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Calculate gradient
        grad = -focal_weight * (y - y_pred)
        
        # Calculate hessian (second derivative)
        hess = focal_weight * y_pred * (1 - y_pred)
        
        return grad, hess

class XGBoostModel(BaseModel, BaseEstimator, ClassifierMixin):
    """
    XGBoost implementation of the BaseModel interface.
    
    This class wraps XGBoost's XGBClassifier to conform
    to our BaseModel interface and scikit-learn's estimator interface.
    """

    @property
    def model(self):
        """Return the underlying XGBClassifier."""
        return self._base

    def __init__(self, n_estimators=100, max_depth=5, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, gamma=0, 
                 objective='binary:logistic', random_state=42, n_jobs=-1,
                 class_weight=None, use_focal_loss=False, 
                 focal_gamma=2.0, focal_alpha=0.25,
                 min_child_weight=1, reg_alpha=0, reg_lambda=1, 
                 scale_pos_weight=1, **kwargs):
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
        class_weight : dict or 'balanced', default=None
            Weights associated with classes
            If 'balanced', class weights are automatically calculated
        use_focal_loss : bool, default=False
            Whether to use focal loss for imbalanced classes
        focal_gamma : float, default=2.0
            Focusing parameter for focal loss
        focal_alpha : float, default=0.25
            Class balancing parameter for focal loss
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
        
        # Store additional parameters for class imbalance handling
        self.class_weight = class_weight
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        
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
        # Use our existing train method
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
        
        # Handle class imbalance
        if self.use_focal_loss:
            # Create and fit a scaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Use custom focal loss objective function
            dtrain = xgb.DMatrix(X_scaled, label=y)
            focal_loss_obj = FocalLoss(gamma=self.focal_gamma, alpha=self.focal_alpha)
            
            # Update parameters for custom objective
            params = self.params.copy()
            params.pop('n_estimators', None)  # Remove n_estimators (used in fit)
            params.pop('objective', None)     # Remove default objective
            
            # Train with custom objective
            booster = xgb.train(
                params=params,
                dtrain=dtrain,
                num_boost_round=self.params['n_estimators'],
                obj=focal_loss_obj
            )
            
            # Create a ModelAdapter with the fitted scaler
            adapter = ModelAdapter(booster=booster, use_sigmoid=True, scaler=None)
            
            # Store the booster in base
            self._base = booster
            
            # Create a pipeline with the scaler and adapter for consistency
            # This ensures self._clf is always a pipeline
            self._clf = make_pipeline(scaler, adapter)
            
            # Fit the pipeline to ensure the scaler is fitted
            # The adapter doesn't need fitting as it already has the trained booster
            self._clf.fit(X, y)
            
            return self
            
        elif self.class_weight == 'balanced':
            # Calculate balanced class weights
            class_counts = np.bincount(y)
            total_samples = len(y)
            weight_for_0 = total_samples / (2 * class_counts[0])
            weight_for_1 = total_samples / (2 * class_counts[1])
            
            # Set sample weights
            sample_weights = np.ones(len(y))
            sample_weights[y == 0] = weight_for_0
            sample_weights[y == 1] = weight_for_1
            
            # Build pipeline with scaling
            self._clf = make_pipeline(StandardScaler(), self._base)
            self._clf.fit(X, y, xgbclassifier__sample_weight=sample_weights)
            
        elif isinstance(self.class_weight, dict):
            # Convert class weight dictionary to sample weights
            sample_weights = np.ones(len(y))
            for class_val, weight in self.class_weight.items():
                sample_weights[y == class_val] = weight
            
            # Build pipeline with scaling
            self._clf = make_pipeline(StandardScaler(), self._base)
            self._clf.fit(X, y, xgbclassifier__sample_weight=sample_weights)
            
        else:
            # Standard training without class weights
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
        # Always use predict_proba from the pipeline
        return self._clf.predict_proba(X)[:, 1]  # Probability of positive class
        
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
        # Use the pipeline's predict_proba method
        return self._clf.predict_proba(X)

    def get_feature_importance(self):
        """
        Return feature importance scores.
        
        Returns:
        --------
        dict or np.ndarray
            Feature importance scores
        """
        # Determine the fitted estimator from the training pipeline
        estimator = None
        if isinstance(self._clf, Pipeline):
            estimator = self._clf.steps[-1][1]
        elif isinstance(self._clf, CalibratedClassifierCV):
            if hasattr(self._clf, 'calibrated_classifiers_') and self._clf.calibrated_classifiers_:
                base = self._clf.calibrated_classifiers_[0].estimator
            else:
                base = self._clf.estimator
            if isinstance(base, Pipeline):
                estimator = base.steps[-1][1]
            else:
                estimator = base
        elif self._base is not None:
            estimator = self._base

        if estimator is None:
            raise ValueError("Model has not been trained yet.")

        if self.feature_names is None:
            if self.use_focal_loss:
                # Get importance for custom objective
                return estimator.get_score(importance_type='gain')
            else:
                # Standard importance
                return estimator.feature_importances_
        else:
            if self.use_focal_loss:
                # Get feature importance for custom objective
                importances = estimator.get_score(importance_type='gain')
                
                # Convert feature index to feature names
                named_importances = {}
                for key, value in importances.items():
                    if key.startswith('f'):
                        # If key is f0, f1, etc., convert to feature name
                        try:
                            idx = int(key[1:])
                            if idx < len(self.feature_names):
                                named_importances[self.feature_names[idx]] = value
                        except ValueError:
                            named_importances[key] = value
                    else:
                        named_importances[key] = value
                        
                return named_importances
            else:
                # Return dictionary mapping feature names to importance scores
                return dict(zip(self.feature_names, estimator.feature_importances_))

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
               
        if self.use_focal_loss:
            base_str += f", focal_loss(gamma={self.focal_gamma}, alpha={self.focal_alpha})"
        elif self.class_weight:
            base_str += f", class_weight={self.class_weight}"
            
        return base_str