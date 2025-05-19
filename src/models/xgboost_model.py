"""
XGBoost model implementation.
"""

import os
import pickle
import numpy as np
import pandas as pd
import logging
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

# Set up logging
logger = logging.getLogger(__name__)

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
                 subsample=0.8, colsample_bytree=0.8, gamma=0, min_child_weight=1,
                 objective='binary:logistic', scale_pos_weight=1.0, random_state=42, n_jobs=-1, 
                 focal_alpha=None, focal_gamma=None):
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
        min_child_weight : int, default=1
            Minimum sum of instance weight needed in a child
        objective : str, default='binary:logistic'
            Learning task objective
        scale_pos_weight : float, default=1.0
            Control the balance of positive and negative weights (for class imbalance)
        random_state : int, default=42
            Random seed for reproducibility
        n_jobs : int, default=-1
            Number of parallel threads used to run xgboost
        focal_alpha : float, default=None
            Alpha parameter for focal loss (if using binary:logitraw objective and focal loss)
        focal_gamma : float, default=None
            Gamma parameter for focal loss (if using binary:logitraw objective and focal loss)
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
            'min_child_weight': min_child_weight,
            'objective': objective,
            'scale_pos_weight': scale_pos_weight,
            'random_state': random_state,
            'n_jobs': n_jobs
        }
        
        # Store focal loss parameters for later use
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Set up the XGBoost classifier
        self._base = xgb.XGBClassifier(**self.params)
        self._clf = None  # Will hold the pipeline
        self.feature_names = None

    def _calculate_class_weights(self, y):
        """
        Calculate class weights for imbalanced datasets.
        
        Parameters:
        -----------
        y : pd.Series or np.ndarray
            Target values
            
        Returns:
        --------
        float
            Appropriate scale_pos_weight value
        """
        # Count classes
        if isinstance(y, pd.Series):
            class_counts = y.value_counts()
        else:
            class_counts = np.bincount(y)
        
        # Calculate scale_pos_weight (negative_samples / positive_samples)
        if len(class_counts) > 1 and 1 in class_counts.index and 0 in class_counts.index:
            neg_count = class_counts[0]
            pos_count = class_counts[1]
            
            if pos_count > 0:
                return neg_count / pos_count
        
        # Default to 1.0 if something goes wrong
        return 1.0

    def _create_focal_loss_objective(self, alpha=0.25, gamma=2.0):
        """
        Create a custom focal loss objective function for XGBoost.
        
        Focal Loss: -alpha * (1-p)^gamma * log(p) for y=1, -alpha * p^gamma * log(1-p) for y=0
        
        Parameters:
        -----------
        alpha : float, default=0.25
            Balances positive and negative samples
        gamma : float, default=2.0
            Focuses more on hard to classify examples
            
        Returns:
        --------
        function
            Custom objective function for XGBoost
        """
        def focal_loss_obj(predt, dtrain):
            """
            Custom objective function for focal loss.
            
            Parameters:
            -----------
            predt : np.ndarray
                Raw predictions (logits)
            dtrain : xgb.DMatrix
                Training data
                
            Returns:
            --------
            tuple
                Gradient and Hessian
            """
            y = dtrain.get_label()
            
            # Convert raw predictions to probabilities using sigmoid
            p = 1.0 / (1.0 + np.exp(-predt))
            
            # Calculate gradient
            pt = p * y + (1.0 - p) * (1.0 - y)  # p_t in the paper
            grad = alpha * np.power(1.0 - pt, gamma) * (y - p)
            
            # Calculate hessian
            h1 = alpha * np.power(1.0 - pt, gamma)
            h2 = alpha * gamma * np.power(1.0 - pt, gamma - 1.0) * pt
            hess = h1 * p * (1.0 - p) - h2 * (y - p) * (1.0 - 2.0 * p)
            
            return grad, hess
        
        return focal_loss_obj

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
        
        try:
            # Check for class imbalance and adjust scale_pos_weight if not set manually
            if self.params['scale_pos_weight'] == 1.0:
                # Only automatically calculate if not explicitly set
                self.params['scale_pos_weight'] = self._calculate_class_weights(y)
                self._base.scale_pos_weight = self.params['scale_pos_weight']
                logger.info(f"Class imbalance detected. Set scale_pos_weight to {self.params['scale_pos_weight']:.2f}")
            
            # Check if focal loss should be used
            if self.focal_alpha is not None and self.focal_gamma is not None:
                logger.info(f"Using focal loss with alpha={self.focal_alpha}, gamma={self.focal_gamma}")
                
                # If using focal loss, create a custom objective
                focal_obj = self._create_focal_loss_objective(self.focal_alpha, self.focal_gamma)
                
                # Convert to DMatrix for custom objective
                dmat = xgb.DMatrix(X, label=y)
                
                # Train with custom objective
                params = self.params.copy()
                params.pop('n_estimators', None)  # Remove n_estimators as it's passed separately
                params.pop('objective', None)     # Remove default objective
                
                # Create booster directly
                self._base_booster = xgb.train(
                    params=params,
                    dtrain=dmat,
                    num_boost_round=self.params['n_estimators'],
                    obj=focal_obj
                )
                
                # Store trained model with custom objective
                self._base.get_booster = lambda: self._base_booster
                
                # Define predict method for pipeline compatibility
                def predict_proba(X):
                    dmat = xgb.DMatrix(X)
                    preds = self._base_booster.predict(dmat)
                    # Convert to 2D array of [1-p, p] format
                    return np.vstack((1-preds, preds)).T
                
                self._base.predict_proba = predict_proba
                
                # Create pipeline with scaling and custom model
                self._clf = make_pipeline(StandardScaler(), self._base)
                return self
            
            # Standard training with regular objective
            self._clf = make_pipeline(StandardScaler(), self._base)
            self._clf.fit(X, y)
            return self
            
        except Exception as e:
            logger.error(f"Error during XGBoost training: {e}")
            # Fallback to standard training if custom objective fails
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
        
        logger.info(f"Model saved to {path}")

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
               f"learning_rate={self.params['learning_rate']}, " \
               f"scale_pos_weight={self.params['scale_pos_weight']})"
