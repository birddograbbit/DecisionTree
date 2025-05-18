"""
Stacked ensemble model implementation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from .base_model import BaseModel
from .model_factory import ModelFactory

class StackedModel(BaseModel):
    """
    Stacked ensemble model implementation of the BaseModel interface.
    
    This model combines multiple base models and uses their predictions
    as features for a meta-model.
    """
    
    def __init__(self, base_models=None, meta_model_type='logistic_regression', meta_model_params=None):
        """
        Initialize the stacked model.
        
        Parameters:
        -----------
        base_models : list of dicts, optional
            List of base model configurations in the format:
            [{'type': 'random_forest', 'params': {...}}, ...]
            If None, default models will be used.
        meta_model_type : str, default='logistic_regression'
            Type of meta-model to use
        meta_model_params : dict, optional
            Parameters for the meta-model
        """
        # Initialize base models
        self.base_models = []
        self.meta_model = None
        self.feature_names = None
        
        # Setup base models
        if base_models is None:
            # Default set of base models
            base_models = [
                {'type': 'decision_tree', 'params': {'max_depth': 5}},
                {'type': 'random_forest', 'params': {'n_estimators': 100, 'max_depth': 5}}
            ]
            
            # Add XGBoost if available
            try:
                import xgboost
                base_models.append({
                    'type': 'xgboost', 
                    'params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
                })
            except ImportError:
                print("XGBoost not available, proceeding without it in the stack")
        
        # Create base models
        for config in base_models:
            model = ModelFactory.create_model(config['type'], **config['params'])
            self.base_models.append({
                'type': config['type'],
                'model': model,
                'params': config['params']
            })
        
        # Setup meta model
        self.meta_model_type = meta_model_type
        self.meta_model_params = meta_model_params or {}
        
        # Use appropriate meta-model based on type
        if meta_model_type == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            # Use the saga solver with full parallelism for best performance
            self.meta_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, solver='saga', n_jobs=-1,
                                   **self.meta_model_params)
            )
        elif meta_model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            self.meta_model = RandomForestClassifier(**self.meta_model_params)
        else:
            raise ValueError(f"Unsupported meta-model type: {meta_model_type}")

    def train(self, X, y):
        """
        Train the stacked model using K-fold cross-validation.
        
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
        
        # Convert inputs to numpy for more efficient processing
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        # Number of folds for cross-validation
        n_folds = 5
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        # Initialize meta-features array
        meta_features = np.zeros((X_array.shape[0], len(self.base_models)))
        
        # Generate meta-features through cross-validation
        for i, (train_idx, val_idx) in enumerate(kf.split(X_array)):
            X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X_array[train_idx]
            y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y_array[train_idx]
            X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X_array[val_idx]
            
            # Train each base model on this fold
            for j, base_model_config in enumerate(self.base_models):
                base_model = base_model_config['model']
                base_model.train(X_train_fold, y_train_fold)
                
                # Generate predictions on validation fold
                meta_features[val_idx, j] = base_model.predict(X_val_fold)
        
        # Train meta-model on the generated meta-features
        self.meta_model.fit(meta_features, y_array)
        
        # Finally, train all base models on the full dataset
        for base_model_config in self.base_models:
            base_model = base_model_config['model']
            base_model.train(X, y)
        
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
            Predicted probabilities for positive class
        """
        # Generate predictions from each base model
        X_array = X.values if hasattr(X, 'values') else X
        meta_features = np.zeros((X_array.shape[0], len(self.base_models)))
        
        for i, base_model_config in enumerate(self.base_models):
            base_model = base_model_config['model']
            meta_features[:, i] = base_model.predict(X)
        
        # Use meta-model to generate final predictions
        if hasattr(self.meta_model, 'predict_proba'):
            return self.meta_model.predict_proba(meta_features)[:, 1]
        else:
            return self.meta_model.predict(meta_features)

    def get_feature_importance(self):
        """
        Return feature importance scores.
        
        For stacked models, this returns the importance of each base model
        in the meta-model.
        
        Returns:
        --------
        dict
            Feature importance scores for base models
        """
        # Get importance of each base model in the meta-model
        if hasattr(self.meta_model, 'coef_'):
            importances = self.meta_model.coef_[0]
        elif hasattr(self.meta_model, 'feature_importances_'):
            importances = self.meta_model.feature_importances_
        else:
            # If meta-model doesn't provide feature importance, return equal weights
            return {f"Model_{i}": 1.0/len(self.base_models) for i in range(len(self.base_models))}
        
        # Create a dictionary mapping base models to their importance
        base_model_importance = {}
        for i, base_model_config in enumerate(self.base_models):
            model_type = base_model_config['type']
            base_model_importance[f"{model_type}_{i}"] = importances[i]
        
        return base_model_importance

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
        StackedModel
            Loaded model instance
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)
        
        # Ensure the loaded object is a StackedModel
        if not isinstance(model, cls):
            raise TypeError(f"Loaded model is not a {cls.__name__}")
        
        return model
        
    def __str__(self):
        """String representation of the model."""
        base_models_str = ', '.join(f"{config['type']}" for config in self.base_models)
        return f"StackedModel(base_models=[{base_models_str}], meta_model={self.meta_model_type})"
