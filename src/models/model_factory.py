"""
Model factory for creating instances of trading models.
"""

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost is not installed. XGBoostModel will not be available.")
    print("To install XGBoost: pip install xgboost")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna is not installed. Hyperparameter optimization will not be available.")
    print("To install Optuna: pip install optuna")

from .decision_tree_model import DecisionTreeModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel
from .stacking_model import StackingModel
import config

class ModelFactory:
    """
    Factory class for creating model instances.
    
    This class provides a centralized way to create different types of models
    with consistent interface, following the factory pattern.
    """

    @staticmethod
    def create_model(model_type, **params):
        """
        Create a model instance based on the specified type.
        
        Parameters:
        -----------
        model_type : str
            Type of model to create ('decision_tree', 'random_forest', 'xgboost', 'stacking')
        params : dict
            Parameters to pass to the model constructor
            
        Returns:
        --------
        BaseModel
            Instance of the requested model
            
        Raises:
        -------
        ValueError
            If the specified model type is not recognized
        ImportError
            If XGBoost is requested but not installed
        """
        if model_type == 'decision_tree':
            calibrate = params.pop('calibrate', True)  # Changed default to True
            # Fix: Remove class_weight parameter if it exists as Decision Tree model doesn't support it
            params.pop('class_weight', None)
            return DecisionTreeModel(calibrate=calibrate, **params)
        
        elif model_type == 'random_forest':
            calibrate = params.pop('calibrate', True)  # Changed default to True
            return RandomForestModel(calibrate=calibrate, **params)
        
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Cannot create XGBoostModel.")
            
            # Handle class imbalance parameters
            use_focal_loss = params.pop('use_focal_loss', False)
            focal_gamma = params.pop('focal_gamma', 2.0)
            focal_alpha = params.pop('focal_alpha', 0.25)
            class_weight = params.pop('class_weight', None)
            
            return XGBoostModel(
                use_focal_loss=use_focal_loss, 
                focal_gamma=focal_gamma, 
                focal_alpha=focal_alpha,
                class_weight=class_weight,
                **params
            )
        
        elif model_type == 'stacking':
            # Special handling for stacking model creation
            
            # Create base models
            base_models_config = params.pop('base_models', [
                {'model_type': 'decision_tree', 'model_params': {'max_depth': 5}},
                {'model_type': 'random_forest', 'model_params': {'n_estimators': 100, 'max_depth': 5}}
            ])
            
            base_models = []
            for config in base_models_config:
                model_type = config['model_type']
                model_params = config.get('model_params', {})
                try:
                    base_model = ModelFactory.create_model(model_type, **model_params)
                    base_models.append(base_model)
                except ImportError:
                    # Skip unavailable models (e.g., XGBoost if not installed)
                    print(f"Warning: Model type '{model_type}' is not available. Skipping this base model.")
            
            # Check if we should use a BaseModel for meta-model or direct sklearn model
            use_basemodel_metamodel = params.pop('use_basemodel_metamodel', False)
            
            # Handle the case where 'meta_model' is a dict configuration (from strategy_runner.py)
            if 'meta_model' in params:
                meta_model_config = params.pop('meta_model')
                # If meta_model is a dict with model_type and model_params
                if isinstance(meta_model_config, dict) and 'model_type' in meta_model_config:
                    use_basemodel_metamodel = True
                    meta_model_type = meta_model_config.get('model_type', 'random_forest')
                    meta_model_params = meta_model_config.get('model_params', {})
                    try:
                        meta_model = ModelFactory.create_model(meta_model_type, **meta_model_params)
                    except ImportError:
                        print(f"Warning: Meta-model type '{meta_model_type}' is not available. Using default.")
                        meta_model = None
                        use_basemodel_metamodel = False
                else:
                    # If meta_model is provided directly as an instance
                    meta_model = meta_model_config
            else:
                meta_model = None
            
            if use_basemodel_metamodel and meta_model is not None:
                # Create stacking model with BaseModel meta-model
                return StackingModel(
                    base_models=base_models,
                    meta_model=meta_model,
                    **params
                )
            else:
                # Use direct sklearn meta-model (more efficient)
                meta_model_type = params.pop('meta_model_type', 'logistic_regression')
                meta_model_params = params.pop('meta_model_params', {})
                
                # Create stacking model with sklearn meta-model
                return StackingModel(
                    base_models=base_models,
                    meta_model=None,
                    meta_model_type=meta_model_type,
                    meta_model_params=meta_model_params,
                    **params
                )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def get_available_models():
        """
        Get a list of available model types.
        
        Returns:
        --------
        list
            List of available model types
        """
        models = ['decision_tree', 'random_forest', 'stacking']
        
        if XGBOOST_AVAILABLE:
            models.append('xgboost')
            
        return models

    @staticmethod
    def get_default_params(model_type):
        """
        Get default parameters for the specified model type.
        
        Parameters:
        -----------
        model_type : str
            Type of model ('decision_tree', 'random_forest', 'xgboost', 'stacking')
            
        Returns:
        --------
        dict
            Default parameters for the model
            
        Raises:
        -------
        ValueError
            If the specified model type is not recognized
        """
        if model_type == 'decision_tree':
            return {
                'calibrate': True,  # Changed from False
                'max_depth': 5,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': None,
                'criterion': 'gini',
                # Fix: Remove class_weight from default params for decision tree
                'random_state': 42
            }
        
        elif model_type == 'random_forest':
            return {
                'calibrate': True,  # Changed from False
                'n_estimators': 100,
                'max_depth': 5,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'criterion': 'gini',
                'class_weight': 'balanced',  # Added class weight
                'random_state': 42,
                'n_jobs': -1
            }
        
        elif model_type == 'xgboost':
            return {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'objective': 'binary:logistic',
                'random_state': 42,
                'n_jobs': -1,
                'use_focal_loss': False,  # Added focal loss parameter
                'focal_gamma': 2.0,       # Added focal loss gamma
                'focal_alpha': 0.25,      # Added focal loss alpha
                'class_weight': 'balanced'  # Added class weight
            }
        
        elif model_type == 'stacking':
            params = {
                'base_models': [
                    {'model_type': 'decision_tree', 'model_params': {'max_depth': 5}},  # Removed class_weight
                    {'model_type': 'random_forest', 'model_params': {'n_estimators': 100, 'max_depth': 5, 'class_weight': 'balanced'}}
                ],
                'cv': 5,
                'use_features': False,
                'use_basemodel_metamodel': False,
                'meta_model_type': 'logistic_regression',
                'meta_model_params': {'C': 1.0, 'max_iter': 1000, 'random_state': 42}  # Increased max_iter
            }
            
            # Add XGBoost as base model if available
            if XGBOOST_AVAILABLE:
                params['base_models'].append({
                    'model_type': 'xgboost', 
                    'model_params': {
                        'n_estimators': 100, 
                        'max_depth': 5, 
                        'learning_rate': 0.1,
                        'class_weight': 'balanced'
                    }
                })
                
            return params
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @staticmethod
    def create_optimized_model(model_type, X, y, n_trials=None, n_splits=None, random_state=None):
        """
        Create a model with optimized hyperparameters using Optuna.
        
        Parameters:
        -----------
        model_type : str
            Type of model to create ('decision_tree', 'random_forest', 'xgboost')
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target values
        n_trials : int or None
            Number of optimization trials (default: from config)
        n_splits : int or None
            Number of splits for TimeSeriesSplit (default: from config)
        random_state : int or None
            Random seed for reproducibility (default: from config)
            
        Returns:
        --------
        BaseModel
            Instance of the optimized model
            
        Raises:
        -------
        ImportError
            If Optuna is not installed
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna is not installed. Cannot optimize hyperparameters.")
        
        # Import here to avoid circular imports
        from .hyperparameter_optimization import optimize_hyperparameters
        
        # Use default values from config if not specified
        n_trials = n_trials if n_trials is not None else config.OPTUNA_TRIALS
        n_splits = n_splits if n_splits is not None else config.TIMESERIES_CV_SPLITS
        random_state = random_state if random_state is not None else config.RANDOM_STATE
        
        # Optimize hyperparameters
        best_params = optimize_hyperparameters(
            model_type, X, y, 
            n_trials=n_trials, 
            n_splits=n_splits, 
            random_state=random_state
        )
        
        # Handle focal loss parameters separately for XGBoost
        if model_type == 'xgboost':
            use_focal_loss = best_params.pop('use_focal_loss', False)
            focal_gamma = best_params.pop('focal_gamma', 2.0) if use_focal_loss else None
            focal_alpha = best_params.pop('focal_alpha', 0.25) if use_focal_loss else None
            
            # Create model with best hyperparameters
            model = ModelFactory.create_model(
                model_type, 
                use_focal_loss=use_focal_loss, 
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                **best_params
            )
        else:
            # Fix: For decision tree, ensure class_weight is removed
            if model_type == 'decision_tree' and 'class_weight' in best_params:
                best_params.pop('class_weight')
                
            # Create model with best hyperparameters
            model = ModelFactory.create_model(model_type, **best_params)
        
        return model