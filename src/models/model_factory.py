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

from .decision_tree_model import DecisionTreeModel
from .random_forest_model import RandomForestModel
from .xgboost_model import XGBoostModel
from .stacking_model import StackingModel

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
            return DecisionTreeModel(calibrate=calibrate, **params)
        
        elif model_type == 'random_forest':
            calibrate = params.pop('calibrate', True)  # Changed default to True
            return RandomForestModel(calibrate=calibrate, **params)
        
        elif model_type == 'xgboost':
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Cannot create XGBoostModel.")
            return XGBoostModel(**params)
        
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
                'n_jobs': -1
            }
        
        elif model_type == 'stacking':
            params = {
                'base_models': [
                    {'model_type': 'decision_tree', 'model_params': {'max_depth': 5}},
                    {'model_type': 'random_forest', 'model_params': {'n_estimators': 100, 'max_depth': 5}}
                ],
                'cv': 5,
                'use_features': False,
                'use_basemodel_metamodel': False,
                'meta_model_type': 'logistic_regression',
                'meta_model_params': {'C': 1.0, 'random_state': 42}
            }
            
            # Add XGBoost as base model if available
            if XGBOOST_AVAILABLE:
                params['base_models'].append({
                    'model_type': 'xgboost', 
                    'model_params': {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
                })
                
            return params
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")