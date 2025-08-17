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
from .hyperparameter_manager import HyperparameterManager
import config
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ModelFactory:
    """
    Factory class for creating model instances.
    
    This class provides a centralized way to create different types of models
    with consistent interface, following the factory pattern.
    """

    # Create a singleton instance of HyperparameterManager
    _hyperparameter_manager = None
    
    @classmethod
    def get_hyperparameter_manager(cls):
        """
        Get or create the HyperparameterManager instance.
        
        Returns:
        --------
        HyperparameterManager
            Singleton instance of HyperparameterManager
        """
        if cls._hyperparameter_manager is None:
            cls._hyperparameter_manager = HyperparameterManager()
        return cls._hyperparameter_manager

    @staticmethod
    def create_model(model_type, **params):
        """
        Create a model instance based on the specified type.
        
        Parameters:
        -----------
        model_type : str
            Type of model to create ('decision_tree', 'random_forest', 'xgboost', 'transformer', 'hybrid', 'stacking')
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
        # Check if we should use optimized hyperparameters
        use_optimized = params.pop('use_optimized', False)
        regime = params.pop('regime', None)
        
        if use_optimized:
            # Get hyperparameters from HyperparameterManager
            hyperparams = ModelFactory.get_hyperparameter_manager().get_best_params(model_type, regime)
            # Update with explicitly provided parameters (they take precedence)
            hyperparams.update(params)
            params = hyperparams
            logger.info(f"Using optimized hyperparameters for {model_type}")
            
        if model_type == 'decision_tree':
            calibrate = params.pop('calibrate', False)  # Disable calibration by default for intraday
            # DecisionTree does support class_weight parameter - keep it
            return DecisionTreeModel(calibrate=calibrate, **params)
        
        elif model_type == 'random_forest':
            calibrate = params.pop('calibrate', False)  # Disable calibration by default for intraday
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
        elif model_type == 'transformer':
            from .transformer.transformer_wrapper import TransformerModelWrapper
            return TransformerModelWrapper(**params)

        elif model_type == 'hybrid':
            from .ensemble.hybrid_strategy import HybridMLStrategy
            dt_params = params.pop("dt_params", {})
            tf_params = params.pop("tf_params", {})
            dt_model = ModelFactory.create_model("decision_tree", **dt_params)
            tf_model = ModelFactory.create_model("transformer", **tf_params)
            return HybridMLStrategy(dt_model, tf_model, **params)
        
        elif model_type == 'stacking':
            # Special handling for stacking model creation
            
            # Create base models
            base_models_config = params.pop('base_models', [
                {'model_type': 'decision_tree', 'model_params': {'max_depth': 5}},
                {'model_type': 'random_forest', 'model_params': {'n_estimators': 100, 'max_depth': 5}}
            ])
            
            base_models = []
            for config in base_models_config:
                base_model_type = config['model_type']
                model_params = config.get('model_params', {})
                
                # Apply optimized parameters to base models if requested
                if use_optimized:
                    optimized_params = ModelFactory.get_hyperparameter_manager().get_best_params(base_model_type, regime)
                    # Update with explicitly provided parameters
                    model_params_updated = optimized_params.copy()
                    model_params_updated.update(model_params)
                    model_params = model_params_updated
                
                try:
                    base_model = ModelFactory.create_model(base_model_type, **model_params)
                    base_models.append(base_model)
                except ImportError:
                    # Skip unavailable models (e.g., XGBoost if not installed)
                    logger.warning(f"Model type '{base_model_type}' is not available. Skipping this base model.")
            
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
                        logger.warning(f"Meta-model type '{meta_model_type}' is not available. Using default.")
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
        models = ["decision_tree", "random_forest", "stacking", "transformer", "hybrid"]
        
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
            Type of model ('decision_tree', 'random_forest', 'xgboost', 'transformer', 'hybrid', 'stacking')
            
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
                'calibrate': False,  # Disable calibration by default for intraday
                'max_depth': 12,  # Increased from 5 for better intraday patterns
                'min_samples_split': 20,
                'min_samples_leaf': 5,  # Increased to reduce overfitting
                'max_features': None,
                'criterion': 'gini',
                'class_weight': 'balanced',  # Add class weight for better balance
                'random_state': 42
            }
        
        elif model_type == 'random_forest':
            return {
                'calibrate': False,  # Disable calibration by default for intraday
                'n_estimators': 300,  # Increased for better stability
                'max_depth': 12,  # Increased from 5 for better patterns
                'min_samples_split': 10,
                'min_samples_leaf': 5,  # Increased from 1 to reduce overfitting
                'max_features': 'sqrt',
                'criterion': 'gini',
                'class_weight': 'balanced_subsample',  # Better for RF
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
        
        elif model_type == "transformer":
            return TRANSFORMER_CONFIG["default"]

        elif model_type == "hybrid":
            return HYBRID_CONFIG["balanced"]
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

    @classmethod
    def create_optimized_model(cls, model_type, X=None, y=None, prices=None, n_trials=None, regime=None,
                               force_optimization=False):
        """
        Create a model with optimized hyperparameters using HyperparameterManager.
        
        Parameters:
        -----------
        model_type : str
            Type of model to create ('decision_tree', 'random_forest', 'xgboost', 'transformer', 'hybrid')
        X : pd.DataFrame or None
            Feature matrix (required if force_optimization=True)
        y : pd.Series or None
            Target values (required if force_optimization=True)
        n_trials : int or None
            Number of optimization trials (default: from config)
        regime : str or None
            Market regime (if provided, optimize for this specific regime)
        force_optimization : bool
            Whether to force optimization even if saved parameters exist
            
        Returns:
        --------
        BaseModel
            Instance of the optimized model
            
        Raises:
        -------
        ImportError
            If Optuna is not installed
        ValueError
            If force_optimization=True but X or y is None
        """
        if force_optimization and (X is None or y is None or prices is None):
            raise ValueError("X, y, and prices must be provided when force_optimization=True")
            
        if not OPTUNA_AVAILABLE and force_optimization:
            raise ImportError("Optuna is not installed. Cannot optimize hyperparameters.")
        
        # Use HyperparameterManager to create optimized model
        return cls.get_hyperparameter_manager().create_optimized_model(
            model_type=model_type,
            X=X,
            y=y,
            prices=prices,
            regime=regime,
            force_optimization=force_optimization,
            n_trials=n_trials
        )
