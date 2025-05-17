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
    def create_model(model_type, **kwargs):
        """
        Create a model instance based on the specified type.
        
        Parameters:
        -----------
        model_type : str
            Type of model to create ('decision_tree', 'random_forest', 'xgboost', 'stacking')
        **kwargs : dict
            Additional parameters to pass to the model constructor
            
        Returns:
        --------
        BaseModel
            Model instance
            
        Raises:
        -------
        ValueError
            If model_type is not recognized
        """
        model_type = model_type.lower()
        
        if model_type == "decision_tree":
            return DecisionTreeModel(calibrate=kwargs.pop('calibrate', False), **kwargs)
        
        elif model_type == "random_forest":
            return RandomForestModel(calibrate=kwargs.pop('calibrate', False), **kwargs)
        
        elif model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost is not installed. Please install it with 'pip install xgboost'")
            return XGBoostModel(**kwargs)
        
        elif model_type == "stacking":
            base_models = kwargs.pop('base_models', None)
            if base_models:
                # Convert model configs to actual model instances
                model_instances = []
                for model_config in base_models:
                    model = ModelFactory.create_model(
                        model_config['model_type'],
                        **model_config.get('model_params', {})
                    )
                    model_instances.append(model)
                return StackingModel(base_models=model_instances, **kwargs)
            else:
                return StackingModel(**kwargs)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def get_available_models():
        """
        Get a list of available model types.
        
        Returns:
        --------
        list
            List of available model types
        """
        models = ["decision_tree", "random_forest", "stacking"]
        if XGBOOST_AVAILABLE:
            models.append("xgboost")
        return models