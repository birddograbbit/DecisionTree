"""
Base model interface for trading models.
"""

from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all trading models.
    
    This interface ensures all models implement a common API,
    allowing them to be used interchangeably throughout the system.
    """

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_feature_importance(self):
        """
        Return feature importance scores.
        
        Returns:
        --------
        dict or np.ndarray
            Feature importance scores, ideally as a dict mapping feature names to scores
        """
        pass

    @abstractmethod
    def save(self, path):
        """
        Save model to disk.
        
        Parameters:
        -----------
        path : str
            Path to save model
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path):
        """
        Load model from disk.
        
        Parameters:
        -----------
        path : str
            Path to saved model
            
        Returns:
        --------
        BaseModel
            Loaded model instance
        """
        pass