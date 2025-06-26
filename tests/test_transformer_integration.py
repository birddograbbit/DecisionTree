import numpy as np
import pandas as pd
from src.models.model_factory import ModelFactory
from config import TRANSFORMER_CONFIG, HYBRID_CONFIG


def test_model_factory_integration():
    model = ModelFactory.create_model('transformer', **TRANSFORMER_CONFIG['default'])
    assert hasattr(model, 'train')
    assert hasattr(model, 'predict')

def test_factory_train_predict():
    X = pd.DataFrame(np.random.randn(50, TRANSFORMER_CONFIG['default']['n_features']))
    y = pd.Series(np.random.randint(0,2,size=50))
    model = ModelFactory.create_model('transformer', **TRANSFORMER_CONFIG['default'])
    model.train(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)


def test_hybrid_factory_creation():
    model = ModelFactory.create_model('hybrid', dt_params={}, tf_params=TRANSFORMER_CONFIG['default'])
    assert hasattr(model, 'predict')

