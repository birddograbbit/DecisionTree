from src.models.model_factory import ModelFactory
from config import TRANSFORMER_CONFIG


def test_model_factory_integration():
    model = ModelFactory.create_model('transformer', **TRANSFORMER_CONFIG['default'])
    assert hasattr(model, 'train')
    assert hasattr(model, 'predict')
