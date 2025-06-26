from src.models.model_factory import ModelFactory


def test_create_all_models():
    for mtype in ['decision_tree','random_forest','transformer','hybrid']:
        model = ModelFactory.create_model(mtype)
        assert model is not None


def test_param_passing():
    model = ModelFactory.create_model('decision_tree', max_depth=3)
    assert model.model.max_depth == 3


def test_invalid_model():
    import pytest
    with pytest.raises(ValueError):
        ModelFactory.create_model('unknown')
