import pandas as pd
import torch
import numpy as np
import pytest
from src.models.transformer.transformer_wrapper import TransformerModelWrapper


def test_empty_data_handling():
    model = TransformerModelWrapper()
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        model.train(empty_df, np.array([]))


def test_single_sample_prediction():
    model = TransformerModelWrapper()
    X = pd.DataFrame(np.random.randn(1, 9))
    model.feature_columns = list(X.columns)
    model.preparator = None
    model.model = lambda x: torch.zeros(len(x), 1)
    model.is_fitted = True
    with pytest.raises(Exception):
        model.predict(X)
