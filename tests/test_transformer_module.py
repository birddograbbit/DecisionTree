import numpy as np
import pandas as pd
import torch
import pytest

from src.models.transformer.sequence_preparation import SequencePreparator
from src.models.transformer.transformer_wrapper import TransformerModelWrapper


def test_strict_feature_validation():
    df = pd.DataFrame({'open': [1, 2], 'close': [1, 2]})
    preparator = SequencePreparator(feature_columns=['open', 'high'], strict=True)
    with pytest.raises(ValueError):
        preparator.fit(df)


def test_short_sequence_prediction_returns_neutral():
    df = pd.DataFrame({
        'open': np.arange(3),
        'high': np.arange(3),
        'low': np.arange(3),
        'close': np.arange(3)
    })
    wrapper = TransformerModelWrapper(seq_length=5, n_features=4, target_column='close')
    wrapper.feature_columns = list(df.columns)
    wrapper.preparator = SequencePreparator(seq_length=5, feature_columns=wrapper.feature_columns, target_column='close')
    wrapper.preparator.fit(df)
    wrapper.model = torch.nn.Identity()
    wrapper.is_fitted = True

    preds = wrapper.predict(df)
    assert np.allclose(preds, 0.5)

def test_non_strict_drops_missing_features():
    df = pd.DataFrame({'open': [1, 2], 'close': [1, 2]})
    preparator = SequencePreparator(feature_columns=['open', 'high'], strict=False)
    preparator.fit(df)
    assert preparator.feature_columns == ['open']

