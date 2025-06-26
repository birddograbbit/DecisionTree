import pandas as pd
import numpy as np
from src.models.transformer.sequence_preparation import SequencePreparator, StockSequenceDataset


def make_df(n=40):
    data = {
        'open': np.arange(n) + 1,
        'high': np.arange(n) + 2,
        'low': np.arange(n),
        'close': np.arange(n) + 1,
        'volume': np.ones(n)
    }
    return pd.DataFrame(data)


def test_fit_transform_shapes():
    df = make_df()
    sp = SequencePreparator(seq_length=5, prediction_length=1)
    sp.fit(df)
    X, y = sp.transform(df)
    assert X.shape[1] == 5
    assert len(X) == len(df) - 5 - 1 + 1
    assert y.shape[0] == len(X)


def test_dataset_len():
    df = make_df()
    sp = SequencePreparator(seq_length=4)
    sp.fit(df)
    X, y = sp.transform(df)
    ds = StockSequenceDataset(X, y)
    assert len(ds) == len(X)
