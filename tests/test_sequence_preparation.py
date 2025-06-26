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

def test_various_sequence_lengths():
    df = make_df(20)
    for sl in [2,3,5]:
        sp = SequencePreparator(seq_length=sl, prediction_length=1)
        sp.fit(df)
        X, y = sp.transform(df)
        assert X.shape[1] == sl
        assert len(X) == len(df) - sl - 1 + 1
        assert y.shape[0] == len(X)


def test_single_sample_edge_case():
    df = make_df(3)
    sp = SequencePreparator(seq_length=5)
    with np.testing.assert_raises(ValueError):
        sp.fit(df)
        sp.transform(df)
