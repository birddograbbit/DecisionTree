import pandas as pd
import numpy as np
from src.models.transformer.transformer_wrapper import TransformerModelWrapper
import torch
import unittest.mock as mock


def make_df(n=40):
    data = {
        'open': np.arange(n)+1,
        'high': np.arange(n)+2,
        'low': np.arange(n),
        'close': np.arange(n)+1,
        'volume': np.ones(n)
    }
    df = pd.DataFrame(data)
    df['target'] = np.random.randint(0,2,size=n)
    return df


def test_train_predict(tmp_path):
    df = make_df()
    model = TransformerModelWrapper(seq_length=5, epochs=1, batch_size=4)
    model.train(df.drop('target',axis=1), df['target'])
    preds = model.predict(df.drop('target',axis=1))
    assert len(preds) == len(df)

    save_path = tmp_path/"model.pt"
    model.save(save_path)
    orig_load = torch.load
    with mock.patch('torch.load', side_effect=lambda *a, **k: orig_load(*a, weights_only=False, **k)):
        loaded = TransformerModelWrapper.load(save_path)
    assert loaded.is_fitted


def test_missing_data():
    df = make_df()
    model = TransformerModelWrapper(seq_length=5, epochs=1)
    df.loc[0,'open'] = np.nan
    model.train(df.drop('target',axis=1).fillna(0), df['target'])
    preds = model.predict(df.drop('target',axis=1).fillna(0))
    assert len(preds) == len(df)


def test_single_sample_batch():
    df = make_df(n=6)
    model = TransformerModelWrapper(seq_length=5, epochs=1, batch_size=4)
    model.train(df.drop('target', axis=1), df['target'])
    preds = model.predict(df.drop('target', axis=1))
    assert len(preds) == len(df)
