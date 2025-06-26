import pandas as pd
import numpy as np
from src.models.ensemble.hybrid_strategy import HybridMLStrategy
from src.models.model_factory import ModelFactory


def make_df(n=30):
    data = {
        'open': np.arange(n)+1,
        'high': np.arange(n)+2,
        'low': np.arange(n),
        'close': np.arange(n)+1,
        'volume': np.ones(n)
    }
    return pd.DataFrame(data)


def test_hybrid_prediction():
    dt = ModelFactory.create_model('decision_tree')
    tf = ModelFactory.create_model('transformer', seq_length=3, epochs=1)
    strategy = HybridMLStrategy(dt, tf)
    df = make_df()
    df['target'] = np.random.randint(0,2,size=len(df))
    tf.train(df.drop('target',axis=1), df['target'])
    dt.train(df.drop('target',axis=1), df['target'])
    signals = strategy.predict(df.drop('target',axis=1))
    assert len(signals) == len(df)


def test_dynamic_weighting():
    dt_pred = np.array([0.2,0.8])
    tf_pred = np.array([0.6,0.4])
    regimes = pd.Series(['trending','ranging'])
    strategy = HybridMLStrategy(None, None)
    combined = strategy._combine_predictions(dt_pred, tf_pred, regimes)
    assert combined.shape[0] == 2
