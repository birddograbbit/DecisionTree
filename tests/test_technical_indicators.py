import pandas as pd
import numpy as np
from src.features.transformers.technical_indicators_transformer import add_technical_indicators


def test_indicator_calculation_no_nan():
    data = list(range(1, 51))
    df = pd.DataFrame({
        'open': data,
        'high': data,
        'low': data,
        'close': data,
        'volume': np.ones(50)
    })
    out = add_technical_indicators(df)
    assert 'rsi' in out.columns
    assert not out.isna().any().any()
