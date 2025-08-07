import pandas as pd
import numpy as np

from src.models.hyperparameter_optimization import optimize_decision_tree

def test_optimize_decision_tree_returns_params():
    np.random.seed(0)
    dates = pd.date_range('2020-01-01', periods=60, freq='D')
    X = pd.DataFrame({'feat': np.random.randn(60)}, index=dates)
    y = pd.Series((np.random.randn(60) > 0).astype(int), index=dates)
    prices = pd.Series(100 + np.cumsum(np.random.randn(60)), index=dates, name='close')
    params = optimize_decision_tree(X, y, prices, n_trials=1, n_splits=2)
    assert isinstance(params, dict)
    assert 'max_depth' in params
