import numpy as np
import pandas as pd
from src.models.decision_tree_model import DecisionTreeModel


def test_probability_spread():
    X = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'b', 'c'])
    y = np.random.randint(0, 2, 200)
    model = DecisionTreeModel(calibrate=True)
    model.train(X, y)
    preds = model.predict(X)
    assert preds.std() > 0
