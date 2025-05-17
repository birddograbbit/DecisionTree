#!/usr/bin/env python
"""
Test script for probability calibration and stacking model changes.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.models.model_factory import ModelFactory

def main():
    print('Creating test data...')
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('\nTesting DecisionTree with calibration...')
    dt = ModelFactory.create_model('decision_tree', calibrate=True)
    dt.train(X_train, y_train)
    dt_probs = dt.predict(X_test)
    print(f'Probability distribution: min={dt_probs.min():.4f}, max={dt_probs.max():.4f}')
    print(f'Standard deviation of probabilities: {np.std(dt_probs):.4f}')
    print(f'Model property access test: {dt.model is not None}')

    print('\nTesting DecisionTree without calibration...')
    dt_uncal = ModelFactory.create_model('decision_tree', calibrate=False)
    dt_uncal.train(X_train, y_train)
    dt_uncal_probs = dt_uncal.predict(X_test)
    print(f'Probability distribution: min={dt_uncal_probs.min():.4f}, max={dt_uncal_probs.max():.4f}')
    print(f'Standard deviation of probabilities: {np.std(dt_uncal_probs):.4f}')
    print(f'Model property access test: {dt_uncal.model is not None}')

    print('\nTesting RandomForest with calibration...')
    rf = ModelFactory.create_model('random_forest', calibrate=True)
    rf.train(X_train, y_train)
    rf_probs = rf.predict(X_test)
    print(f'Probability distribution: min={rf_probs.min():.4f}, max={rf_probs.max():.4f}')
    print(f'Standard deviation of probabilities: {np.std(rf_probs):.4f}')
    print(f'Model property access test: {rf.model is not None}')

    print('\nTesting RandomForest without calibration...')
    rf_uncal = ModelFactory.create_model('random_forest', calibrate=False)
    rf_uncal.train(X_train, y_train)
    rf_uncal_probs = rf_uncal.predict(X_test)
    print(f'Probability distribution: min={rf_uncal_probs.min():.4f}, max={rf_uncal_probs.max():.4f}')
    print(f'Standard deviation of probabilities: {np.std(rf_uncal_probs):.4f}')
    print(f'Model property access test: {rf_uncal.model is not None}')

    print('\nTesting Stacking model with pipeline...')
    stacking = ModelFactory.create_model('stacking')
    base_models = [
        ModelFactory.create_model('decision_tree', calibrate=True),
        ModelFactory.create_model('random_forest', calibrate=True)
    ]
    for model in base_models:
        model.train(X_train, y_train)
    
    # Create predictions from base models
    base_preds = np.column_stack([model.predict(X_train) for model in base_models])
    
    # Train meta-learner
    stacking.meta_learner.fit(base_preds, y_train)
    
    # Test predictions
    test_base_preds = np.column_stack([model.predict(X_test) for model in base_models])
    stacking_probs = stacking.meta_learner.predict_proba(test_base_preds)[:, 1]
    print(f'Probability distribution: min={stacking_probs.min():.4f}, max={stacking_probs.max():.4f}')
    print(f'Standard deviation of probabilities: {np.std(stacking_probs):.4f}')

if __name__ == "__main__":
    main()

