#!/usr/bin/env python
# tests/test_focal_loss.py

"""
Test script to verify the fix for focal loss in hyperparameter tuning.
This script specifically tests the XGBoostModel with focal loss enabled.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.xgboost_model import XGBoostModel
from src.features.feature_engineering import prepare_train_test_data

def test_focal_loss():
    """Test XGBoostModel with focal loss."""
    # Load a small dataset for testing
    data_path = "data/raw/historical_data_STOCK_SPY_1_day2010-2025.csv"
    print(f"Loading data from {data_path}...")
    
    # Load data
    df = pd.read_csv(data_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # Prepare data
    train_end_date = '2020-12-31'
    X_train, X_test, y_train, y_test, dates_train, dates_test, scaler = prepare_train_test_data(
        df, train_end_date=train_end_date
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Create model with focal loss
    print("Creating XGBoostModel with focal loss...")
    model = XGBoostModel(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        use_focal_loss=True,
        focal_gamma=2.0,
        focal_alpha=0.25,
        random_state=42
    )
    
    # Train model
    print("Training model...")
    model.train(X_train, y_train)
    
    # Test prediction
    print("Testing predict method...")
    y_pred = model.predict(X_test)
    
    # Print prediction statistics
    print(f"Prediction stats: min={y_pred.min():.4f}, max={y_pred.max():.4f}, mean={y_pred.mean():.4f}")
    
    # Convert to binary predictions
    threshold = 0.5
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Calculate accuracy
    accuracy = (y_pred_binary == y_test).mean()
    print(f"Accuracy: {accuracy:.4f}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_focal_loss()
