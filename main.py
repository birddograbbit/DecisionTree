# main.py (updated for IBKR data files)

import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt

from src.data.preprocessing import load_ibkr_data, preprocess_data
from src.features.feature_engineering import prepare_train_test_data
from src.models.decision_tree import train_decision_tree, evaluate_model, save_model, visualize_decision_tree
import config

def main():
    """
    Main entry point.
    """
    print("Decision Tree Trading Strategy")
    print("-----------------------------\n")
    
    # Define data file paths
    train_file = '/Users/jt/TWS/decision_tree_trading/data/raw/historical_data_STOCK_SPY_1_day2000-2009.csv'
    test_file = '/Users/jt/TWS/decision_tree_trading/data/raw/historical_data_STOCK_SPY_1_day2010-2025.csv'
    
    # Check if data files exist
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("Data files not found. Please check the file paths.")
        return
    
    # Load and combine data
    df = load_ibkr_data(train_file, test_file)
    
    if df is None or df.empty:
        print("Failed to load data. Exiting...")
        return
    
    # Further preprocess the data if needed
    df = preprocess_data(df)
    
    print(f"Data loaded and preprocessed. Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Sample data:\n{df.head()}")
    
    # Prepare training and testing data
    train_end_date = '2009-12-31'
    X_train, X_test, y_train, y_test, dates_train, dates_test, scaler = prepare_train_test_data(
        df, train_end_date
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Train model
    model = train_decision_tree(X_train, y_train, max_depth=5)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    print("\nModel Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    if metrics['auc'] is not None:
        print(f"AUC: {metrics['auc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Save model
    model_file = os.path.join('data/models', "SPY_decision_tree.pkl")
    save_model(model, model_file, scaler)
    
    # Visualize tree
    fig = visualize_decision_tree(model, X_train.columns)
    
    # Save visualization
    viz_file = os.path.join('data/models', "SPY_decision_tree.png")
    fig.savefig(viz_file)
    
    print(f"Decision tree visualization saved to {viz_file}")
    
    # Print feature importances
    feature_importance = pd.Series(model.feature_importances_, index=X_train.columns)
    feature_importance = feature_importance.sort_values(ascending=False)
    
    print("\nFeature Importances:")
    for feature, importance in feature_importance.items():
        print(f"{feature}: {importance:.4f}")
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='bar')
    plt.title('Feature Importances')
    plt.tight_layout()
    
    # Save plot
    importance_file = os.path.join('data/models', "SPY_feature_importance.png")
    plt.savefig(importance_file)
    
    print(f"Feature importance plot saved to {importance_file}")

if __name__ == "__main__":
    main()