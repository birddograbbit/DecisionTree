# examples/hyperparameter_tuning_example.py

"""
Example script for hyperparameter optimization and feature pruning.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.feature_engineering import prepare_train_test_data, audit_features, prune_features
from src.models.model_factory import ModelFactory
from src.models.hyperparameter_optimization import optimize_hyperparameters, save_hyperparameters
import config

def load_data(data_path):
    """
    Load price data from CSV file.
    
    Parameters:
    -----------
    data_path : str
        Path to CSV file
        
    Returns:
    --------
    pd.DataFrame
        Price data
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Set date as index
    df = df.set_index('date')
    
    return df

def find_optimal_threshold(model, X_val, y_val):
    """
    Find the optimal threshold for converting probabilities to class labels.
    
    Parameters:
    -----------
    model : object
        Trained model with predict method
    X_val : pd.DataFrame
        Validation feature matrix
    y_val : pd.Series
        Validation target values
        
    Returns:
    --------
    float
        Optimal threshold
    dict
        Metrics at optimal threshold
    """
    # Get probability predictions
    y_prob = model.predict(X_val)
    
    # Try different thresholds
    thresholds = np.arange(0.3, 0.7, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in thresholds:
        # Convert probabilities to binary predictions
        y_pred = (y_prob > threshold).astype(int)
        
        # Calculate metrics
        try:
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            accuracy = accuracy_score(y_val, y_pred)
            
            # Check if this threshold gives better F1 score
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_metrics = {
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1
                }
        except Exception as e:
            # Skip thresholds that cause errors (e.g., all predictions in one class)
            continue
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Metrics at optimal threshold:")
    for metric, value in best_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return best_threshold, best_metrics

def optimize_and_evaluate(df, model_type='xgboost', use_feature_pruning=True, n_trials=50):
    """
    Optimize hyperparameters and evaluate model performance.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data
    model_type : str
        Model type (default: 'xgboost')
    use_feature_pruning : bool
        Whether to prune features (default: True)
    n_trials : int
        Number of optimization trials (default: 50)
        
    Returns:
    --------
    tuple
        (best_params, metrics, feature_importances, top_features)
    """
    # Split data into train, validation, and test sets
    train_end_date = '2020-12-31'
    X_train, X_test, y_train, y_test, dates_train, dates_test, scaler = prepare_train_test_data(
        df, train_end_date=train_end_date
    )
    
    # Further split test set into validation and test
    X_val, X_test, y_val, y_test, dates_val, dates_test = train_test_split(
        X_test, y_test, dates_test, test_size=0.5, random_state=config.RANDOM_STATE
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Optimize hyperparameters
    print(f"Optimizing hyperparameters for {model_type} model...")
    best_params = optimize_hyperparameters(
        model_type, X_train, y_train, 
        n_trials=n_trials, 
        n_splits=config.TIMESERIES_CV_SPLITS,
        random_state=config.RANDOM_STATE
    )
    print(f"Best hyperparameters: {best_params}")
    
    # Create model with best hyperparameters
    # Note: Since we've updated the XGBoostModel to accept all parameters
    # directly, we don't need special handling anymore - just pass them all
    model = ModelFactory.create_model(model_type, **best_params)
    
    # Train model on unscaled data (scaling is done in pipeline)
    model.train(X_train, y_train)
    
    # Find optimal threshold using validation set
    best_threshold, val_metrics = find_optimal_threshold(model, X_val, y_val)
    
    # Feature importance and pruning with optimal threshold
    train_importances, test_importances, top_features = audit_features(
        model, X_train, y_train, X_test, y_test,
        n_repeats=config.FEATURE_AUDIT_N_REPEATS,
        n_top_features=config.TOP_N_FEATURES,
        random_state=config.RANDOM_STATE,
        threshold=best_threshold
    )
    
    print(f"Top {len(top_features)} features: {top_features}")
    
    # Prune features if requested
    if use_feature_pruning:
        X_train_pruned, X_test_pruned = prune_features(X_train, X_test, top_features)
        
        # Train new model on pruned features
        model_pruned = ModelFactory.create_model(model_type, **best_params)
        model_pruned.train(X_train_pruned, y_train)
        
        # Predictions on pruned features
        y_pred = model_pruned.predict(X_test_pruned)
    else:
        # Predictions on all features
        y_pred = model.predict(X_test)
    
    # Convert probabilities to binary predictions using optimal threshold
    y_pred_binary = (y_pred > best_threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred_binary),
        'precision': precision_score(y_test, y_pred_binary),
        'recall': recall_score(y_test, y_pred_binary),
        'f1': f1_score(y_test, y_pred_binary),
        'threshold': best_threshold
    }
    
    # Compute and display confusion matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nTrue Positives: {cm[1][1]}")
    print(f"False Positives: {cm[0][1]}")
    print(f"True Negatives: {cm[0][0]}")
    print(f"False Negatives: {cm[1][0]}")
    
    # Print metrics
    print(f"\nModel performance metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Class distribution
    pos_count = np.sum(y_test == 1)
    neg_count = np.sum(y_test == 0)
    pos_pct = pos_count / len(y_test) * 100
    neg_pct = neg_count / len(y_test) * 100
    print(f"\nTest set class distribution:")
    print(f"  Positive (1): {pos_count} ({pos_pct:.2f}%)")
    print(f"  Negative (0): {neg_count} ({neg_pct:.2f}%)")
    
    # Save hyperparameters
    save_hyperparameters(best_params, model_type)
    
    return best_params, metrics, train_importances, top_features

def plot_feature_importance(importances, title):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importances : pd.DataFrame
        Feature importance data
    title : str
        Plot title
    """
    # Sort by importance
    importances = importances.sort_values('importance_mean', ascending=True)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot importance
    plt.barh(importances['feature'], importances['importance_mean'])
    
    # Add error bars
    plt.errorbar(
        importances['importance_mean'], 
        importances['feature'],
        xerr=importances['importance_std'],
        fmt='none', 
        ecolor='black',
        capsize=5
    )
    
    # Add labels and title
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    
    # Adjust layout
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save figure
    plt.savefig(f'results/{title.replace(" ", "_").lower()}.png')
    
    # Show figure
    plt.show()

def main():
    """Main function."""
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Hyperparameter optimization example')
    parser.add_argument('--data', type=str, required=True, help='Path to data file')
    parser.add_argument('--model', type=str, default='xgboost', choices=['decision_tree', 'random_forest', 'xgboost'], help='Model type')
    parser.add_argument('--no-pruning', action='store_true', help='Disable feature pruning')
    parser.add_argument('--trials', type=int, default=50, help='Number of optimization trials')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = load_data(args.data)
    
    # Optimize and evaluate
    best_params, metrics, importances, top_features = optimize_and_evaluate(
        df, 
        model_type=args.model, 
        use_feature_pruning=not args.no_pruning,
        n_trials=args.trials
    )
    
    # Plot feature importance
    plot_feature_importance(importances, f'{args.model.capitalize()} Feature Importance')
    
if __name__ == '__main__':
    main()