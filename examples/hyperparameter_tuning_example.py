# examples/hyperparameter_tuning_example.py

"""
Example script for hyperparameter optimization and feature pruning.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

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

def find_optimal_threshold(y_true, y_pred_proba):
    """
    Find the optimal threshold to convert probabilities to binary predictions.
    
    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_pred_proba : array-like
        Predicted probabilities
        
    Returns:
    --------
    float
        Optimal threshold
    """
    # Calculate precision and recall for different thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 score for each threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # Add small epsilon to avoid division by zero
    
    # Find threshold with highest F1 score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # Return optimal threshold
    return optimal_threshold

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
    # Split data into train and test sets
    train_end_date = '2020-12-31'
    X_train, X_test, y_train, y_test, dates_train, dates_test, scaler = prepare_train_test_data(
        df, train_end_date=train_end_date
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    
    # Print class distribution
    print(f"Class distribution in training data: {np.bincount(y_train)}")
    print(f"Class distribution in testing data: {np.bincount(y_test)}")
    
    # Calculate class weights
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}
    print(f"Computed class weights: {class_weight_dict}")
    
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
    model = ModelFactory.create_model(model_type, **best_params)
    
    # Train model on unscaled data (scaling is done in pipeline)
    model.train(X_train, y_train)
    
    # Feature importance and pruning
    train_importances, test_importances, top_features = audit_features(
        model, X_train, y_train, X_test, y_test,
        n_repeats=config.FEATURE_AUDIT_N_REPEATS,
        n_top_features=config.TOP_N_FEATURES,
        random_state=config.RANDOM_STATE
    )
    
    print(f"Top {len(top_features)} features: {top_features}")
    
    # Evaluate on training data first to check for overfitting
    y_train_pred = model.predict(X_train)
    optimal_train_threshold = find_optimal_threshold(y_train, y_train_pred)
    y_train_pred_binary = (y_train_pred > optimal_train_threshold).astype(int)
    
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred_binary),
        'precision': precision_score(y_train, y_train_pred_binary, zero_division=0),
        'recall': recall_score(y_train, y_train_pred_binary, zero_division=0),
        'f1': f1_score(y_train, y_train_pred_binary, zero_division=0)
    }
    
    print(f"\nTraining data metrics (threshold={optimal_train_threshold:.4f}):")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Check probability distribution
    print(f"\nProbability distribution on training data:")
    print(f"  Min: {np.min(y_train_pred):.4f}")
    print(f"  Max: {np.max(y_train_pred):.4f}")
    print(f"  Mean: {np.mean(y_train_pred):.4f}")
    print(f"  Median: {np.median(y_train_pred):.4f}")
    print(f"  25th percentile: {np.percentile(y_train_pred, 25):.4f}")
    print(f"  75th percentile: {np.percentile(y_train_pred, 75):.4f}")
    
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
    
    # Check probability distribution
    print(f"\nProbability distribution on test data:")
    print(f"  Min: {np.min(y_pred):.4f}")
    print(f"  Max: {np.max(y_pred):.4f}")
    print(f"  Mean: {np.mean(y_pred):.4f}")
    print(f"  Median: {np.median(y_pred):.4f}")
    print(f"  25th percentile: {np.percentile(y_pred, 25):.4f}")
    print(f"  75th percentile: {np.percentile(y_pred, 75):.4f}")
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(y_test, y_pred)
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    
    # Try different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, optimal_threshold]
    print("\nPerformance with different thresholds:")
    print(f"{'Threshold':^10} | {'Accuracy':^10} | {'Precision':^10} | {'Recall':^10} | {'F1 Score':^10}")
    print("-" * 60)
    
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = None
    
    for threshold in thresholds:
        y_pred_binary = (y_pred > threshold).astype(int)
        
        threshold_metrics = {
            'accuracy': accuracy_score(y_test, y_pred_binary),
            'precision': precision_score(y_test, y_pred_binary, zero_division=0),
            'recall': recall_score(y_test, y_pred_binary, zero_division=0),
            'f1': f1_score(y_test, y_pred_binary, zero_division=0)
        }
        
        print(f"{threshold:^10.2f} | {threshold_metrics['accuracy']:^10.4f} | "
              f"{threshold_metrics['precision']:^10.4f} | {threshold_metrics['recall']:^10.4f} | "
              f"{threshold_metrics['f1']:^10.4f}")
        
        if threshold_metrics['f1'] > best_f1:
            best_f1 = threshold_metrics['f1']
            best_threshold = threshold
            best_metrics = threshold_metrics
    
    print(f"\nBest threshold: {best_threshold:.4f}")
    
    # Save hyperparameters
    save_hyperparameters(best_params, model_type)
    
    return best_params, best_metrics, train_importances, top_features

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