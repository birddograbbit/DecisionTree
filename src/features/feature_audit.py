"""
Feature importance analysis and selection module.

This module provides functions for analyzing and selecting features
based on permutation importance and feature correlations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

def calculate_permutation_importance(model, X, y, n_repeats=10, random_state=42):
    """
    Calculate permutation importance for features.
    
    Parameters:
    -----------
    model : object
        Trained model with predict or predict_proba method
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target values
    n_repeats : int, default=10
        Number of times to permute each feature
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with feature importance scores
    """
    # Create a pipeline with StandardScaler
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    # Calculate permutation importance
    if hasattr(model, 'predict_proba'):
        scoring = 'roc_auc'  # Use ROC AUC for probabilistic models
    else:
        scoring = 'accuracy'  # Use accuracy for non-probabilistic models
    
    # Calculate permutation importance
    result = permutation_importance(
        pipe, X, y, n_repeats=n_repeats, 
        random_state=random_state, scoring=scoring
    )
    
    # Create DataFrame for results
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance_mean': result.importances_mean,
        'importance_std': result.importances_std
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance_mean', ascending=False)
    
    return importance_df

def calculate_feature_correlations(X):
    """
    Calculate correlations between features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
        
    Returns:
    --------
    pd.DataFrame
        Correlation matrix
    """
    return X.corr()

def find_highly_correlated_features(X, threshold=0.9):
    """
    Find highly correlated feature pairs.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    threshold : float, default=0.9
        Correlation threshold
        
    Returns:
    --------
    list
        List of (feature1, feature2, correlation) tuples
    """
    # Calculate correlation matrix
    corr_matrix = calculate_feature_correlations(X)
    
    # Find pairs of highly correlated features
    correlated_features = []
    
    # Get upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find feature pairs with correlation greater than threshold
    for col in upper.columns:
        for idx, value in upper[col].items():
            if abs(value) > threshold:
                correlated_features.append((idx, col, value))
    
    # Sort by absolute correlation
    correlated_features.sort(key=lambda x: abs(x[2]), reverse=True)
    
    return correlated_features

def select_features_by_importance(importance_df, top_n=None, threshold=0.01):
    """
    Select top features based on importance.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with feature importance scores
    top_n : int or None, default=None
        Number of top features to select
    threshold : float, default=0.01
        Minimum importance threshold
        
    Returns:
    --------
    list
        List of selected feature names
    """
    if top_n is not None:
        # Select top N features
        selected_features = importance_df.head(top_n)['feature'].tolist()
    else:
        # Select features above threshold
        selected_features = importance_df[importance_df['importance_mean'] > threshold]['feature'].tolist()
    
    return selected_features

def prune_correlated_features(importance_df, correlated_features):
    """
    Prune highly correlated features, keeping the more important one.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with feature importance scores
    correlated_features : list
        List of (feature1, feature2, correlation) tuples
        
    Returns:
    --------
    list
        List of features to remove
    """
    # Create a dictionary for feature importance lookup
    importance_dict = dict(zip(importance_df['feature'], importance_df['importance_mean']))
    
    # Features to remove
    features_to_remove = []
    
    # Process each correlated pair
    for feat1, feat2, _ in correlated_features:
        # Get importance scores
        importance1 = importance_dict.get(feat1, 0)
        importance2 = importance_dict.get(feat2, 0)
        
        # Remove the less important feature
        if importance1 < importance2:
            if feat1 not in features_to_remove:
                features_to_remove.append(feat1)
        else:
            if feat2 not in features_to_remove:
                features_to_remove.append(feat2)
    
    return features_to_remove

def plot_feature_importance(importance_df, figsize=(12, 8), output_path=None):
    """
    Plot feature importance.
    
    Parameters:
    -----------
    importance_df : pd.DataFrame
        DataFrame with feature importance scores
    figsize : tuple, default=(12, 8)
        Figure size
    output_path : str or None, default=None
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot feature importance
    sns.barplot(
        x='importance_mean', y='feature',
        data=importance_df.head(20),  # Top 20 features
        xerr=importance_df.head(20)['importance_std'],
        ax=ax
    )
    
    ax.set_title('Feature Importance (Permutation)')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    # Add grid lines
    ax.grid(True, axis='x', alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot if output path is provided
    if output_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save figure
        plt.savefig(output_path)
    
    return fig

def plot_correlation_heatmap(X, figsize=(14, 12), output_path=None):
    """
    Plot correlation heatmap for features.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature matrix
    figsize : tuple, default=(14, 12)
        Figure size
    output_path : str or None, default=None
        Path to save the plot
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    # Calculate correlation matrix
    corr_matrix = calculate_feature_correlations(X)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        cmap='coolwarm',
        vmin=-1, vmax=1,
        annot=False,
        square=True,
        ax=ax
    )
    
    ax.set_title('Feature Correlation Matrix')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot if output path is provided
    if output_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save figure
        plt.savefig(output_path)
    
    return fig

def feature_audit(model, X, y, output_dir=None, correlation_threshold=0.9, importance_threshold=0.01):
    """
    Perform a comprehensive feature audit.
    
    Parameters:
    -----------
    model : object
        Trained model with predict or predict_proba method
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target values
    output_dir : str or None, default=None
        Directory to save results and plots
    correlation_threshold : float, default=0.9
        Threshold for feature correlation
    importance_threshold : float, default=0.01
        Threshold for feature importance
        
    Returns:
    --------
    dict
        Audit results including selected features, etc.
    """
    # Create output directory if provided
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Calculate feature importance
    importance_df = calculate_permutation_importance(model, X, y)
    
    # Find highly correlated features
    correlated_features = find_highly_correlated_features(X, threshold=correlation_threshold)
    
    # Identify features to remove based on correlation
    features_to_remove = prune_correlated_features(importance_df, correlated_features)
    
    # Select features based on importance
    selected_features = select_features_by_importance(importance_df, threshold=importance_threshold)
    
    # Final feature set (selected features minus those to be removed)
    final_features = [feat for feat in selected_features if feat not in features_to_remove]
    
    # Log results
    logger.info(f"Total features: {X.shape[1]}")
    logger.info(f"Highly correlated feature pairs: {len(correlated_features)}")
    logger.info(f"Features to remove due to correlation: {len(features_to_remove)}")
    logger.info(f"Features above importance threshold: {len(selected_features)}")
    logger.info(f"Final feature set: {len(final_features)}")
    
    # Create plots if output directory is provided
    if output_dir is not None:
        # Plot feature importance
        importance_plot_path = os.path.join(output_dir, 'feature_importance.png')
        plot_feature_importance(importance_df, output_path=importance_plot_path)
        
        # Plot correlation heatmap
        corr_plot_path = os.path.join(output_dir, 'feature_correlation.png')
        plot_correlation_heatmap(X, output_path=corr_plot_path)
        
        # Save results to CSV
        importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        
        # Save correlated features to CSV
        pd.DataFrame(correlated_features, columns=['feature1', 'feature2', 'correlation']).to_csv(
            os.path.join(output_dir, 'correlated_features.csv'), index=False
        )
        
        # Save final feature set to text file
        with open(os.path.join(output_dir, 'selected_features.txt'), 'w') as f:
            for feature in final_features:
                f.write(f"{feature}\n")
    
    # Return results
    return {
        'importance': importance_df,
        'correlated_features': correlated_features,
        'features_to_remove': features_to_remove,
        'selected_features': selected_features,
        'final_features': final_features
    }
