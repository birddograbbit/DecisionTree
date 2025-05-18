# src/models/decision_tree.py
"""
Module for decision tree model implementation.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def train_decision_tree(X_train, y_train, max_depth=None, min_samples_split=2):
    """
    Train a decision tree classifier.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix
    y_train : pd.Series
        Training target values
    max_depth : int, optional
        Maximum depth of the tree (default: None)
    min_samples_split : int, optional
        Minimum samples required to split an internal node (default: 2)
        
    Returns:
    --------
    DecisionTreeClassifier
        Trained decision tree model
    """
    model = DecisionTreeClassifier(
        criterion='gini',  # or 'entropy'
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    return model

def tune_decision_tree(X_train, y_train, X_val, y_val):
    """
    Tune decision tree hyperparameters.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training feature matrix
    y_train : pd.Series
        Training target values
    X_val : pd.DataFrame
        Validation feature matrix
    y_val : pd.Series
        Validation target values
        
    Returns:
    --------
    tuple
        (best_model, best_params)
    """
    # Parameter grid
    param_grid = {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Grid search
    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    # Best parameters
    best_params = grid_search.best_params_
    
    # Validate on validation set
    best_model = grid_search.best_estimator_
    val_accuracy = accuracy_score(y_val, best_model.predict(X_val))
    
    print(f"Best parameters: {best_params}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    return best_model, best_params

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier
        Trained model
    X_test : pd.DataFrame
        Test feature matrix
    y_test : pd.Series
        Test target values
        
    Returns:
    --------
    dict
        Performance metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calculate AUC if possible
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = None
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': conf_matrix
    }
    
    return metrics

def save_model(model, filepath, scaler=None):
    """
    Save model and scaler to disk.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier
        Trained model
    filepath : str
        Path to save model
    scaler : StandardScaler, optional
        Fitted scaler object
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Save model
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler if provided
    if scaler is not None:
        scaler_path = filepath.replace('.pkl', '_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    print(f"Model saved to {filepath}")
    if scaler is not None:
        print(f"Scaler saved to {scaler_path}")

def load_model(filepath):
    """
    Load model from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to saved model
        
    Returns:
    --------
    tuple
        (model, scaler)
    """
    # Load model
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    # Try to load scaler
    scaler = None
    scaler_path = filepath.replace('.pkl', '_scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    return model, scaler

def visualize_decision_tree(model, feature_names, class_names=['Down', 'Up']):
    """
    Visualize decision tree model.
    
    Parameters:
    -----------
    model : DecisionTreeClassifier
        Trained model
    feature_names : list
        List of feature names
    class_names : list, optional
        List of class names (default: ['Down', 'Up'])
        
    Returns:
    --------
    matplotlib.figure.Figure
        Matplotlib figure
    """
    # Set figure size
    plt.figure(figsize=(20, 10))
    
    # Plot tree
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=10
    )
    
    # Add title
    plt.title('Decision Tree Classifier', fontsize=14)
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()
