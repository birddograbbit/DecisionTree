"""
Example script demonstrating transformer integration with DecisionTree system.

This script shows how to:
1. Load and prepare data
2. Train both decision tree and transformer models
3. Create a hybrid strategy
4. Run backtests
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import transformer modules
from scripts.transformer_wrapper import TransformerModelWrapper
from scripts.technical_indicators_transformer import (
    add_technical_indicators, 
    create_target_variable
)
from scripts.hybrid_strategy import HybridTransformerStrategy

# Import existing system modules (these would be the actual imports when integrated)
# from src.models.model_factory import ModelFactory
# from src.features.feature_engineering import FeatureEngineer
# from src.features.regime_detection import RegimeDetector
# from src.backtesting.backtester import Backtester


def load_sample_data(file_path):
    """
    Load sample stock data from CSV.
    
    Parameters:
    -----------
    file_path : str
        Path to CSV file
        
    Returns:
    --------
    pd.DataFrame
        Loaded data with datetime index
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert to datetime index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    
    # Ensure column names are lowercase
    df.columns = df.columns.str.lower()
    
    return df


def prepare_data_for_models(df):
    """
    Prepare data for both decision tree and transformer models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw OHLCV data
        
    Returns:
    --------
    dict
        Dictionary with prepared data
    """
    # Add technical indicators
    df_with_indicators = add_technical_indicators(df)
    
    # Create target variable
    df_with_indicators['target'] = create_target_variable(
        df_with_indicators, 
        forward_periods=1
    )
    
    # Remove NaN values
    df_clean = df_with_indicators.dropna()
    
    # Split features and target
    feature_columns = [col for col in df_clean.columns 
                      if col not in ['target']]
    
    X = df_clean[feature_columns]
    y = df_clean['target']
    
    # Split into train/test
    split_date = df_clean.index[int(len(df_clean) * 0.8)]
    
    X_train = X[X.index <= split_date]
    X_test = X[X.index > split_date]
    y_train = y[y.index <= split_date]
    y_test = y[y.index > split_date]
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'full_data': df_clean,
        'feature_columns': feature_columns
    }


def train_decision_tree_model(X_train, y_train):
    """
    Train a decision tree based model.
    
    Note: In actual integration, this would use ModelFactory
    from the existing system.
    """
    from sklearn.ensemble import RandomForestClassifier
    
    # Create and train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Create wrapper with predict method for compatibility
    class DTWrapper:
        def __init__(self, model):
            self.model = model
            
        def predict(self, X):
            return self.model.predict_proba(X)[:, 1]
            
        def train(self, X, y):
            self.model.fit(X, y)
            return self
    
    return DTWrapper(model)


def main():
    """Main execution function."""
    print("Transformer-DecisionTree Integration Example")
    print("=" * 50)
    
    # 1. Load data
    print("\n1. Loading data...")
    # Use sample data or load from file
    # df = load_sample_data('data/SPY_daily.csv')
    
    # For demonstration, create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    df = pd.DataFrame({
        'open': 100 + np.random.randn(len(dates)).cumsum(),
        'high': 102 + np.random.randn(len(dates)).cumsum(),
        'low': 98 + np.random.randn(len(dates)).cumsum(),
        'close': 100 + np.random.randn(len(dates)).cumsum(),
        'volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    print(f"Loaded {len(df)} rows of data")
    
    # 2. Prepare data
    print("\n2. Preparing data...")
    data = prepare_data_for_models(df)
    print(f"Training samples: {len(data['X_train'])}")
    print(f"Test samples: {len(data['X_test'])}")
    print(f"Features: {len(data['feature_columns'])}")
    
    # 3. Train decision tree model
    print("\n3. Training decision tree model...")
    dt_model = train_decision_tree_model(
        data['X_train'], 
        data['y_train']
    )
    dt_accuracy = (dt_model.predict(data['X_test']) > 0.5).mean()
    print(f"Decision tree test accuracy: {dt_accuracy:.3f}")
    
    # 4. Train transformer model
    print("\n4. Training transformer model...")
    tf_model = TransformerModelWrapper(
        seq_length=30,
        n_features=len(data['feature_columns']),
        d_model=64,
        n_heads=4,
        n_layers=2,
        epochs=10,  # Reduced for demo
        batch_size=32
    )
    
    tf_model.train(data['X_train'], data['y_train'])
    tf_predictions = tf_model.predict(data['X_test'])
    tf_accuracy = ((tf_predictions > 0.5) == data['y_test']).mean()
    print(f"Transformer test accuracy: {tf_accuracy:.3f}")
    
    # 5. Create hybrid strategy
    print("\n5. Creating hybrid strategy...")
    hybrid_strategy = HybridTransformerStrategy(
        dt_model=dt_model,
        tf_model=tf_model,
        regime_detector=None  # Could add regime detection here
    )
    
    # 6. Generate signals
    print("\n6. Generating trading signals...")
    signals = hybrid_strategy.generate_signals(data['X_test'])
    print(f"Generated {len(signals)} signals")
    print("\nSignal distribution:")
    print(f"Long signals: {(signals['position'] > 0).sum()}")
    print(f"Short signals: {(signals['position'] < 0).sum()}")
    print(f"Neutral: {(signals['position'] == 0).sum()}")
    
    # 7. Run simple backtest
    print("\n7. Running backtest...")
    test_data = data['full_data'][data['full_data'].index > data['X_train'].index[-1]]
    backtest_results = hybrid_strategy.backtest(test_data)
    
    print("\nBacktest Results:")
    print(f"Total Return: {backtest_results['total_return']:.2%}")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
    print(f"Number of Trades: {backtest_results['n_trades']}")
    
    # 8. Save models
    print("\n8. Saving models...")
    os.makedirs('models', exist_ok=True)
    tf_model.save('models/transformer_model.pt')
    print("Transformer model saved to models/transformer_model.pt")
    
    # 9. Performance comparison
    print("\n9. Performance Comparison:")
    print("-" * 40)
    print(f"{'Model':<20} {'Accuracy':<10} {'Sharpe':<10}")
    print("-" * 40)
    
    # Calculate individual model Sharpe ratios
    dt_signals = pd.DataFrame({
        'position': np.where(dt_model.predict(test_data[data['feature_columns']]) > 0.6, 1, 
                           np.where(dt_model.predict(test_data[data['feature_columns']]) < 0.4, -1, 0))
    }, index=test_data.index)
    
    tf_signals = pd.DataFrame({
        'position': np.where(tf_model.predict(test_data[data['feature_columns']]) > 0.6, 1,
                           np.where(tf_model.predict(test_data[data['feature_columns']]) < 0.4, -1, 0))
    }, index=test_data.index)
    
    price_returns = test_data['close'].pct_change()
    
    dt_returns = dt_signals['position'].shift(1) * price_returns
    dt_sharpe = np.sqrt(252) * dt_returns.mean() / dt_returns.std()
    
    tf_returns = tf_signals['position'].shift(1) * price_returns
    tf_sharpe = np.sqrt(252) * tf_returns.mean() / tf_returns.std()
    
    print(f"{'Decision Tree':<20} {dt_accuracy:<10.3f} {dt_sharpe:<10.2f}")
    print(f"{'Transformer':<20} {tf_accuracy:<10.3f} {tf_sharpe:<10.2f}")
    print(f"{'Hybrid':<20} {'N/A':<10} {backtest_results['sharpe_ratio']:<10.2f}")
    
    print("\n✅ Integration test completed successfully!")
    

if __name__ == "__main__":
    main()
