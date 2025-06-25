# Transformer Integration for DecisionTree Trading System

This directory contains the transformer module implementation for integration with the DecisionTree trading system.

## Files Overview

### Core Transformer Components

1. **transformer_model.py**
   - Core TimeSeriesTransformer implementation using PyTorch
   - Multi-head attention architecture for temporal pattern recognition
   - Configurable layers, dimensions, and attention heads

2. **sequence_preparation.py**
   - Converts tabular time series data to sequences
   - Handles scaling and sequence windowing
   - PyTorch Dataset implementation for efficient batching

3. **transformer_wrapper.py**
   - BaseModel-compatible wrapper for seamless integration
   - Handles training, prediction, and model persistence
   - Bridges transformer architecture with existing system

### Supporting Components

4. **technical_indicators_transformer.py**
   - Technical indicator calculations optimized for transformers
   - Includes RSI, Bollinger Bands, MACD, ATR, and more
   - Feature engineering specific to sequence models

5. **hybrid_strategy.py**
   - Combines DecisionTree and Transformer predictions
   - Dynamic weighting based on market regimes
   - Adaptive strategy with performance-based weight adjustment

6. **test_transformer_integration.py**
   - Example script demonstrating full integration
   - Shows data preparation, model training, and backtesting
   - Performance comparison between models

## Quick Start

### 1. Install Dependencies

```bash
pip install -r scripts/requirements_transformer.txt
```

Requires **Python 3.10+** and a compatible PyTorch build. Install the CPU or GPU
version of PyTorch depending on your hardware. The `ta` library is optional but
recommended.

### 2. Basic Usage

```python
from scripts.transformer_wrapper import TransformerModelWrapper
from scripts.technical_indicators_transformer import add_technical_indicators

# Prepare data
df = pd.read_csv('your_data.csv')
df = add_technical_indicators(df)

# Create and train transformer
transformer = TransformerModelWrapper(
    seq_length=30,
    n_features=9,
    d_model=64,
    n_heads=8,
    target_column='close',
    epochs=20
)

transformer.train(X_train, y_train)
predictions = transformer.predict(X_test)
```

### 3. Hybrid Strategy

```python
from scripts.hybrid_strategy import HybridTransformerStrategy

# Create hybrid strategy
strategy = HybridTransformerStrategy(
    dt_model=your_decision_tree_model,
    tf_model=transformer,
    regime_detector=your_regime_detector
)

# Generate signals
signals = strategy.generate_signals(data)
```

## Integration Steps

### Step 1: Copy Transformer Module

Copy the transformer files to your project:

```bash
cp scripts/transformer_*.py src/models/transformer/
cp scripts/sequence_preparation.py src/models/transformer/
```

### Step 2: Update Model Factory

Add transformer support to `src/models/model_factory.py`:

```python
def create_model(model_type, **kwargs):
    if model_type == 'transformer':
        from .transformer.transformer_wrapper import TransformerModelWrapper
        return TransformerModelWrapper(**kwargs)
    # ... existing models
```

### Step 3: Add Configuration

Update `config.py` with transformer parameters:

```python
TRANSFORMER_PARAMS = {
    'seq_length': 30,
    'n_features': 9,
    'd_model': 64,
    'n_heads': 8,
    'n_layers': 2,
    'dropout': 0.1,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 20
}
```

### Step 4: Create Hybrid Strategy

Add to `src/strategies/`:

```python
from src.strategies.base_strategy import BaseStrategy
from scripts.hybrid_strategy import HybridTransformerStrategy

class HybridMLStrategy(BaseStrategy, HybridTransformerStrategy):
    """Unified hybrid strategy for the system."""

    def __init__(self, dt_model, tf_model, regime_detector):
        BaseStrategy.__init__(self)
        HybridTransformerStrategy.__init__(
            self,
            dt_model=dt_model,
            tf_model=tf_model,
            regime_detector=regime_detector,
        )
```

Calling both parent constructors ensures attributes from `BaseStrategy` and
`HybridTransformerStrategy` are initialized correctly when using multiple
inheritance.

## Architecture Benefits

### Transformer Advantages
- Captures long-range dependencies in price data
- Parallel processing of sequences
- Attention mechanism highlights important time periods
- Better for trending markets

### Decision Tree Advantages
- Interpretable decision rules
- Handles non-linear relationships well
- Better for ranging/choppy markets
- Lower computational requirements

### Hybrid Benefits
- Combines strengths of both approaches
- Adaptive to different market conditions
- Improved risk-adjusted returns
- More robust predictions

## Performance Considerations

### Memory Usage
- Transformers require more memory than decision trees
- Batch size affects memory consumption
- Consider GPU acceleration for large datasets

### Training Time
- Initial training is slower than decision trees
- Use saved models for production
- Implement incremental learning for updates

### Inference Speed
- Transformer predictions are fast once trained
- Batch predictions for efficiency
- Cache sequence preparations

## Testing

Run the automated test suite:

```bash
pytest
```

Expected output:
- Model training confirmation
- Accuracy metrics for both models
- Hybrid strategy performance
- Backtest results with Sharpe ratio > 1.0

## Next Steps

1. **Optimization**
   - Hyperparameter tuning with Optuna
   - GPU acceleration setup
   - Distributed training for large datasets

2. **Enhancements**
   - Add more sophisticated attention mechanisms
   - Implement multi-task learning
   - Create market-specific transformer variants

3. **Production**
   - Real-time prediction pipeline
   - Model versioning and A/B testing
   - Performance monitoring dashboard

## Troubleshooting

### Import Errors
- Ensure PyTorch is installed: `pip install torch`
- Check Python path includes project root
- Verify all dependencies are installed

### Memory Issues
- Reduce batch size
- Decrease sequence length
- Use gradient accumulation

### Performance Issues
- Enable GPU if available
- Use mixed precision training
- Implement data parallelism

## References

- Original Transformer Paper: "Attention Is All You Need" (Vaswani et al., 2017)
- Financial Transformers: "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- Hybrid Systems: "Ensemble Methods in Machine Learning" (Dietterich, 2000)

## License

This integration follows the same license as the main DecisionTree project.
