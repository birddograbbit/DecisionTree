# Transformer-DecisionTree Integration Plan (Ship-Fast Version)

## Executive Summary

This plan outlines a minimal-complexity, ship-fast approach to integrate the Transformer module into the existing DecisionTree trading system. The integration leverages existing architecture patterns to minimize code changes while adding powerful temporal pattern recognition capabilities.

## Integration Principles

1. **Minimal Complexity**: Use wrappers and adapters to integrate without modifying core systems
2. **Ship Fast**: Focus on getting a working hybrid system quickly, optimize later
3. **Preserve Existing**: Keep all current functionality intact
4. **Modular Design**: Transformer components as separate modules for easy maintenance

## Phase 1: Module Structure & Conversion (Day 1-2)

### 1.1 Create Transformer Module Structure
```
src/models/transformer/
├── __init__.py
├── transformer_model.py        # Core transformer architecture
├── transformer_wrapper.py      # BaseModel-compatible wrapper
├── sequence_preparation.py     # Data preparation for sequences
└── attention_layers.py        # Custom attention components
```

### 1.2 Convert Notebook to Python Scripts
- Extract transformer model class to `transformer_model.py`
- Extract data preparation logic to `sequence_preparation.py`
- Create technical indicator integration in existing `indicators.py`

## Phase 2: Integration Components (Day 3-4)

### 2.1 TransformerModelWrapper
Create wrapper class that implements BaseModel interface:

```python
# src/models/transformer/transformer_wrapper.py
class TransformerModelWrapper(BaseModel):
    def __init__(self, seq_length=30, **kwargs):
        self.model = TimeSeriesTransformer(**kwargs)
        self.seq_length = seq_length
        self.scaler = MinMaxScaler()
        
    def train(self, X, y):
        # Convert to sequences
        X_seq = self.prepare_sequences(X)
        # Train transformer
        self.model.train(X_seq, y)
        return self
        
    def predict(self, X):
        # Convert to sequences and predict
        X_seq = self.prepare_sequences(X)
        return self.model.predict(X_seq)
```

### 2.2 Update ModelFactory
Extend existing factory to support transformer models:

```python
# In src/models/model_factory.py
def create_model(model_type, **kwargs):
    if model_type == 'transformer':
        from .transformer.transformer_wrapper import TransformerModelWrapper
        return TransformerModelWrapper(**kwargs)
    # ... existing model types
```

## Phase 3: Hybrid Strategy Implementation (Day 5-6)

### 3.1 Create HybridStrategy
```python
# src/strategies/hybrid_strategy.py
class HybridStrategy(BaseStrategy):
    def __init__(self, dt_model, tf_model, regime_detector):
        self.dt_model = dt_model      # Decision tree model
        self.tf_model = tf_model      # Transformer model
        self.regime_detector = regime_detector
        
    def generate_signals(self, data):
        # 1. Detect market regime
        regime = self.regime_detector.detect(data)
        
        # 2. Get predictions from both models
        dt_signals = self.dt_model.predict(data)
        tf_signals = self.tf_model.predict(data)
        
        # 3. Combine based on regime
        if regime == 'trending':
            # Weight transformer higher for trends
            signals = 0.7 * tf_signals + 0.3 * dt_signals
        else:
            # Weight decision tree higher for ranging
            signals = 0.3 * tf_signals + 0.7 * dt_signals
            
        return signals
```

## Phase 4: Configuration & Testing (Day 7)

### 4.1 Update config.py
Add transformer-specific parameters:

```python
# Transformer settings
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

# Hybrid strategy weights
HYBRID_WEIGHTS = {
    'trending': {'transformer': 0.7, 'decision_tree': 0.3},
    'ranging': {'transformer': 0.3, 'decision_tree': 0.7},
    'volatile': {'transformer': 0.5, 'decision_tree': 0.5}
}
```

### 4.2 Create Test Script
```python
# test_hybrid_system.py
def test_hybrid_system():
    # Load data
    data = load_data('data/raw')
    
    # Create models
    dt_model = ModelFactory.create('random_forest')
    tf_model = ModelFactory.create('transformer')
    
    # Create hybrid strategy
    strategy = HybridStrategy(dt_model, tf_model)
    
    # Run backtest
    results = backtest(strategy, data)
    print(f"Sharpe Ratio: {results['sharpe_ratio']}")
```

## Implementation Timeline

| Day | Task | Deliverable |
|-----|------|-------------|
| 1-2 | Convert notebook & create module structure | Python scripts ready |
| 3-4 | Build integration components | Wrapper & factory updates |
| 5-6 | Implement hybrid strategy | Working hybrid system |
| 7   | Testing & configuration | Validated integration |

## Key Integration Points

### 1. Data Pipeline
- Use existing feature engineering pipeline
- Add sequence preparation as additional step
- Share technical indicators between models

### 2. Model Training
```python
# In strategy_runner.py
if args.model == 'hybrid':
    # Train decision tree component
    dt_model = ModelFactory.create('random_forest')
    dt_model.train(X_features, y)
    
    # Train transformer component
    tf_model = ModelFactory.create('transformer')
    tf_model.train(X_features, y)
    
    # Create hybrid strategy
    strategy = HybridStrategy(dt_model, tf_model)
```

### 3. Backtesting
- Use existing backtesting framework
- No changes needed to backtesting engine
- Hybrid strategy outputs standard signals

## Minimal Changes Required

### Existing Files to Modify:
1. `src/models/__init__.py` - Add transformer imports
2. `src/models/model_factory.py` - Add transformer case
3. `src/strategies/__init__.py` - Add hybrid strategy
4. `config.py` - Add transformer parameters
5. `strategy_runner.py` - Add hybrid option

### New Files to Create:
1. `src/models/transformer/` - All transformer modules
2. `src/strategies/hybrid_strategy.py` - Hybrid implementation
3. `test_hybrid_system.py` - Testing script

## Risk Mitigation

1. **Compatibility**: Wrapper pattern ensures no breaking changes
2. **Performance**: Start with small sequence lengths, optimize later
3. **Memory**: Implement batch processing for large datasets
4. **Testing**: Run parallel to existing system initially

## Future Enhancements (Post-MVP)

1. Advanced ensemble methods (stacking, boosting)
2. Dynamic weight adjustment based on performance
3. Real-time adaptation to market conditions
4. GPU acceleration for transformer
5. AutoML for hyperparameter optimization

## Success Metrics

- System runs without breaking existing functionality
- Hybrid model achieves >10% improvement in Sharpe ratio
- Latency remains under 100ms for predictions
- All existing tests pass with new components

## Conclusion

This ship-fast approach delivers a working hybrid system in 7 days by:
- Leveraging existing architecture patterns
- Using wrappers to avoid core changes
- Focusing on integration over optimization
- Building on proven components

The modular design allows for future enhancements without disrupting the core system.
