# Transformer-DecisionTree Integration Strategy

## Executive Summary

The transformer trading system and DecisionTree project offer complementary strengths that can be combined to create a more robust trading system. The transformer excels at capturing complex temporal patterns in price data, while decision trees provide interpretable rules and can effectively classify market regimes.

## Key Synergies

### 1. **Complementary Strengths**
- **Transformer**: Superior at sequence modeling, non-linear pattern recognition, and continuous price prediction
- **Decision Tree**: Excellent for interpretable rules, feature importance, and discrete classification tasks

### 2. **Ensemble Opportunities**
- Combine continuous predictions (transformer) with discrete signals (decision tree)
- Use decision trees for market regime classification
- Apply transformers for regime-specific price prediction

### 3. **Feature Sharing**
Both systems can benefit from shared:
- Technical indicators (RSI, Bollinger Bands, MA)
- Data preprocessing pipelines
- Feature engineering techniques

## Proposed Architecture

```
┌─────────────────────────────────────────────────┐
│                   Data Layer                     │
│  (Unified data loading and preprocessing)       │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│               Feature Engineering                │
│  (Shared technical indicators and features)     │
└──────────┬─────────────────────┬────────────────┘
           │                     │
┌──────────┴──────────┐ ┌───────┴──────────────┐
│  Decision Tree      │ │   Transformer        │
│    Component        │ │    Component         │
├─────────────────────┤ ├──────────────────────┤
│ • Market Regime     │ │ • Price Prediction   │
│ • Risk Assessment   │ │ • Pattern Recognition│
│ • Signal Generation │ │ • Sequence Modeling  │
└──────────┬──────────┘ └───────┬──────────────┘
           │                     │
┌──────────┴─────────────────────┴────────────────┐
│              Ensemble Layer                      │
│  (Combines predictions based on market state)   │
└─────────────────────┬───────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────┐
│            Strategy Execution                    │
│  (Trading decisions and risk management)        │
└─────────────────────────────────────────────────┘
```

## Integration Approaches

### Approach 1: Sequential Pipeline
```python
def hybrid_prediction(market_data):
    # Step 1: Classify market regime using Decision Tree
    market_regime = decision_tree_classifier.predict(market_data.features)
    
    # Step 2: Select appropriate transformer model
    if market_regime == 'trending':
        model = transformer_trend_model
    elif market_regime == 'ranging':
        model = transformer_range_model
    else:
        model = transformer_volatile_model
    
    # Step 3: Generate prediction
    price_prediction = model.predict(market_data.sequences)
    
    # Step 4: Apply decision tree for risk assessment
    risk_level = decision_tree_risk.predict(market_data.risk_features)
    
    # Step 5: Combine for final signal
    return generate_signal(price_prediction, risk_level)
```

### Approach 2: Parallel Ensemble
```python
def ensemble_prediction(market_data):
    # Get predictions from both models
    dt_prediction = decision_tree_model.predict(market_data.features)
    tf_prediction = transformer_model.predict(market_data.sequences)
    
    # Calculate confidence scores
    dt_confidence = calculate_dt_confidence(market_data)
    tf_confidence = calculate_tf_confidence(market_data)
    
    # Weighted combination
    final_prediction = (
        dt_prediction * dt_confidence + 
        tf_prediction * tf_confidence
    ) / (dt_confidence + tf_confidence)
    
    return final_prediction
```

### Approach 3: Meta-Learning
```python
class MetaLearningTrader:
    def __init__(self):
        self.base_models = {
            'dt_short': DecisionTreeModel(timeframe='5m'),
            'dt_medium': DecisionTreeModel(timeframe='1h'),
            'tf_short': TransformerModel(seq_length=30),
            'tf_long': TransformerModel(seq_length=100)
        }
        self.meta_learner = DecisionTreeRegressor()
    
    def predict(self, market_data):
        # Get predictions from all base models
        base_predictions = {}
        for name, model in self.base_models.items():
            base_predictions[name] = model.predict(market_data)
        
        # Meta-learner combines predictions
        features = prepare_meta_features(base_predictions, market_data)
        final_prediction = self.meta_learner.predict(features)
        
        return final_prediction
```

## Phase 1: Foundation (Completed) ✅

### Completed Tasks:
- ✅ Basic transformer implementation with PyTorch
- ✅ Sequence preparation utilities
- ✅ Technical indicators module
- ✅ Basic integration wrapper
- ✅ Hybrid strategy prototype
- ✅ Fixed 5 out of 6 critical issues

### Remaining Issue:
- ⚠️ Limited test coverage for transformer components

## Phase 2: Production Integration & Testing (Current Phase)

### Step 1: Expand Test Coverage (Priority: HIGH)

#### 1.1 Unit Tests for Core Components
```python
# tests/test_transformer_model.py
import pytest
import torch
import numpy as np
from scripts.transformer_model import TimeSeriesTransformer

class TestTimeSeriesTransformer:
    """Comprehensive tests for transformer model"""
    
    def test_model_initialization(self):
        """Test model can be initialized with various configurations"""
        configs = [
            {'feature_size': 10, 'num_layers': 2, 'd_model': 64},
            {'feature_size': 5, 'num_layers': 4, 'd_model': 128},
        ]
        for config in configs:
            model = TimeSeriesTransformer(**config)
            assert model is not None
    
    def test_forward_pass(self):
        """Test model forward pass with different batch sizes"""
        model = TimeSeriesTransformer(feature_size=9, seq_length=30)
        test_cases = [
            (1, 30, 9),   # Single sample
            (32, 30, 9),  # Normal batch
            (128, 30, 9), # Large batch
        ]
        for batch_size, seq_len, features in test_cases:
            x = torch.randn(batch_size, seq_len, features)
            output = model(x)
            assert output.shape == (batch_size, 1)
    
    def test_gradient_flow(self):
        """Ensure gradients flow properly through the model"""
        model = TimeSeriesTransformer(feature_size=9)
        x = torch.randn(16, 30, 9, requires_grad=True)
        output = model(x)
        loss = output.mean()
        loss.backward()
        assert x.grad is not None
```

#### 1.2 Integration Tests
```python
# tests/test_transformer_integration.py
class TestTransformerIntegration:
    """Test transformer integration with DecisionTree system"""
    
    def test_model_factory_integration(self):
        """Test transformer can be created via model factory"""
        from src.models.model_factory import create_model
        model = create_model('transformer', seq_length=30, n_features=9)
        assert hasattr(model, 'train')
        assert hasattr(model, 'predict')
    
    def test_data_pipeline_compatibility(self):
        """Test transformer works with existing data pipeline"""
        from src.data.data_loader import DataLoader
        from scripts.transformer_wrapper import TransformerModelWrapper
        
        data = DataLoader().load_data('SPY', '2023-01-01', '2023-12-31')
        model = TransformerModelWrapper()
        
        # Should handle standard data format
        X_train, y_train = prepare_features(data)
        model.train(X_train, y_train)
        predictions = model.predict(X_train)
        assert len(predictions) == len(X_train)
```

#### 1.3 Edge Case Tests
```python
# tests/test_transformer_edge_cases.py
class TestTransformerEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_data_handling(self):
        """Test behavior with empty datasets"""
        model = TransformerModelWrapper()
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            model.train(empty_df, np.array([]))
    
    def test_single_sample_prediction(self):
        """Test prediction with single sample"""
        model = create_trained_model()
        single_sample = pd.DataFrame(np.random.randn(1, 9))
        pred = model.predict(single_sample)
        assert pred.shape == (1,)
    
    def test_missing_features_handling(self):
        """Test handling of missing features"""
        model = TransformerModelWrapper(preparator_strict=False)
        data_missing_features = pd.DataFrame({
            'open': [1, 2, 3],
            'close': [1, 2, 3]
        })
        # Should work with warning, not error
        model.feature_columns = ['open', 'close', 'high', 'low']
        model.preparator = SequencePreparator(
            feature_columns=model.feature_columns,
            strict=False
        )
        model.preparator.fit(data_missing_features)
```

### Step 2: Complete Integration with Main System (Priority: HIGH)

#### 2.1 Move Modules to Proper Locations
```bash
# Migration script
#!/bin/bash
# scripts/migrate_transformer_modules.sh

echo "Migrating transformer modules to main system..."

# Create transformer directory structure
mkdir -p src/models/transformer
mkdir -p src/models/ensemble
mkdir -p src/features/transformers

# Move core transformer files
cp scripts/transformer_model.py src/models/transformer/
cp scripts/transformer_wrapper.py src/models/transformer/
cp scripts/sequence_preparation.py src/models/transformer/

# Move supporting files
cp scripts/technical_indicators_transformer.py src/features/transformers/
cp scripts/hybrid_strategy.py src/models/ensemble/

# Create __init__ files
echo "from .transformer_model import TimeSeriesTransformer" > src/models/transformer/__init__.py
echo "from .transformer_wrapper import TransformerModelWrapper" >> src/models/transformer/__init__.py

echo "Migration complete!"
```

#### 2.2 Update Model Factory
```python
# src/models/model_factory.py
def create_model(model_type: str, **kwargs) -> BaseModel:
    """Factory function to create models"""
    
    if model_type == 'decision_tree':
        from .decision_tree import DecisionTreeModel
        return DecisionTreeModel(**kwargs)
    
    elif model_type == 'random_forest':
        from .random_forest import RandomForestModel
        return RandomForestModel(**kwargs)
    
    elif model_type == 'xgboost':
        from .xgboost_model import XGBoostModel
        return XGBoostModel(**kwargs)
    
    elif model_type == 'transformer':
        from .transformer.transformer_wrapper import TransformerModelWrapper
        return TransformerModelWrapper(**kwargs)
    
    elif model_type == 'hybrid':
        from .ensemble.hybrid_strategy import HybridMLStrategy
        dt_model = create_model('decision_tree', **kwargs.get('dt_params', {}))
        tf_model = create_model('transformer', **kwargs.get('tf_params', {}))
        return HybridMLStrategy(dt_model, tf_model, **kwargs)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

#### 2.3 Configuration Integration
```python
# config.py - Add transformer configuration
TRANSFORMER_CONFIG = {
    'default': {
        'seq_length': 30,
        'prediction_length': 1,
        'n_features': 9,
        'd_model': 64,
        'n_heads': 8,
        'n_layers': 2,
        'dropout': 0.1,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 20,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    },
    'fast': {
        'seq_length': 20,
        'd_model': 32,
        'n_heads': 4,
        'n_layers': 1,
        'epochs': 10
    },
    'accurate': {
        'seq_length': 60,
        'd_model': 128,
        'n_heads': 16,
        'n_layers': 4,
        'epochs': 50
    }
}

# Hybrid model configurations
HYBRID_CONFIG = {
    'conservative': {
        'dt_weight': 0.7,
        'tf_weight': 0.3,
        'regime_adaptive': True
    },
    'balanced': {
        'dt_weight': 0.5,
        'tf_weight': 0.5,
        'regime_adaptive': True
    },
    'aggressive': {
        'dt_weight': 0.3,
        'tf_weight': 0.7,
        'regime_adaptive': True
    }
}
```

### Step 3: Performance Optimization (Priority: MEDIUM)

#### 3.1 GPU Acceleration
```python
# src/models/transformer/gpu_optimizer.py
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

class GPUOptimizedTransformer:
    """GPU-optimized transformer implementation"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.scaler = GradScaler()
        
        # Enable cudnn optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    
    def train_with_mixed_precision(self, dataloader, optimizer, epochs):
        """Train with automatic mixed precision for faster training"""
        for epoch in range(epochs):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                
                # Mixed precision training
                with autocast():
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                
                # Scale loss and backward
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
```

#### 3.2 Batch Prediction Optimization
```python
# src/models/transformer/batch_predictor.py
class BatchPredictor:
    """Optimized batch prediction for transformers"""
    
    def __init__(self, model, batch_size=256):
        self.model = model
        self.batch_size = batch_size
    
    def predict_large_dataset(self, data, show_progress=True):
        """Efficiently predict on large datasets"""
        from torch.utils.data import DataLoader, TensorDataset
        from tqdm import tqdm
        
        # Prepare data
        dataset = TensorDataset(data)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )
        
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            iterator = tqdm(dataloader) if show_progress else dataloader
            for batch in iterator:
                batch_data = batch[0].to(self.model.device)
                batch_pred = self.model(batch_data)
                predictions.append(batch_pred.cpu())
        
        return torch.cat(predictions)
```

#### 3.3 Model Quantization
```python
# src/models/transformer/quantization.py
import torch.quantization as quantization

def quantize_transformer(model, calibration_data):
    """Quantize transformer model for faster inference"""
    
    # Prepare model for quantization
    model.eval()
    model.qconfig = quantization.get_default_qconfig('fbgemm')
    
    # Prepare model
    quantization.prepare(model, inplace=True)
    
    # Calibrate with representative data
    with torch.no_grad():
        for data in calibration_data:
            model(data)
    
    # Convert to quantized model
    quantized_model = quantization.convert(model, inplace=False)
    
    return quantized_model
```

### Step 4: Documentation Enhancement (Priority: MEDIUM)

#### 4.1 API Documentation
```python
# docs/api/transformer.md
"""
# Transformer API Reference

## TransformerModelWrapper

Main interface for transformer integration with DecisionTree system.

### Constructor Parameters

- `seq_length` (int, default=30): Length of input sequences
- `prediction_length` (int, default=1): Number of steps to predict
- `n_features` (int, default=9): Number of input features
- `d_model` (int, default=64): Transformer model dimension
- `n_heads` (int, default=8): Number of attention heads
- `n_layers` (int, default=2): Number of transformer layers
- `dropout` (float, default=0.1): Dropout rate
- `learning_rate` (float, default=0.001): Learning rate
- `batch_size` (int, default=32): Training batch size
- `epochs` (int, default=20): Number of training epochs
- `device` (str, optional): Device for computation ('cuda' or 'cpu')
- `target_column` (str, default='close'): Target column for predictions
- `preparator_strict` (bool, default=False): Strict feature validation

### Methods

#### train(X, y)
Train the transformer model on provided data.

**Parameters:**
- `X` (pd.DataFrame or np.ndarray): Feature matrix
- `y` (pd.Series or np.ndarray): Target values

**Returns:**
- `self`: For method chaining

#### predict(X)
Generate predictions for given features.

**Parameters:**
- `X` (pd.DataFrame or np.ndarray): Feature matrix

**Returns:**
- `np.ndarray`: Predicted probabilities

### Example Usage

```python
from src.models.transformer import TransformerModelWrapper

# Initialize model
model = TransformerModelWrapper(
    seq_length=30,
    n_features=9,
    d_model=64,
    n_heads=8,
    epochs=20
)

# Train model
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```
"""
```

#### 4.2 Integration Guide
```markdown
# Transformer Integration Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r scripts/requirements_transformer.txt
```

### 2. Run Migration Script
```bash
bash scripts/migrate_transformer_modules.sh
```

### 3. Test Integration
```bash
pytest tests/test_transformer_integration.py
```

## Configuration

### Using Config Files
```yaml
# config/models/transformer.yaml
model:
  type: transformer
  parameters:
    seq_length: 30
    n_features: 9
    d_model: 64
    n_heads: 8
    device: cuda
```

### Programmatic Configuration
```python
from config import TRANSFORMER_CONFIG
model = create_model('transformer', **TRANSFORMER_CONFIG['default'])
```

## Performance Tuning

### GPU Acceleration
- Ensure CUDA is installed: `torch.cuda.is_available()`
- Use larger batch sizes (64-256) for GPU
- Enable mixed precision training

### Memory Optimization
- Reduce sequence length for limited memory
- Use gradient accumulation for large batches
- Clear cache between training runs

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce batch_size
   - Decrease seq_length
   - Use gradient checkpointing

2. **Slow Training**
   - Enable GPU acceleration
   - Use mixed precision
   - Increase batch size

3. **Poor Performance**
   - Increase model size (d_model, n_layers)
   - Tune hyperparameters
   - Add more training data
```

### Step 5: Advanced Features (Priority: LOW)

#### 5.1 Attention Visualization
```python
# src/models/transformer/attention_viz.py
import matplotlib.pyplot as plt
import seaborn as sns

class AttentionVisualizer:
    """Visualize transformer attention patterns"""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        
        # Register hooks to capture attention
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on attention layers"""
        for name, module in self.model.named_modules():
            if 'attention' in name:
                module.register_forward_hook(
                    lambda m, i, o: self._save_attention(name, o)
                )
    
    def visualize_attention(self, input_sequence, layer_idx=0, head_idx=0):
        """Visualize attention weights for specific layer and head"""
        # Get attention weights
        _ = self.model(input_sequence)
        attention = self.attention_weights[f'layer_{layer_idx}'][head_idx]
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention.detach().cpu().numpy(),
            cmap='Blues',
            cbar=True,
            square=True
        )
        plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Keys')
        plt.ylabel('Queries')
        return plt.gcf()
```

#### 5.2 Multi-Task Learning
```python
# src/models/transformer/multi_task.py
class MultiTaskTransformer(nn.Module):
    """Transformer with multiple prediction heads"""
    
    def __init__(self, base_transformer, task_configs):
        super().__init__()
        self.base = base_transformer
        self.task_heads = nn.ModuleDict({
            task_name: self._create_task_head(config)
            for task_name, config in task_configs.items()
        })
    
    def _create_task_head(self, config):
        """Create task-specific prediction head"""
        return nn.Sequential(
            nn.Linear(config['input_dim'], config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'], config['output_dim'])
        )
    
    def forward(self, x, task_name=None):
        """Forward pass with optional task specification"""
        base_features = self.base.get_features(x)
        
        if task_name:
            return self.task_heads[task_name](base_features)
        else:
            return {
                task: head(base_features)
                for task, head in self.task_heads.items()
            }
```

#### 5.3 Online Learning
```python
# src/models/transformer/online_learning.py
class OnlineTransformer:
    """Transformer with online/incremental learning capability"""
    
    def __init__(self, model, buffer_size=1000):
        self.model = model
        self.experience_buffer = deque(maxlen=buffer_size)
        self.update_frequency = 100
        self.updates_count = 0
    
    def online_update(self, new_data, new_labels):
        """Update model with new data"""
        # Add to experience buffer
        self.experience_buffer.extend(zip(new_data, new_labels))
        
        # Periodic updates
        if len(self.experience_buffer) >= self.update_frequency:
            self._perform_update()
    
    def _perform_update(self):
        """Perform model update with buffered data"""
        # Sample from buffer
        batch_size = min(32, len(self.experience_buffer))
        batch = random.sample(self.experience_buffer, batch_size)
        
        # Quick gradient update
        X_batch = torch.stack([x for x, _ in batch])
        y_batch = torch.tensor([y for _, y in batch])
        
        self.optimizer.zero_grad()
        loss = self.criterion(self.model(X_batch), y_batch)
        loss.backward()
        self.optimizer.step()
        
        self.updates_count += 1
```

### Step 6: Production Readiness (Priority: MEDIUM)

#### 6.1 Comprehensive Logging
```python
# src/models/transformer/logging_config.py
import logging
from datetime import datetime

class TransformerLogger:
    """Comprehensive logging for transformer models"""
    
    def __init__(self, log_dir='logs/transformer'):
        self.log_dir = log_dir
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging with multiple handlers"""
        # Create logger
        self.logger = logging.getLogger('transformer')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler for all logs
        fh = logging.FileHandler(
            f'{self.log_dir}/transformer_{datetime.now():%Y%m%d_%H%M%S}.log'
        )
        fh.setLevel(logging.DEBUG)
        
        # File handler for errors
        eh = logging.FileHandler(f'{self.log_dir}/errors.log')
        eh.setLevel(logging.ERROR)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        eh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(eh)
        self.logger.addHandler(ch)
    
    def log_training_start(self, config):
        """Log training configuration"""
        self.logger.info("="*50)
        self.logger.info("Starting transformer training")
        self.logger.info(f"Configuration: {config}")
        self.logger.info("="*50)
    
    def log_epoch(self, epoch, loss, metrics):
        """Log epoch results"""
        self.logger.info(
            f"Epoch {epoch}: Loss={loss:.4f}, "
            f"Metrics={metrics}"
        )
    
    def log_prediction(self, n_samples, time_taken):
        """Log prediction performance"""
        self.logger.debug(
            f"Predicted {n_samples} samples in {time_taken:.2f}s "
            f"({n_samples/time_taken:.1f} samples/sec)"
        )
```

#### 6.2 Error Recovery
```python
# src/models/transformer/error_recovery.py
class RobustTransformer:
    """Transformer with error recovery mechanisms"""
    
    def __init__(self, model_config, checkpoint_dir='checkpoints'):
        self.config = model_config
        self.checkpoint_dir = checkpoint_dir
        self.model = None
        self.last_checkpoint = None
    
    def train_with_recovery(self, data, labels, resume=True):
        """Train with automatic checkpoint and recovery"""
        try:
            # Try to resume from checkpoint
            if resume and self.last_checkpoint:
                self.load_checkpoint(self.last_checkpoint)
            
            # Training loop with periodic checkpoints
            for epoch in range(self.config['epochs']):
                try:
                    loss = self._train_epoch(data, labels)
                    
                    # Save checkpoint every 10 epochs
                    if epoch % 10 == 0:
                        self.save_checkpoint(epoch)
                    
                except Exception as e:
                    self.logger.error(f"Error in epoch {epoch}: {e}")
                    # Try to recover from last checkpoint
                    self.load_checkpoint(self.last_checkpoint)
                    continue
                    
        except Exception as e:
            self.logger.critical(f"Training failed: {e}")
            raise
    
    def predict_with_fallback(self, data):
        """Predict with fallback to simpler model"""
        try:
            return self.model.predict(data)
        except Exception as e:
            self.logger.warning(f"Transformer prediction failed: {e}")
            # Fallback to moving average
            return self.fallback_predictor.predict(data)
```

#### 6.3 Model Versioning
```python
# src/models/transformer/versioning.py
import json
from pathlib import Path

class ModelVersionManager:
    """Manage transformer model versions"""
    
    def __init__(self, model_dir='models/transformer'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.version_file = self.model_dir / 'versions.json'
        self.versions = self._load_versions()
    
    def save_model(self, model, metrics, config, tag=None):
        """Save model with versioning"""
        # Generate version ID
        version_id = self._generate_version_id()
        
        # Create version directory
        version_dir = self.model_dir / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = version_dir / 'model.pt'
        model.save(model_path)
        
        # Save metadata
        metadata = {
            'version_id': version_id,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'metrics': metrics,
            'tag': tag or 'untagged'
        }
        
        with open(version_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Update versions registry
        self.versions[version_id] = metadata
        self._save_versions()
        
        return version_id
    
    def load_model(self, version_id=None, tag=None):
        """Load specific model version"""
        if tag:
            version_id = self._find_version_by_tag(tag)
        elif version_id is None:
            version_id = self._get_latest_version()
        
        version_dir = self.model_dir / version_id
        model_path = version_dir / 'model.pt'
        
        # Load model
        model = TransformerModelWrapper.load(model_path)
        
        # Load metadata
        with open(version_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
```

#### 6.4 Performance Monitoring
```python
# src/models/transformer/monitoring.py
class PerformanceMonitor:
    """Monitor transformer model performance in production"""
    
    def __init__(self, model, alert_thresholds):
        self.model = model
        self.thresholds = alert_thresholds
        self.metrics_buffer = deque(maxlen=1000)
        self.degradation_alerts = []
    
    def monitor_prediction(self, features, true_label=None):
        """Monitor single prediction"""
        start_time = time.time()
        
        # Make prediction
        prediction = self.model.predict(features)
        
        # Measure latency
        latency = time.time() - start_time
        
        # Record metrics
        metrics = {
            'timestamp': datetime.now(),
            'latency': latency,
            'prediction': prediction
        }
        
        if true_label is not None:
            metrics['error'] = abs(prediction - true_label)
        
        self.metrics_buffer.append(metrics)
        
        # Check for degradation
        self._check_degradation()
        
        return prediction
    
    def _check_degradation(self):
        """Check for performance degradation"""
        if len(self.metrics_buffer) < 100:
            return
        
        recent_metrics = list(self.metrics_buffer)[-100:]
        
        # Check latency
        avg_latency = np.mean([m['latency'] for m in recent_metrics])
        if avg_latency > self.thresholds['max_latency']:
            self._raise_alert('high_latency', avg_latency)
        
        # Check accuracy (if labels available)
        errors = [m.get('error') for m in recent_metrics if m.get('error')]
        if errors:
            avg_error = np.mean(errors)
            if avg_error > self.thresholds['max_error']:
                self._raise_alert('high_error', avg_error)
```

## Timeline and Milestones

### Week 1-2: Testing & Quality Assurance
- [ ] Implement comprehensive test suite
- [ ] Achieve >90% code coverage
- [ ] Performance benchmarks
- [ ] Integration tests pass

### Week 3-4: System Integration
- [ ] Complete module migration
- [ ] Update all configuration files
- [ ] Integrate with backtesting framework
- [ ] End-to-end testing

### Week 5-6: Performance & Optimization
- [ ] GPU acceleration implemented
- [ ] Batch prediction optimized
- [ ] Model quantization tested
- [ ] Latency targets met (<10ms)

### Week 7-8: Documentation & Advanced Features
- [ ] Complete API documentation
- [ ] Production deployment guide
- [ ] Attention visualization tools
- [ ] Initial multi-task experiments

### Week 9-10: Production Hardening
- [ ] Logging framework complete
- [ ] Error recovery tested
- [ ] Model versioning operational
- [ ] Monitoring dashboard live

### Week 11-12: Final Testing & Deployment
- [ ] Production stress testing
- [ ] A/B testing framework
- [ ] Performance monitoring
- [ ] Go-live preparation

## Success Metrics

### Technical Metrics
- Test coverage: >90%
- Inference latency: <10ms (95th percentile)
- Training time: <30 minutes for 1 year of data
- Memory usage: <4GB for inference
- Model size: <100MB (quantized)

### Trading Performance
- Sharpe ratio: >1.5 (hybrid model)
- Win rate: >55%
- Maximum drawdown: <15%
- Risk-adjusted returns: >20% annually

### Operational Metrics
- System uptime: >99.9%
- Error recovery time: <1 minute
- Model update time: <5 minutes
- Alert response time: <30 seconds

## Risk Mitigation

### Technical Risks
1. **GPU compatibility issues**
   - Mitigation: CPU fallback, cloud GPU options
   
2. **Memory constraints**
   - Mitigation: Model quantization, batch processing
   
3. **Integration conflicts**
   - Mitigation: Gradual rollout, feature flags

### Performance Risks
1. **Model overfitting**
   - Mitigation: Regularization, validation sets
   
2. **Latency spikes**
   - Mitigation: Caching, pre-computation
   
3. **Accuracy degradation**
   - Mitigation: Online learning, model refresh

## Conclusion

The Phase 2 integration plan provides a comprehensive roadmap for productionizing the transformer-decision tree hybrid system. By following this structured approach with clear priorities and timelines, we can ensure a robust, scalable, and high-performance trading system that leverages the best of both architectures.