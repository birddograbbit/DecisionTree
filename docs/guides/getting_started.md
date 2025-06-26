# Getting Started

This guide walks you through installing dependencies and training your first transformer model.

1. **Install Requirements**
   ```bash
   pip install -r requirements.txt -r requirements-testing.txt
   ```
2. **Prepare Data**
   Place CSV files inside `data/raw/` with columns such as `open`, `high`, `low`, `close`, `volume`.
3. **Train a Transformer Model**
   ```python
   from src.models.model_factory import ModelFactory
   from config import TRANSFORMER_CONFIG

   model = ModelFactory.create_model('transformer', **TRANSFORMER_CONFIG['default'])
   model.train(train_df.drop('target', axis=1), train_df['target'])
   ```
4. **Make Predictions**
   ```python
   preds = model.predict(test_df.drop('target', axis=1))
   ```
