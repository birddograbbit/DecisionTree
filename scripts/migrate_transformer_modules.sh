#!/bin/bash
# Move prototype transformer modules into the production src tree

set -e

mkdir -p src/models/transformer src/models/ensemble src/features/transformers

cp scripts/transformer_model.py src/models/transformer/
cp scripts/transformer_wrapper.py src/models/transformer/
cp scripts/sequence_preparation.py src/models/transformer/
cp scripts/hybrid_strategy.py src/models/ensemble/
cp scripts/technical_indicators_transformer.py src/features/transformers/

echo "from .transformer_model import TimeSeriesTransformer" > src/models/transformer/__init__.py
echo "from .transformer_wrapper import TransformerModelWrapper" >> src/models/transformer/__init__.py

echo "from .hybrid_strategy import HybridMLStrategy" > src/models/ensemble/__init__.py

echo "Migration complete!"
