# Transformer Integration Guide

This guide explains how to work with the transformer modules inside the DecisionTree project.

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run tests**
   ```bash
   pytest -q
   ```
3. **Enable GPU acceleration**
   - Ensure `torch.cuda.is_available()` returns `True`.
   - Training will automatically use the GPU when available.
4. **Deploy Hybrid Model**
   - Create models via `ModelFactory.create_model('hybrid')`.
   - Use `strategy_runner.py --model hybrid` to run the strategy.
