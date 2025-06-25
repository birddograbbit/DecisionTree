# Transformer Integration Guide

This guide explains how to migrate the prototype Transformer modules into the production `src` tree and test the hybrid system.

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r scripts/requirements_transformer.txt
   ```
2. **Run the migration script**
   ```bash
   bash scripts/migrate_transformer_modules.sh
   ```
3. **Execute tests**
   ```bash
   pytest tests/test_transformer_integration.py
   ```
