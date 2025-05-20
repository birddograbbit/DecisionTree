# Decision Tree Classifier Trading Strategy

This repository contains an implementation of a decision tree classifier for trading S&P 500 stocks. The project is designed with a complete pipeline from data acquisition to live trading using Interactive Brokers (IBKR).

## Project Overview

### Objectives
- Implement a decision tree classifier for trading S&P 500 stocks
- Develop a complete pipeline from data acquisition to live trading
- Achieve a superior CAGR to maximum drawdown ratio compared to S&P 500 buy-and-hold
- Create a framework that allows for continuous improvement and model updating

### Success Metrics
- CAGR/Max Drawdown ratio > 0.40 (compared to S&P 500's ~0.18)
- Accuracy of directional prediction > 60%
- Strategy Sharpe ratio > 1.0
- Maximum drawdown < 25%

## Testing

Run the hyperparameter tuning example with:
```bash
python examples/hyperparameter_tuning_example.py --data data/raw/historical_data_STOCK_SPY_1_day2010-2025.csv --model xgboost --trials 20
```

To specifically test focal loss functionality:
```bash
python tests/test_focal_loss.py
```

See the [Decision Tree Classifier Strategy](Decision_Tree_Classifier_Strategy.md) for full documentation. The [v0.2 roadmap](v0.2_roadmap.md) describes planned improvements for the next release.

To audit features using permutation importance:
```bash
python run_feature_audit.py --data data/raw/historical_data_STOCK_SPY_1_day2010-2025.csv --model random_forest --output results/audit
```

Run hyperparameter optimization before executing strategies:
```bash
python optimize_hyperparameters.py --data data/raw --model all --output data/hyperparameters
```
