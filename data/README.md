# Data directory

This directory contains data files for the Decision Tree Trading Strategy project.

## Structure

- `raw/`: Raw price data from IBKR or other sources
- `processed/`: Preprocessed data ready for model training
- `models/`: Trained model files

## Data Sources

Data files should be formatted as CSV files with the following columns:
- date: Date in YYYY-MM-DD format
- open: Opening price
- high: Highest price
- low: Lowest price
- close: Closing price
- volume: Trading volume

Example files:
- historical_data_STOCK_SPY_1_day2000-2009.csv
- historical_data_STOCK_SPY_1_day2010-2025.csv
