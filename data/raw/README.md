# Raw Data Directory

This directory contains raw price data files obtained from IBKR or other sources.

## Expected Files

- `historical_data_STOCK_SPY_1_day2000-2009.csv`
- `historical_data_STOCK_SPY_1_day2010-2025.csv`
- `historical_data_STOCK_SPY_5_mins_*.csv`
- `historical_data_STOCK_SPY_1_min_*.csv`
- `historical_data_INDEX_SPX_5_mins_*.csv`
- `historical_data_INDEX_SPX_1_min_*.csv`

Intraday filenames follow the pattern `historical_data_{ASSET}_{SYMBOL}_{TIMEFRAME}_{DATE-RANGE}.csv`. `ASSET` is either `STOCK` or `INDEX` and timestamps should be UTC without timezone information.
