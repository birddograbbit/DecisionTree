# Integrating JFK_DSRsi and MPO-3TF Strategies

This document outlines the steps required to add the intraday **JFK_DSRsi** and **MPO-3TF** strategies to the DecisionTree trading system. Both strategies currently exist only as standalone reference scripts under the `reference/` directory and must be converted into first-class strategy adapters that work with the existing infrastructure and support 1‑minute data.

## Source Reference
- `reference/jfk_dsrsi/jfk_dsrsi_hyperopt.py`
- `reference/jfk_dsrsi/tema_backtest_v3.py`
- `reference/MPO-3TF/MPO-3TF.py`

These scripts contain the indicator calculations, signal rules, and ATR-based risk management used by the strategies. Their logic should be ported directly when building the adapters.

## Integration Tasks
1. **Analyze Reference Scripts**
   - Extract formulas for Double-Smoothed RSI (DSRSI), Kase Permission Stochastic (KPS) with Jurik smoothing, and any auxiliary indicators.
   - Record all default parameters used for 5‑minute and 1‑minute variants. Final hyper‑optimized values will be supplied later.

2. **Implement Strategy Adapters**
   - Create `JFKDSRSIAdapter` and `MPO3TFAdapter` in `src/strategies/adapters/`.
   - Each adapter should subclass `BaseStrategy` and implement:
     - `initialize` for parameter handling.
     - `get_required_features` and `get_required_timeframes` (include `"1min"`).
     - `_add_indicators` to compute DSRSI, KPS/Jurik, multi‑timeframe RSI/Stoch/MFI, and ATR metrics.
     - `generate_signals` producing `signal`, `entry_price`, `stop_loss`, `take_profit`, and fixed `position_size`.
   - Risk rules should mirror the reference scripts and rely primarily on ATR-based stops. Additional time-based exits can be deferred.

3. **Register Strategies and Provide Configs**
   - Map keys `"jfk_dsrsi"` and `"mpo_3tf"` to the adapters in `src/strategies/strategy_registry.py`.
   - Add default configurations in `strategy_configs.py` with placeholder parameters and a default timeframe of `5min`; provide optional `1min` variants.

4. **Enable 1‑Minute Support**
   - Ensure data loading accepts `timeframe='1min'` via `load_1min_data` and the CLI.
   - Update feature pipelines and the `MultiTimeframeAggregator` to handle 1‑minute base data resampled to 5/10/15‑minute bars.
   - Create fixtures for 1‑minute OHLCV data. Example input format:

     ```
     date,open,high,low,close,volume,average,barCount
     2024-07-24 13:30:00+00:00,5505.84,5506.78,5501.68,5506.25,0,0,59
     2024-07-24 13:31:00+00:00,5506.34,5508.04,5505.05,5506.13,0,0,58
     2024-07-24 13:32:00+00:00,5506.54,5507.28,5504.73,5505.04,0,0,59
     2024-07-24 13:33:00+00:00,5504.37,5504.37,5500.91,5501.09,0,0,60
     ```

     Timestamps are UTC and ordered from oldest to newest.

5. **Testing and Validation**
   - Add unit tests verifying indicator calculations and signal generation for both strategies on 5‑minute and 1‑minute data.
   - Smoke-test using the CLI:

     ```bash
     python strategy_runner.py --data data/raw --model jfk_dsrsi --mode single --output results_jfk_5m --timeframe 5min --symbol SPY
     python strategy_runner.py --data data/raw --model jfk_dsrsi --mode single --output results_jfk_1m --timeframe 1min --symbol SPX
    python strategy_runner.py --data data/raw --model mpo_3tf --mode single --output results_mpo_5m --timeframe 5min --symbol SPY
    python strategy_runner.py --data data/raw --model mpo_3tf --mode single --output results_mpo_1m --timeframe 1min --symbol SPX
    ```

6. **Documentation and Meta-Strategy Integration**
   - Update project documentation to describe the strategies, their parameters, and usage examples.
   - Add mappings so `MetaStrategy` can select these strategies; regime definitions will need to be extended.

### Updated CLI Options

- `--asset-type` to specify `STOCK` or `INDEX`
- `--train-data` and `--test-data` for explicit file paths
- Timestamps are normalized to timezone-naive UTC during loading

## Open Questions
- **Default Parameters:** final hyper‑optimized parameter sets for 1‑minute and 5‑minute operation will be supplied later.
- **Position Sizing:** system currently lacks adaptive sizing; adapters should use a fixed `position_size` for now.
- **Hybrid Strategies:** once integrated, consider hybrids where ML models confirm or weight JFK_DSRsi and MPO-3TF signals.

---
Following this guide will elevate the reference scripts into fully supported intraday strategies within the DecisionTree platform, with optional expansion to meta‑strategy and hybrid configurations.
