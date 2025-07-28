# RegimeAdaptiveStrategy Date Ambiguity Fix

## Issue
Pandas ambiguity error: "'date' is both an index level and a column label, which is ambiguous."

This error occurs when a DataFrame has 'date' as both an index name and a column name, causing pandas to throw an error when trying to access the date information.

## Root Cause
The RegimeAdaptiveStrategy's `generate_signals` method was creating a DataFrame with dates as the index, then adding a 'date' column with the same data, creating ambiguity.

## Solution
Consistently handle dates by:
1. Resetting any date index on input features
2. Creating signals DataFrame without date index
3. Using explicit date parameter instead of extracting from features

## Changes Made

### In `generate_signals` method (line ~447):
```python
# Before:
signals = pd.DataFrame(index=dates)
signals['date'] = dates  # Creates ambiguity!

# After:
# Handle date ambiguity upfront
if isinstance(features, pd.DataFrame):
    if features.index.name == 'date' and 'date' in features.columns:
        features = features.reset_index(drop=True)
    elif features.index.name == 'date':
        features = features.reset_index(drop=True)

# Create signals without date index
signals = pd.DataFrame()
signals['date'] = dates  # No ambiguity
```

## Impact
- Fixes the error that was causing RegimeAdaptiveStrategy to fall back to base TrendFollowingStrategy
- Enables proper regime-specific trading behavior
- Allows the strategy to adapt parameters based on market conditions

## Testing
No rollback needed - this is a bug fix with no functional changes to the strategy logic.

### Test Commands
```bash
# Test regime adaptive strategy
python strategy_runner.py --data data/raw --model random_forest --strategy regime_adaptive --mode single --output regime_fixed_test

# Verify no date ambiguity errors in logs
grep -i "date.*ambiguous" regime_fixed_test/*.log
```

## Expected Behavior After Fix
1. No "date ambiguity" errors in logs
2. RegimeAdaptiveStrategy should detect and use different market regimes
3. Performance metrics should show regime-specific adaptations
4. Log should show messages like:
   - "Detecting regimes for the full dataset"
   - "Current regime: [regime_name]"
   - "Using regime-specific parameters"

## Related Code Sections
- `src/strategies/regime_adaptive_strategy.py`: Lines 447-460 (generate_signals)
- `src/strategies/regime_adaptive_strategy.py`: Lines 723-730 (backtest method - already has fix)

## Future Prevention
When working with DataFrames containing dates:
1. Avoid using dates as both index and column
2. Be explicit about date handling
3. Reset index before operations that might create ambiguity
4. Use consistent date representation throughout the pipeline