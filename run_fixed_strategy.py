#!/usr/bin/env python
"""
Run the strategy with fixed regime adaptation

This script will run the fixed RegimeAdaptiveStrategy on the provided data.
"""

import os
import sys
import argparse
import strategy_runner

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(description='Run trading strategy with fixed regime adaptive strategy')
    
    parser.add_argument('--data', type=str, default='data/raw',
                        help='Path to historical data file or directory')
    
    parser.add_argument('--model', type=str, choices=['decision_tree', 'random_forest', 'xgboost', 'stacking'],
                        default='random_forest',
                        help='Model type for single mode (default: random_forest)')
    
    parser.add_argument('--output', type=str, default='results',
                        help='Directory to save results (default: results)')
    
    parser.add_argument('--train-end', type=str, default=None,
                        help='End date for training data (format: YYYY-MM-DD)')
    
    parser.add_argument('--symbol', type=str, default='SPY',
                        help='Trading symbol (default: SPY)')
    
    parser.add_argument('--calibrate', action='store_true',
                        help='Use probability calibration for Decision Tree and Random Forest models')
    
    parser.add_argument('--use-optimized', action='store_true',
                        help='Use optimized hyperparameters for models')
    
    args = parser.parse_args()
    
    # Running with regime_adaptive strategy only
    print("\nRunning strategy with fixed RegimeAdaptiveStrategy...")
    strategy_runner.run_single_strategy(
        data_path=args.data,
        model_type=args.model,
        output_dir=args.output,
        train_end_date=args.train_end,
        symbol=args.symbol,
        strategy_type='regime_adaptive',  # Force regime_adaptive
        calibrate=args.calibrate,
        use_optimized_params=args.use_optimized
    )

if __name__ == "__main__":
    main()
