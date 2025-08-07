"""Parameter sweep for MetaStrategy performance tuning."""
import os
import csv
from itertools import product
from strategy_runner import run_single_strategy

PERFORMANCE_WINDOWS = [195, 390, 780]
SWITCH_COOLDOWNS = [39, 78, 156]


def main():
    data_path = 'data/raw'
    timeframe = '5min'
    symbol = 'SPY'
    results = []

    for pw, sc in product(PERFORMANCE_WINDOWS, SWITCH_COOLDOWNS):
        output_dir = f'meta_strategy_perf_test/pw_{pw}_sc_{sc}'
        res = run_single_strategy(
            data_path=data_path,
            model_type='meta_strategy',
            output_dir=output_dir,
            symbol=symbol,
            timeframe=timeframe,
            performance_window=pw,
            switch_cooldown=sc,
        )
        perf = res.get('performance', {})
        results.append({
            'performance_window': pw,
            'switch_cooldown': sc,
            'sharpe_ratio': perf.get('sharpe_ratio', 0),
            'max_drawdown': perf.get('max_drawdown', 0),
            'num_trades': perf.get('num_trades', 0),
        })

    os.makedirs('meta_strategy_perf_test', exist_ok=True)
    csv_path = os.path.join('meta_strategy_perf_test', 'sweep_results.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['performance_window', 'switch_cooldown', 'sharpe_ratio', 'max_drawdown', 'num_trades'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to {csv_path}")


if __name__ == '__main__':
    main()
