import unittest
import pandas as pd
import numpy as np
from src.strategies.adapters.jfk_dsrsi_adapter import JFKDSRSIAdapter

class TestJFKDSRSIAdapter(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2023-01-01', periods=200, freq='5T')
        prices = np.linspace(100, 110, len(dates)) + np.random.normal(0, 0.5, len(dates))
        self.data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1_000, 5_000, len(dates))
        }, index=dates)
        self.config = {'symbol': 'TEST', 'timeframe': '5min'}

    def test_initialization(self):
        adapter = JFKDSRSIAdapter()
        adapter.initialize(self.config)
        self.assertEqual(adapter.dsrsi_length, 30)
        self.assertEqual(adapter.kps_length, 14)

    def test_signal_generation(self):
        adapter = JFKDSRSIAdapter()
        adapter.initialize(self.config)
        features, _, dates = adapter.generate_features(self.data)
        signals = adapter.generate_signals(features, None, dates)
        self.assertIn('signal', signals.columns)
        self.assertTrue(set(signals['signal'].unique()).issubset({-1,0,1}))

    def test_backtest(self):
        adapter = JFKDSRSIAdapter()
        adapter.initialize(self.config)
        results = adapter.backtest(self.data)
        self.assertIn('total_return', results)
        self.assertIn('trades', results)

    def test_stop_loss_enforced(self):
        adapter = JFKDSRSIAdapter()
        adapter.initialize(self.config)
        dates = self.data.index[:3]
        signals = pd.DataFrame({
            'signal': [0, 1, 1],
            'stop_loss': [np.nan, 99, 99],
            'take_profit': [np.nan, 105, 105],
            'entry_price': [np.nan, 100, 100]
        }, index=dates)
        prices = pd.DataFrame({
            'open': [100, 100, 100],
            'high': [101, 101, 101],
            'low': [99.5, 99.5, 98],
            'close': [100, 100, 100]
        }, index=dates)
        managed = adapter.apply_risk_management(signals, prices)
        self.assertEqual(managed.iloc[2]['signal'], 0)

if __name__ == '__main__':
    unittest.main()
