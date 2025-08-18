import unittest
import pandas as pd
import numpy as np
from src.strategies.adapters.mpo_3tf_adapter import MPO3TFAdapter

class TestMPO3TFAdapter(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2023-01-01', periods=200, freq='1T')
        prices = np.linspace(50, 55, len(dates)) + np.random.normal(0, 0.2, len(dates))
        self.data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(500, 2000, len(dates))
        }, index=dates)
        self.config = {'symbol': 'TEST', 'timeframe': '1min'}

    def test_initialization(self):
        adapter = MPO3TFAdapter()
        adapter.initialize(self.config)
        self.assertEqual(adapter.mpo_length, 14)
        self.assertEqual(adapter.atr_length, 14)

    def test_signal_generation(self):
        adapter = MPO3TFAdapter()
        adapter.initialize(self.config)
        features, _, dates = adapter.generate_features(self.data)
        signals = adapter.generate_signals(features, None, dates)
        self.assertIn('signal', signals.columns)
        self.assertTrue(set(signals['signal'].unique()).issubset({-1,0,1}))

    def test_backtest(self):
        adapter = MPO3TFAdapter()
        adapter.initialize(self.config)
        results = adapter.backtest(self.data)
        self.assertIn('total_return', results)
        self.assertIn('trades', results)

    def test_stop_loss_enforced(self):
        adapter = MPO3TFAdapter()
        adapter.initialize(self.config)
        dates = self.data.index[:3]
        signals = pd.DataFrame({
            'signal': [0, -1, -1],
            'stop_loss': [np.nan, 51, 51],
            'take_profit': [np.nan, 45, 45],
            'entry_price': [np.nan, 50, 50]
        }, index=dates)
        prices = pd.DataFrame({
            'open': [50, 50, 50],
            'high': [50.5, 50.5, 52],
            'low': [49.5, 49, 46],
            'close': [50, 50, 50]
        }, index=dates)
        managed = adapter.apply_risk_management(signals, prices)
        self.assertEqual(managed.iloc[2]['signal'], 0)

if __name__ == '__main__':
    unittest.main()
