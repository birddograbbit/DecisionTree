"""
Unit tests for momentum strategy adapters.

This module tests the BB-RSI-ADX, TEMA, and Quod strategy adapters
to ensure they function correctly and generate valid signals.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.adapters.bbrsiadx_adapter import BBRSIADXAdapter
from src.strategies.adapters.tema_adapter import TEMAAdapter
from src.strategies.adapters.quod_adapter import QuodAdapter


class TestMomentumAdapters(unittest.TestCase):
    """Test suite for momentum strategy adapters."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate synthetic price data with trend
        trend = np.linspace(100, 120, len(dates))
        noise = np.random.normal(0, 2, len(dates))
        prices = trend + noise
        
        # Create OHLCV DataFrame
        self.test_data = pd.DataFrame({
            'open': prices * 0.99,
            'high': prices * 1.01,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates))
        }, index=dates)
        
        # Basic configuration
        self.config = {
            'symbol': 'TEST',
            'position_size': 0.1
        }
    
    def test_bbrsiadx_initialization(self):
        """Test BB-RSI-ADX adapter initialization."""
        adapter = BBRSIADXAdapter()
        adapter.initialize(self.config)
        
        # Check default parameters
        self.assertEqual(adapter.bb_period, 20)
        self.assertEqual(adapter.rsi_period, 14)
        self.assertEqual(adapter.adx_primary_threshold, 20)
        self.assertEqual(adapter.adx_secondary_threshold, 40)
        
        # Check required features
        features = adapter.get_required_features()
        self.assertIn('rsi', features)
        self.assertIn('bb_upper', features)
        self.assertIn('adx', features)
        
        # Check required timeframes
        timeframes = adapter.get_required_timeframes()
        self.assertEqual(len(timeframes), 2)  # Primary + 4h
    
    def test_tema_initialization(self):
        """Test TEMA adapter initialization."""
        adapter = TEMAAdapter()
        adapter.initialize(self.config)
        
        # Check default parameters
        self.assertEqual(adapter.tema_primary_fast, 10)
        self.assertEqual(adapter.tema_primary_slow, 80)
        self.assertEqual(adapter.adx_threshold, 40)
        self.assertTrue(adapter.use_dual_timeframe)
        
        # Check required features
        features = adapter.get_required_features()
        self.assertIn('tema_fast', features)
        self.assertIn('tema_slow', features)
        self.assertIn('cmo', features)
    
    def test_quod_initialization(self):
        """Test Quod adapter initialization."""
        adapter = QuodAdapter()
        adapter.initialize(self.config)
        
        # Check default parameters
        self.assertTrue(adapter.use_stoch_reversal)
        self.assertTrue(adapter.use_stoch_pullback)
        self.assertTrue(adapter.use_d60_trend_exit)
        self.assertEqual(adapter.stoch_k_period, 14)
        
        # Check required features
        features = adapter.get_required_features()
        self.assertIn('stoch_k', features)
        self.assertIn('stoch_d', features)
        self.assertIn('d60_trend', features)
    
    def test_bbrsiadx_signal_generation(self):
        """Test BB-RSI-ADX signal generation."""
        adapter = BBRSIADXAdapter()
        adapter.initialize(self.config)
        
        # Generate features
        features, _, dates = adapter.generate_features(self.test_data)
        
        # Generate signals
        signals = adapter.generate_signals(features, None, dates)
        
        # Validate signal structure
        self.assertIn('signal', signals.columns)
        self.assertIn('entry_price', signals.columns)
        self.assertIn('stop_loss', signals.columns)
        self.assertIn('take_profit', signals.columns)
        self.assertIn('position_size', signals.columns)
        
        # Check signal values are valid
        self.assertTrue(all(signals['signal'].isin([0, 1, -1])))
        self.assertTrue(all(signals['position_size'] >= 0))
    
    def test_tema_signal_generation(self):
        """Test TEMA signal generation."""
        adapter = TEMAAdapter()
        config = self.config.copy()
        config['use_dual_timeframe'] = False  # Simplify for testing
        adapter.initialize(config)
        
        # Generate features
        features, _, dates = adapter.generate_features(self.test_data)
        
        # Generate signals
        signals = adapter.generate_signals(features, None, dates)
        
        # Validate signal structure
        self.assertIn('signal', signals.columns)
        self.assertIn('entry_price', signals.columns)
        self.assertIn('stop_loss', signals.columns)
        self.assertIn('take_profit', signals.columns)
        
        # Check that stop loss and take profit are set correctly for long signals
        long_signals = signals[signals['signal'] == 1]
        if len(long_signals) > 0:
            self.assertTrue(all(long_signals['stop_loss'] < long_signals['entry_price']))
            self.assertTrue(all(long_signals['take_profit'] > long_signals['entry_price']))
    
    def test_quod_signal_generation(self):
        """Test Quod signal generation."""
        adapter = QuodAdapter()
        adapter.initialize(self.config)
        
        # Generate features
        features, _, dates = adapter.generate_features(self.test_data)
        
        # Generate signals
        signals = adapter.generate_signals(features, None, dates)
        
        # Validate signal structure
        self.assertIn('signal', signals.columns)
        self.assertIn('signal_type', signals.columns)
        self.assertIn('stop_loss', signals.columns)
        self.assertIn('take_profit', signals.columns)
        
        # Check signal types
        valid_signal_types = ['', 'reversal', 'pullback', 'exit', 'eod_exit']
        self.assertTrue(all(signals['signal_type'].isin(valid_signal_types)))
    
    def test_order_management_config(self):
        """Test order management configurations."""
        # BB-RSI-ADX uses limit orders
        bb_adapter = BBRSIADXAdapter()
        bb_config = bb_adapter.get_order_management_config()
        self.assertEqual(bb_config['order_type'], 'limit')
        self.assertEqual(bb_config['limit_offset_atr'], 0)
        
        # TEMA uses limit orders with ATR offset
        tema_adapter = TEMAAdapter()
        tema_config = tema_adapter.get_order_management_config()
        self.assertEqual(tema_config['order_type'], 'limit')
        self.assertEqual(tema_config['limit_offset_atr'], 1)
        
        # Quod uses market orders
        quod_adapter = QuodAdapter()
        quod_config = quod_adapter.get_order_management_config()
        self.assertEqual(quod_config['order_type'], 'market')
    
    def test_backtest_methods(self):
        """Test that backtest methods work without errors."""
        adapters = [BBRSIADXAdapter(), TEMAAdapter(), QuodAdapter()]
        
        for adapter in adapters:
            adapter.initialize(self.config)
            
            # Run backtest
            results = adapter.backtest(self.test_data)
            
            # Validate results structure
            self.assertIn('total_return', results)
            self.assertIn('sharpe_ratio', results)
            self.assertIn('max_drawdown', results)
            self.assertIn('num_trades', results)
            self.assertIn('win_rate', results)
            self.assertIn('strategy', results)
    
    def test_risk_management(self):
        """Test risk management methods."""
        adapters = [BBRSIADXAdapter(), TEMAAdapter(), QuodAdapter()]
        
        for adapter in adapters:
            adapter.initialize(self.config)
            
            # Generate dummy signals
            dates = self.test_data.index[:10]
            signals = pd.DataFrame({
                'signal': [0, 1, 0, -1, 0, 1, 0, 0, -1, 0],
                'stop_loss': [np.nan, 95, np.nan, 105, np.nan, 97, np.nan, np.nan, 103, np.nan],
                'take_profit': [np.nan, 105, np.nan, 95, np.nan, 103, np.nan, np.nan, 97, np.nan]
            }, index=dates)
            
            # Apply risk management
            managed_signals = adapter.apply_risk_management(
                signals, self.test_data.loc[dates]
            )
            
            # Signals should be unchanged (risk management is built into signal generation)
            pd.testing.assert_frame_equal(signals, managed_signals)
    
    def test_parameter_customization(self):
        """Test that custom parameters are applied correctly."""
        # Test BB-RSI-ADX with custom parameters
        bb_config = self.config.copy()
        bb_config.update({
            'bb_period': 30,
            'rsi_period': 21,
            'adx_primary_threshold': 25
        })
        bb_adapter = BBRSIADXAdapter()
        bb_adapter.initialize(bb_config)
        
        self.assertEqual(bb_adapter.bb_period, 30)
        self.assertEqual(bb_adapter.rsi_period, 21)
        self.assertEqual(bb_adapter.adx_primary_threshold, 25)
        
        # Test TEMA with custom parameters
        tema_config = self.config.copy()
        tema_config.update({
            'tema_primary_fast': 15,
            'tema_primary_slow': 60,
            'use_dual_timeframe': False
        })
        tema_adapter = TEMAAdapter()
        tema_adapter.initialize(tema_config)
        
        self.assertEqual(tema_adapter.tema_primary_fast, 15)
        self.assertEqual(tema_adapter.tema_primary_slow, 60)
        self.assertFalse(tema_adapter.use_dual_timeframe)
        
        # Test Quod with custom parameters
        quod_config = self.config.copy()
        quod_config.update({
            'use_stoch_reversal': False,
            'use_trailing_stop': True,
            'long_tp_perc': 1.02
        })
        quod_adapter = QuodAdapter()
        quod_adapter.initialize(quod_config)
        
        self.assertFalse(quod_adapter.use_stoch_reversal)
        self.assertTrue(quod_adapter.use_trailing_stop)
        self.assertEqual(quod_adapter.long_tp_perc, 1.02)


if __name__ == '__main__':
    unittest.main()