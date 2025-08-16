from src.utils.threshold_manager import ThresholdManager
import numpy as np


def test_default_thresholds():
    tm = ThresholdManager()
    buy, sell = tm.get_thresholds()
    assert buy == 0.55
    assert sell == 0.45


def test_prob_to_signal_applies_thresholds():
    tm = ThresholdManager()
    assert tm.prob_to_signal(0.6) == 1
    assert tm.prob_to_signal(0.4) == -1
    assert tm.prob_to_signal(0.5) == 0


def test_variance_aware_fallback():
    tm = ThresholdManager({'use_adaptive_thresholds': 'always'})
    preds = np.full(100, 0.5)
    buy, sell = tm.get_thresholds(preds)
    assert buy == 0.52 and sell == 0.48
