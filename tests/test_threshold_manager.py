from src.utils.threshold_manager import ThresholdManager


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
