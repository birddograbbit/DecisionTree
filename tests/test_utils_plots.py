import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from pathlib import Path

UTILS_PATH = Path(__file__).resolve().parents[1] / "src" / "utils.py"
spec = importlib.util.spec_from_file_location("core_utils", UTILS_PATH)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)
set_pandas_options = utils.set_pandas_options
plot_time_series = utils.plot_time_series
plot_fill_between_dates = utils.plot_fill_between_dates


def test_set_pandas_options():
    set_pandas_options()
    assert pd.get_option('future.no_silent_downcasting') is True


def test_plot_time_series_and_fill():
    dates = pd.date_range('2021-01-01', periods=5)
    values = np.arange(5)

    fig, ax = plt.subplots()
    line = plot_time_series(ax, dates, values, label="test", color="blue")
    assert line.get_label() == "test"
    assert len(line.get_xdata()) == 5

    fig2, ax2 = plt.subplots()
    plot_fill_between_dates(ax2, dates, values, min_values=values-1, label="fill")
    assert len(ax2.patches) == 1
