# src/utils.py
"""
Utility functions for the decision tree trading strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoLocator


def set_pandas_options():
    """Set pandas options to avoid warnings."""
    # Set pandas option to avoid FutureWarning about downcasting arrays
    pd.set_option('future.no_silent_downcasting', True)
    

def format_date_axis(ax, rotation=45):
    """
    Format the x-axis of a plot to display dates properly.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to format
    rotation : int, default=45
        Rotation angle for x-tick labels
    """
    # Format x-axis with dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(AutoLocator())
    plt.setp(ax.get_xticklabels(), rotation=rotation)


def plot_time_series(ax, dates, values, label=None, color=None, alpha=1.0):
    """
    Plot a time series with dates on the x-axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    dates : pd.DatetimeIndex
        Dates for the x-axis
    values : array-like
        Values for the y-axis
    label : str, optional
        Label for the plot
    color : str, optional
        Color for the plot
    alpha : float, default=1.0
        Alpha (transparency) for the plot
    
    Returns:
    --------
    line : matplotlib.lines.Line2D
        The plotted line
    """
    # Convert dates to numeric format matplotlib can handle
    date_nums = mdates.date2num(dates.to_pydatetime())
    
    # Plot time series
    line = ax.plot(date_nums, values, label=label, color=color, alpha=alpha)[0]
    
    # Format x-axis
    format_date_axis(ax)
    
    return line


def plot_fill_between_dates(ax, dates, values, min_values=None, color='blue', alpha=0.3, label=None):
    """
    Plot a filled area between two time series with dates on the x-axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    dates : pd.DatetimeIndex
        Dates for the x-axis
    values : array-like
        Upper values for the y-axis
    min_values : array-like, optional
        Lower values for the y-axis (default: zeros)
    color : str, default='blue'
        Color for the fill
    alpha : float, default=0.3
        Alpha (transparency) for the fill
    label : str, optional
        Label for the fill
    """
    # Convert dates to numeric format matplotlib can handle
    date_nums = mdates.date2num(dates.to_pydatetime())
    
    # Use zeros if min_values not provided
    if min_values is None:
        min_values = np.zeros_like(values)
        
    # Use polygon fill to avoid isfinite error with fill_between
    ax.fill(np.append(date_nums, date_nums[::-1]),
            np.append(values, min_values[::-1]),
            color, alpha=alpha, label=label)
    
    # Format x-axis
    format_date_axis(ax)

