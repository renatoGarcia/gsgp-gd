import matplotlib as mpl
import seaborn as sns
from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt


def plot_setup():
    sns.set_context('paper')
    sns.set_style('whitegrid')
    sns.set_palette('muted')
    mpl.rcParams.update({
        'text.usetex': True,
        'text.latex.unicode': True,
        'font.family': 'serif',
        'font.serif': [],
        'font.sans-serif': [],
        'font.monospace': [],
        'pgf.preamble': [
            r'\usepackage[utf8x]{inputenc}',
            r'\usepackage[T1]{fontenc}']
        })


def figure_size(width_scale, height_scale=None, aspect=None, text_width=390):
    """Create a figure size tuple in inches from textwidth scale

    Args:
        width_scale (float): horizontal textwidth scale
        text_width (float, default=390pt): width of the text space in pt

    Returns:
        list: [horizontal, vertical] figsize in inches
    """
    fig_width_pt = text_width
    # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27
    # Convert pt to inch
    golden_mean = (sqrt(5.0) - 1.0) / 2.0
    # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * width_scale

    if height_scale is None:
        # width_scale in inches
        fig_height = fig_width * golden_mean
    else:
        fig_height = fig_width * height_scale

    if aspect == 'equal':
        fig_size = (fig_width, fig_width)
    else:
        fig_size = (fig_width, fig_height)

    return fig_size


def time_plot(x, y, unit, **kwargs):
    import statsmodels.stats.api as sms

    df = (pd.DataFrame({'x': x, 'y': y, 'unit': unit})
          .pivot(index='unit', columns='x', values='y'))
    aa = sms.DescrStatsW(df)
    a, b = aa.tconfint_mean(alpha=0.05)
    mean = aa.mean
    plt.fill_between(df.columns, b, a, alpha=0.4)
    plt.plot(df.columns, mean, '-', **kwargs)
    plt.xlabel(x.name)
    plt.ylabel(y.name)
