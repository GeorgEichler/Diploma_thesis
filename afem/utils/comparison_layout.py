"""Shared figure geometry for Monte Carlo plots displayed side by side."""

import matplotlib.pyplot as plt

from afem.utils.publication_style import PANEL_FIGSIZE


def comparison_figure(*, with_colorbar=False):
    """Keep canvas and data axes identical, with space for a sample-count key."""
    fig = plt.figure(figsize=PANEL_FIGSIZE, layout="none")
    # The footer holds a horizontal color key or ensemble spread legend.
    # Sharing the data rectangle keeps separately included PDFs aligned.
    ax = fig.add_axes((0.22, 0.36, 0.74, 0.60))
    colorbar_ax = fig.add_axes((0.22, 0.10, 0.74, 0.025)) if with_colorbar else None
    return fig, ax, colorbar_ax
