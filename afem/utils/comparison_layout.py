"""Figure geometry for comparison plots and compact mesh panels."""

import matplotlib.pyplot as plt

from afem.utils.publication_style import MESH_AXES_RECT, MESH_FIGSIZE, PANEL_FIGSIZE


def mesh_figure(dim=2):
    """Create a compact mesh panel with no axes, tick labels, or title."""
    fig = plt.figure(figsize=MESH_FIGSIZE, layout="none")
    ax = fig.add_axes(MESH_AXES_RECT, projection="3d" if dim == 3 else None)
    if dim == 2:
        ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, ax


def comparison_figure(*, with_colorbar=False):
    """Keep canvas and data axes identical, with space for a sample-count key."""
    fig = plt.figure(figsize=PANEL_FIGSIZE, layout="none")
    # The footer holds a horizontal color key or ensemble spread legend.
    # Sharing the data rectangle keeps separately included PDFs aligned.
    ax = fig.add_axes((0.22, 0.36, 0.74, 0.60))
    colorbar_ax = fig.add_axes((0.22, 0.10, 0.74, 0.025)) if with_colorbar else None
    return fig, ax, colorbar_ax
