"""Print-size typography and line weights inspired by the Storn paper figures."""

from functools import wraps

import matplotlib as mpl


# Set these to the intended width and height of one panel in the manuscript.
PANEL_WIDTH_CM = 6.5
PANEL_HEIGHT_CM = 6.5
PANEL_FIGSIZE = (PANEL_WIDTH_CM / 2.54, PANEL_HEIGHT_CM / 2.54)

PAPER_RC = {
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "font.size": 9,
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "axes.labelsize": 10,
    "axes.titlesize": 9,
    "axes.labelpad": 3,
    "axes.facecolor": "#f2f2f2",
    "axes.edgecolor": "#666666",
    "axes.linewidth": 0.5,
    "axes.grid": False,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.35,
    "ytick.minor.width": 0.35,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    "lines.linewidth": 0.8,
    "lines.markersize": 2.3,
    "lines.markeredgewidth": 0.35,
    "legend.fontsize": 7.5,
    "legend.fancybox": False,
    "legend.framealpha": 1,
    "legend.facecolor": "white",
    "legend.edgecolor": "#666666",
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 2,
    "legend.handletextpad": 0.5,
}


def paper_style(plot):
    """Apply publication defaults locally without changing the caller's style."""
    @wraps(plot)
    def styled(*args, **kwargs):
        with mpl.rc_context(PAPER_RC):
            return plot(*args, **kwargs)
    return styled
