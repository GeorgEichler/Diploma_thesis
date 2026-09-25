"""Plot Monte Carlo sample-count studies with a compact, ordered color key."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D


def plot_sample_comparison(runs: list[dict], output_dir: Path):
    """Plot each history at its own DOFs; darker blue means more samples.

    Colors identify ordered, discrete sample counts, not an interpolation of
    the solution. Errors are relative H1 seminorm and L2 errors; the shortened
    H1 norm label follows the other comparison plots.
    """
    if not runs:
        raise ValueError("At least one completed run is needed")
    runs = sorted(runs, key=lambda run: run["samples"])
    shades = np.linspace(0.4, 0.95, len(runs)) if len(runs) > 1 else [0.7]
    cmap = ListedColormap(plt.get_cmap("Blues")(shades))
    norm = BoundaryNorm(np.arange(len(runs) + 1) - 0.5, cmap.N)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (output_dir / "sample_errors.png", output_dir / "sample_estimator.png")
    plots = (
        (("relative_h1_semi_error_ref", "-"), ("relative_l2_error_ref", "--")),
        (("estimator", "-"),),
    )

    for panel, (metrics, path) in enumerate(zip(plots, paths)):
        fig, ax = plt.subplots(figsize=(8, 5))
        has_zero = False
        for index, run in enumerate(runs):
            ndofs = np.array([entry["ndofs"] for entry in run["history"]])
            for key, style in metrics:
                values = np.array([entry[key] for entry in run["history"]])
                has_zero = has_zero or bool(np.any(values == 0))
                ax.plot(ndofs, np.where(values > 0, values, np.nan),
                        color=cmap(index), linestyle=style, marker="o", markersize=3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("ndofs")
        ax.set_ylabel("relative error" if panel == 0 else r"Residual $\eta$")
        ax.grid(True, which="both", ls=":")
        if panel == 0:
            ax.legend(handles=[
                Line2D([], [], color="black", linestyle="-", label=r"$H^1$ norm"),
                Line2D([], [], color="black", linestyle="--", label=r"$L^2$ norm"),
            ])
        # A numeric color key replaces a separate legend entry for every curve.
        ticks = np.unique(np.linspace(0, len(runs) - 1, min(8, len(runs))).astype(int))
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=ticks)
        colorbar.set_ticklabels([str(runs[index]["samples"]) for index in ticks])
        colorbar.set_label("Samples per element")
        if has_zero:
            ax.text(0.02, 0.02, "Zero values omitted on log scale", transform=ax.transAxes, fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return paths
