"""Plot Monte Carlo sample-count studies with a compact, ordered color key."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

from afem.utils.comparison_layout import comparison_figure
from afem.utils.figure_export import save_figure
from afem.utils.publication_style import paper_style


@paper_style
def plot_sample_comparison(runs: list[dict], output_dir: Path, *,
                           save_pdf: bool = True, save_png: bool = False):
    """Plot each history at its own DOFs; darker crest shades mean more samples.

    Colors use a continuous logarithmic scale in sample count, not an
    interpolation of the solution. Errors are relative H1 seminorm and L2 errors.
    """
    if not runs:
        raise ValueError("At least one completed run is needed")
    runs = sorted(runs, key=lambda run: run["samples"])
    counts = np.array([run["samples"] for run in runs], dtype=float)
    if np.any(~np.isfinite(counts)) or np.any(counts <= 0):
        raise ValueError("Sample counts must be finite and positive for a logarithmic color scale")
    crest = sns.color_palette("crest", as_cmap=True)
    # Avoid the palest shades, which lose contrast in small printed panels.
    cmap = LinearSegmentedColormap.from_list("crest_print", crest(np.linspace(0.15, 1, 256)))
    vmin, vmax = counts[0], counts[-1]
    if vmin == vmax:
        # Center a lone completed count in a nondegenerate logarithmic range.
        vmin, vmax = vmin / np.sqrt(10), vmax * np.sqrt(10)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (output_dir / "sample_errors.pdf", output_dir / "sample_estimator.pdf")
    plots = (
        (("relative_h1_semi_error_ref", "-"), ("relative_l2_error_ref", "--")),
        (("estimator", "-"),),
    )

    saved_paths = []
    for panel, (metrics, path) in enumerate(zip(plots, paths)):
        fig, ax, colorbar_ax = comparison_figure(with_colorbar=True)
        has_zero = False
        for run in runs:
            ndofs = np.array([entry["ndofs"] for entry in run["history"]])
            for key, style in metrics:
                values = np.array([entry[key] for entry in run["history"]])
                has_zero = has_zero or bool(np.any(values == 0))
                ax.plot(ndofs, np.where(values > 0, values, np.nan),
                        color=cmap(norm(run["samples"])), linestyle=style, marker="o")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\mathrm{ndof}$")
        ax.set_ylabel("relative error" if panel == 0 else r"Residual $\eta$")
        if panel == 0:
            ax.legend(handles=[
                Line2D([], [], color="black", linestyle="-", label=r"$H^1$ seminorm"),
                Line2D([], [], color="black", linestyle="--", label=r"$L^2$ norm"),
            ])
        # A numeric color key replaces a separate legend entry for every curve.
        tick_indices = np.unique(np.linspace(0, len(runs) - 1, min(8, len(runs))).astype(int))
        colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colorbar_ax,
                               ticks=counts[tick_indices], orientation="horizontal")
        colorbar.set_ticklabels([str(runs[index]["samples"]) for index in tick_indices])
        colorbar.minorticks_off()
        colorbar.set_label("Samples per element", fontsize=8, labelpad=3)
        colorbar.ax.xaxis.set_label_position("top")
        colorbar.ax.tick_params(labelsize=7.5, pad=2, top=False, bottom=True)
        # Matplotlib may rasterize colorbar gradients by default; keep this key vector.
        if colorbar.solids is not None:
            colorbar.solids.set_rasterized(False)
        if has_zero:
            ax.text(0.02, 0.02, "Zero values omitted on log scale", transform=ax.transAxes, fontsize=6)
        saved_path = save_figure(fig, path, save_pdf=save_pdf, save_png=save_png)
        if saved_path is not None:
            saved_paths.append(saved_path)
        plt.close(fig)
    return tuple(saved_paths)
