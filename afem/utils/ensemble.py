"""Align Monte Carlo convergence histories without assuming matching meshes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from afem.utils.comparison_layout import comparison_figure
from afem.utils.figure_export import save_figure
from afem.utils.publication_style import paper_style

METRICS = (
    ("relative_h1_semi_error_ref", r"$H^1$", "#0072B2", "-", 0),
    ("relative_l2_error_ref", r"$L^2$", "#D55E00", "--", 0),
    ("estimator", r"Residual $\eta$", "#009E73", "-", 1),
)


def _aligned_values(histories, key, grid):
    """Use the last observation at each DOF count, with no extrapolation."""
    aligned = []
    for history in histories:
        by_dofs = {entry["ndofs"]: entry[key] for entry in history}
        dofs = sorted(by_dofs)
        aligned.append(np.interp(np.log(grid), np.log(dofs), [by_dofs[n] for n in dofs]))
    return np.asarray(aligned)


def aggregate_histories(histories: list[list[dict]], grid_points: int = 100) -> dict:
    """Interpolate errors linearly in log(DOFs), then average across runs.    """
    if not histories or grid_points < 2:
        raise ValueError("Provide at least one history and grid_points >= 2")
    curves = []
    for history in histories:
        if not history:
            raise ValueError("Histories must not be empty")
        by_dofs = {}
        for entry in history:
            n = entry["ndofs"]
            if not np.isfinite(n) or n <= 0:
                raise ValueError("DOF counts must be finite and positive")
            for key, *_ in METRICS:
                value = entry[key]
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"{key} must be finite and nonnegative")
            by_dofs[n] = entry
        curves.append([by_dofs[n] for n in sorted(by_dofs)])
    lower = max(curve[0]["ndofs"] for curve in curves)
    upper = min(curve[-1]["ndofs"] for curve in curves)
    if lower > upper:
        raise ValueError("The runs have no common DOF range")
    grid = np.array([float(lower)]) if lower == upper else np.geomspace(lower, upper, grid_points)
    result = {
        "nruns": len(histories),
        "ndofs": grid.tolist(),
        "alignment": "linear values against log(DOFs), common range only; last repeated DOF observation",
        "band": "pointwise 25th-75th percentiles across runs; not a confidence interval",
        "metrics": {},
    }
    for key, *_ in METRICS:
        aligned = _aligned_values(curves, key, grid)
        result["metrics"][key] = {
            "mean": np.mean(aligned, axis=0).tolist(),
            "median": np.median(aligned, axis=0).tolist(),
            "q25": np.quantile(aligned, 0.25, axis=0).tolist(),
            "q75": np.quantile(aligned, 0.75, axis=0).tolist(),
        }
    return result


@paper_style
def plot_ensemble_error_deviations(histories, statistics, output_dir, *,
                                   save_pdf=True, save_png=False):
    """Plot 100 * (run error - ensemble mean) / mean on the common DOF grid.

    Both error norms use the same alignment as the ensemble statistics.
    Zero-mean locations have undefined percentages and are omitted.
    """
    fig, ax, _ = comparison_figure()
    grid = np.asarray(statistics["ndofs"])
    zero_mean = False
    handles = []
    ax.axhline(0, color="gray", linewidth=0.6)
    for key, label, color, style, panel in METRICS:
        if panel != 0:
            continue
        stats = statistics["metrics"][key]
        mean = np.asarray(stats["mean"])
        valid = mean > 0
        zero_mean |= bool(np.any(~valid))

        def percentage(values):
            values = np.asarray(values)
            return np.divide(100 * (values - mean), mean,
                             out=np.full(values.shape, np.nan), where=valid)

        for values in _aligned_values(histories, key, grid):
            ax.plot(grid, percentage(values), color=color, linestyle=style,
                    alpha=0.4, linewidth=0.55, marker="o" if len(grid) == 1 else None,
                    markersize=2)
        ax.fill_between(grid, percentage(stats["q25"]), percentage(stats["q75"]),
                        color=color, alpha=0.16, linewidth=0)
        handles.append(Line2D([], [], color=color, linestyle=style, label=label))
    ax.set_xscale("log")
    ax.set_xlabel(r"$\mathrm{ndof}$")
    ax.set_ylabel("deviation from mean (%)")
    ax.legend(handles=handles)
    fig.legend(handles=[
        Line2D([], [], color="gray", alpha=0.5, linewidth=0.55, label="Individual runs"),
        Patch(color="gray", alpha=0.16, label="25-75% of runs"),
    ], loc="lower left", bbox_to_anchor=(0.22, 0.06), frameon=False, borderaxespad=0)
    if zero_mean:
        ax.text(0.02, 0.02, "Zero-mean points omitted", transform=ax.transAxes, fontsize=6)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = save_figure(fig, output_dir / "ensemble_error_deviations.pdf",
                       save_pdf=save_pdf, save_png=save_png)
    plt.close(fig)
    return path


@paper_style
def plot_ensemble(histories, statistics, output_dir, *, summary="median",
                  show_individual=True, show_band=True, save_pdf=True, save_png=False):
    """Export errors, estimator, and mean-relative error deviations.

    Deviation plots always show individual runs and their interquartile band;
    summary and visibility options apply to the original convergence plots.
    Return PDF paths, or PNG paths for PNG-only output.
    """
    if summary not in ("mean", "median", "none"):
        raise ValueError("summary must be 'mean', 'median', or 'none'")
    if summary == "none" and not show_individual and not show_band:
        raise ValueError("Enable individual curves, a summary, or the spread band")
    figures, axes, _ = zip(*(comparison_figure() for _ in range(2)))
    grid = np.asarray(statistics["ndofs"])

    def positive(values):
        values = np.asarray(values, dtype=float)
        return np.where(values > 0, values, np.nan)

    for key, label, color, style, panel in METRICS:
        ax = axes[panel]
        if show_individual:
            for history in histories:
                ax.plot([h["ndofs"] for h in history], positive([h[key] for h in history]),
                        color=color, linestyle=style, alpha=0.12, linewidth=0.45)
        stats = statistics["metrics"][key]
        if show_band:
            ax.fill_between(grid, positive(stats["q25"]), positive(stats["q75"]),
                            color=color, alpha=0.16, linewidth=0)
        if summary != "none":
            ax.plot(grid, positive(stats[summary]), color=color, linestyle=style,
                    linewidth=1, marker="o" if len(grid) == 1 else None)

    for panel, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\mathrm{ndof}$")
        ax.set_ylabel("relative error" if panel == 0 else r"Residual $\eta$")
        handles = [Line2D([], [], color=color, linestyle=style,
                          label=label + (f" ({summary})" if summary != "none" else ""))
                   for _, label, color, style, p in METRICS if p == panel]
        ax.legend(handles=handles)
        spread_handles = []
        if show_individual:
            spread_handles.append(Line2D([], [], color="gray", alpha=0.5,
                                         linewidth=0.45, label="Individual runs"))
        if show_band:
            spread_handles.append(Patch(color="gray", alpha=0.16, label="25-75% of runs"))
        if spread_handles:
            figures[panel].legend(handles=spread_handles, loc="lower left",
                                  bbox_to_anchor=(0.22, 0.06), frameon=False,
                                  borderaxespad=0)
        if any(h[key] == 0 for history in histories for h in history
               for key, _, _, _, p in METRICS if p == panel):
            ax.text(0.02, 0.02, "Zero values omitted on log scale", transform=ax.transAxes, fontsize=6)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (output_dir / "ensemble_errors.pdf", output_dir / "ensemble_estimator.pdf")
    saved_paths = []
    for fig, path in zip(figures, paths):
        saved_path = save_figure(fig, path, save_pdf=save_pdf, save_png=save_png)
        if saved_path is not None:
            saved_paths.append(saved_path)
        plt.close(fig)
    deviation_path = plot_ensemble_error_deviations(
        histories, statistics, output_dir, save_pdf=save_pdf, save_png=save_png)
    if deviation_path is not None:
        saved_paths.append(deviation_path)
    return tuple(saved_paths)
