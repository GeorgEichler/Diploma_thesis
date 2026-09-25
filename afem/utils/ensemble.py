"""Align Monte Carlo convergence histories without assuming matching meshes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


METRICS = (
    ("relative_h1_semi_error_ref", r"$H^1$ norm", "tab:blue", "-", 0),
    ("relative_l2_error_ref", r"$L^2$ norm", "tab:orange", "--", 0),
    ("estimator", r"Residual $\eta$", "tab:green", "-", 1),
)


def aggregate_histories(histories: list[list[dict]], grid_points: int = 100) -> dict:
    """Interpolate errors linearly in log(DOFs), then average across runs.

    Use only the DOF interval reached by every run; never extrapolate. When a
    run revisits a DOF count, retain its last observation for aggregation.
    Raw plots still include all observations. Zeros remain zero in the data.
    These are descriptive statistics of interpolated curves, not FEM solutions
    on a shared mesh and not confidence intervals.
    """
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
        aligned = np.array([
            np.interp(np.log(grid), np.log([h["ndofs"] for h in curve]), [h[key] for h in curve])
            for curve in curves
        ])
        result["metrics"][key] = {
            "mean": np.mean(aligned, axis=0).tolist(),
            "median": np.median(aligned, axis=0).tolist(),
            "q25": np.quantile(aligned, 0.25, axis=0).tolist(),
            "q75": np.quantile(aligned, 0.75, axis=0).tolist(),
        }
    return result


def plot_ensemble(histories, statistics, output_dir, *, summary="median",
                  show_individual=True, show_band=True):
    """Save separate error and residual-estimator PNGs and return their paths."""
    if summary not in ("mean", "median", "none"):
        raise ValueError("summary must be 'mean', 'median', or 'none'")
    if summary == "none" and not show_individual and not show_band:
        raise ValueError("Enable individual curves, a summary, or the spread band")
    figures, axes = zip(*(plt.subplots(figsize=(7, 5)) for _ in range(2)))
    grid = np.asarray(statistics["ndofs"])

    def positive(values):
        values = np.asarray(values, dtype=float)
        return np.where(values > 0, values, np.nan)

    for key, label, color, style, panel in METRICS:
        ax = axes[panel]
        if show_individual:
            for history in histories:
                ax.plot([h["ndofs"] for h in history], positive([h[key] for h in history]),
                        color=color, linestyle=style, alpha=0.18, linewidth=1)
        stats = statistics["metrics"][key]
        if show_band:
            ax.fill_between(grid, positive(stats["q25"]), positive(stats["q75"]),
                            color=color, alpha=0.16)
        if summary != "none":
            ax.plot(grid, positive(stats[summary]), color=color, linestyle=style,
                    linewidth=2, marker="o" if len(grid) == 1 else None)

    for panel, ax in enumerate(axes):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("ndofs")
        ax.set_ylabel("relative error" if panel == 0 else r"Residual $\eta$")
        handles = [Line2D([], [], color=color, linestyle=style,
                          label=label + (f" ({summary})" if summary != "none" else ""))
                   for _, label, color, style, p in METRICS if p == panel]
        if show_individual:
            handles.append(Line2D([], [], color="gray", alpha=0.3, label="Individual runs"))
        if show_band:
            handles.append(Patch(color="gray", alpha=0.16, label="25–75% of runs"))
        ax.legend(handles=handles)
        ax.grid(True, which="both", ls=":")
        if any(h[key] == 0 for history in histories for h in history
               for key, _, _, _, p in METRICS if p == panel):
            ax.text(0.02, 0.02, "Zero values omitted on log scale", transform=ax.transAxes, fontsize=8)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = (output_dir / "ensemble_errors.png", output_dir / "ensemble_estimator.png")
    for fig, path in zip(figures, paths):
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return paths
