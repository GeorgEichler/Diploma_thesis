from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _format_2d_domain_axes(ax, mesh):
    """Align a two-dimensional domain with five evenly spaced axis ticks."""
    x_min, x_max = mesh.p[0].min(), mesh.p[0].max()
    y_min, y_max = mesh.p[1].min(), mesh.p[1].max()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(np.linspace(x_min, x_max, 5))
    ax.set_yticks(np.linspace(y_min, y_max, 5))
    ax.xaxis.set_major_formatter("{x:.1f}")
    ax.yaxis.set_major_formatter("{x:.1f}")
    ax.set_aspect("equal")


def plot_mesh(mesh, filename: str | Path):
    filename = Path(filename)
    fig = plt.figure(figsize=(6, 5))
    if mesh.p.shape[0] == 2:
        ax = fig.add_subplot(111)
        ax.triplot(mesh.p[0], mesh.p[1], mesh.t.T, linewidth=0.4)
        _format_2d_domain_axes(ax, mesh)
        ax.set_title(f"Mesh: {mesh.t.shape[1]} elements")
    else:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(mesh.p[0], mesh.p[1], mesh.p[2], s=2)
        ax.set_title(f"3D mesh vertices: {mesh.p.shape[1]}")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_solution_2d(mesh, u: np.ndarray, filename: str | Path):
    filename = Path(filename)
    fig, ax = plt.subplots(figsize=(6, 5))
    pc = ax.tripcolor(mesh.p[0], mesh.p[1], mesh.t.T, u, shading="gouraud")
    ax.triplot(mesh.p[0], mesh.p[1], mesh.t.T, linewidth=0.2, alpha=0.4)
    _format_2d_domain_axes(ax, mesh)
    fig.colorbar(pc, ax=ax)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_history(history: list[dict], filename: str | Path):
    filename = Path(filename)
    ndofs = np.array([h["ndofs"] for h in history])
    eta = np.array([h["estimator"] for h in history])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(ndofs, eta, marker="o")
    ax.set_xlabel("ndofs")
    ax.set_ylabel(r"Residual $\eta$")
    ax.grid(True, which="both", ls=":")
    ax.set_title(r"Residual estimator $\eta$")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

def plot_reference_error_history(history, filename: str | Path):
    entries = [
        h for h in history
        if (
            "relative_h1_semi_error_ref" in h
            or "h1_semi_error_ref" in h
        )
    ]
    if not entries:
        return

    ndofs = np.array([h["ndofs"] for h in entries])

    fig, ax = plt.subplots()
    if "relative_h1_semi_error_ref" in entries[0]:
        h1_semi_error = np.array([h["relative_h1_semi_error_ref"] for h in entries])
        ax.loglog(ndofs, h1_semi_error, marker="o", label=r"relative $H^1$ seminorm")
        if "relative_l2_error_ref" in entries[0]:
            l2_error = np.array([h["relative_l2_error_ref"] for h in entries])
            ax.loglog(ndofs, l2_error, marker="s", label=r"relative $L^2$")
        ax.set_ylabel("relative error")
        ax.set_title("Relative reference error")
        ax.legend()
    else:
        error = np.array([h["h1_semi_error_ref"] for h in entries])
        ax.loglog(ndofs, error, marker="o")
        ax.set_ylabel(r"$\|\nabla(u_{\mathrm{ref}} - u_h)\|_{L^2}$")
        ax.set_title(r"Reference energy error")
    ax.set_xlabel("ndofs")
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def plot_reference_error_comparison(
    runs: list[dict], filename: str | Path, *, title: str = "Load approximation comparison",
    reference_rates: tuple[float, ...] = (),
    reference_starts: tuple[tuple[float, float], ...] | None = None,
):
    """Compare relative H1 seminorm (solid) and L2 (dashed) errors.

    Each run supplies a history, a mathtext label, and a color, and may supply
    a marker. Histories may have different mesh sizes because each method
    refines independently. Positive ``reference_rates`` add optional
    ``ndof**(-rate)`` guide lines. ``reference_starts`` may provide one
    positive ``(ndof, error)`` start point per guide; otherwise all guides
    start at the smallest ndof and largest error in the plotted data.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    method_handles = []
    all_ndofs = []
    all_errors = []
    for run in runs:
        history = run["history"]
        marker = run.get("marker", "o")
        ndofs = np.array([entry["ndofs"] for entry in history])
        h1_semi = np.array([entry["relative_h1_semi_error_ref"] for entry in history])
        l2 = np.array([entry["relative_l2_error_ref"] for entry in history])
        all_ndofs.append(ndofs)
        all_errors.extend((h1_semi, l2))
        ax.loglog(ndofs, h1_semi, color=run["color"], linestyle="-", marker=marker)
        ax.loglog(ndofs, l2, color=run["color"], linestyle="--", marker=marker)
        method_handles.append(Line2D(
            [], [], color=run["color"], marker=marker, label=run["label"],
        ))

    reference_handles = []
    if reference_rates:
        x_min = min(values.min() for values in all_ndofs)
        x_max = max(values.max() for values in all_ndofs)
        y_anchor = max(values.max() for values in all_errors)
        if reference_starts is None:
            reference_starts = tuple((x_min, y_anchor) for _ in reference_rates)
        elif len(reference_starts) != len(reference_rates):
            raise ValueError("Provide one reference start for each reference rate")
        linestyles = (
            (0, (6, 2)),
            (0, (2, 2)),
            (0, (6, 2, 1, 2)),
        )
        for index, (rate, start) in enumerate(zip(reference_rates, reference_starts)):
            if rate <= 0:
                raise ValueError("Reference rates must be positive")
            x_start, y_start = start
            if x_start <= 0 or y_start <= 0:
                raise ValueError("Reference starts must contain positive values")
            if x_start >= x_max:
                raise ValueError("Reference lines must start below the largest ndof")
            exponent = Fraction(rate).limit_denominator()
            label = rf"$\mathrm{{ndof}}^{{-{exponent.numerator}/{exponent.denominator}}}$"
            reference_ndofs = np.geomspace(x_start, x_max, 100)
            reference_error = y_start * (reference_ndofs / x_start) ** -rate
            handle, = ax.loglog(
                reference_ndofs, reference_error, color="black",
                linestyle=linestyles[index % len(linestyles)], linewidth=1.2,
                label=label,
            )
            reference_handles.append(handle)

    ax.legend(handles=method_handles + reference_handles)
    ax.set_xlabel("ndofs")
    ax.set_ylabel("error")
    ax.set_title(title)
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
