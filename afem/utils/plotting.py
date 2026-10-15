from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from afem.utils.comparison_layout import mesh_figure
from afem.utils.figure_export import save_figure
from afem.utils.publication_style import MESH_LINEWIDTH, PANEL_FIGSIZE, paper_style


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


@paper_style
def plot_mesh(mesh, filename: str | Path, *, save_pdf: bool = True, save_png: bool = False):
    """Export a mesh panel sized for four columns, without axes or tick labels."""
    filename = Path(filename)
    fig, ax = mesh_figure(mesh.p.shape[0])
    if mesh.p.shape[0] == 2:
        ax.triplot(mesh.p[0], mesh.p[1], mesh.t.T, linewidth=MESH_LINEWIDTH)
        ax.set_xlim(mesh.p[0].min(), mesh.p[0].max())
        ax.set_ylim(mesh.p[1].min(), mesh.p[1].max())
    else:
        ax.scatter(mesh.p[0], mesh.p[1], mesh.p[2], s=2)
    path = save_figure(fig, filename, save_pdf=save_pdf, save_png=save_png)
    plt.close(fig)
    return path


@paper_style
def plot_solution_2d(mesh, u: np.ndarray, filename: str | Path, *,
                     save_pdf: bool = True, save_png: bool = False):
    filename = Path(filename)
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    pc = ax.tripcolor(mesh.p[0], mesh.p[1], mesh.t.T, u, shading="gouraud")
    ax.triplot(mesh.p[0], mesh.p[1], mesh.t.T, linewidth=0.2, alpha=0.4)
    _format_2d_domain_axes(ax, mesh)
    colorbar = fig.colorbar(pc, ax=ax)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
    fig.tight_layout()
    path = save_figure(fig, filename, save_pdf=save_pdf, save_png=save_png)
    plt.close(fig)
    return path


@paper_style
def plot_solution_surface_2d(mesh, u: np.ndarray, filename: str | Path, *,
                             save_pdf: bool = True, save_png: bool = False,
                             elev: float = 30, azim: float = 45):
    """Plot a P1 solution as a vector surface using the actual domain triangles."""
    u = np.asarray(u)
    if mesh.p.shape[0] != 2 or u.shape != (mesh.p.shape[1],):
        raise ValueError("A surface plot requires a 2D mesh and one value per vertex.")
    fig = plt.figure(figsize=PANEL_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    # Explicit connectivity preserves holes and non-convex domain boundaries.
    ax.plot_trisurf(mesh.p[0], mesh.p[1], u, triangles=mesh.t.T,
                    cmap="viridis", linewidth=0, edgecolor="none",
                    antialiased=False, rasterized=False)
    ax.set_xlabel(r"$x$", labelpad=0)
    ax.set_ylabel(r"$y$", labelpad=0)
    ax.set_zlabel(r"$u_h$", labelpad=0)
    ax.set_xlim(mesh.p[0].min(), mesh.p[0].max())
    ax.set_ylim(mesh.p[1].min(), mesh.p[1].max())
    ax.set_xticks(np.linspace(mesh.p[0].min(), mesh.p[0].max(), 3))
    ax.set_yticks(np.linspace(mesh.p[1].min(), mesh.p[1].max(), 3))
    ax.tick_params(axis="both", labelsize=8, pad=0)
    ax.set_box_aspect((1, 1, 0.65), zoom=0.82)
    ax.set_facecolor("white")
    ax.view_init(elev=elev, azim=azim)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.10, top=0.98)
    try:
        return save_figure(fig, filename, save_pdf=save_pdf, save_png=save_png)
    finally:
        plt.close(fig)


@paper_style
def plot_history(history: list[dict], filename: str | Path, *,
                 save_pdf: bool = True, save_png: bool = False):
    filename = Path(filename)
    ndofs = np.array([h["ndofs"] for h in history])
    eta = np.array([h["estimator"] for h in history])
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    ax.loglog(ndofs, eta, marker="o")
    ax.set_xlabel(r"$\mathrm{ndof}$")
    ax.set_ylabel(r"Residual $\eta$")
    ax.set_title(r"Residual estimator $\eta$")
    fig.tight_layout()
    path = save_figure(fig, filename, save_pdf=save_pdf, save_png=save_png)
    plt.close(fig)
    return path

@paper_style
def plot_reference_error_history(history, filename: str | Path, *,
                                 save_pdf: bool = True, save_png: bool = False):
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

    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
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
    ax.set_xlabel(r"$\mathrm{ndof}$")
    fig.tight_layout()
    path = save_figure(fig, filename, save_pdf=save_pdf, save_png=save_png)
    plt.close(fig)
    return path


def _plot_reference_guides(ax, all_ndofs, all_values, reference_rates, reference_starts):
    """Draw anchored power-law guides shared by error and estimator plots."""
    reference_handles = []
    if reference_rates:
        x_min = min(values.min() for values in all_ndofs)
        x_max = max(values.max() for values in all_ndofs)
        y_anchor = max(values.max() for values in all_values)
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

    return reference_handles


@paper_style
def plot_estimator_comparison(
    runs: list[dict], filename: str | Path, *, title: str = "",
    reference_rates: tuple[float, ...] = (),
    reference_starts: tuple[tuple[float, float], ...] | None = None,
    save_pdf: bool = True, save_png: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """Compare saved residual estimators on independently refined meshes.

    Runs use the same history, label, color, and optional marker fields as
    plot_reference_error_comparison. Estimator values are plotted as recorded,
    without normalization or requiring reference-solution errors. Positive
    reference_rates add ndof**(-rate) guides anchored at reference_starts
    (ndof, estimator); None anchors at the smallest ndof and largest estimator.
    """
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    all_ndofs = []
    all_estimators = []
    for run in runs:
        history = run["history"]
        ndofs = np.array([entry["ndofs"] for entry in history])
        estimators = np.array([entry["estimator"] for entry in history])
        all_ndofs.append(ndofs)
        all_estimators.append(estimators)
        ax.loglog(
            ndofs, estimators,
            color=run["color"], marker=run.get("marker", "o"),
            label=run["label"],
        )
    _plot_reference_guides(
        ax, all_ndofs, all_estimators, reference_rates, reference_starts,
    )
    ax.set_xlabel(r"$\mathrm{ndof}$")
    ax.set_ylabel(r"Residual $\eta$")
    ax.set_title(title)
    ax.legend()
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    try:
        return save_figure(fig, filename, save_pdf=save_pdf, save_png=save_png)
    finally:
        plt.close(fig)


@paper_style
def plot_reference_error_comparison(
    runs: list[dict], filename: str | Path, *, title: str = "Load approximation comparison",
    reference_rates: tuple[float, ...] = (),
    reference_starts: tuple[tuple[float, float], ...] | None = None,
    save_pdf: bool = True, save_png: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """Compare relative H1 seminorm (solid) and L2 (dashed) errors.

    Each run supplies a history, a mathtext label, and a color, and may supply
    a marker. Histories may have different mesh sizes because each method
    refines independently. Positive ``reference_rates`` add optional
    ``ndof**(-rate)`` guide lines. ``reference_starts`` may provide one
    positive ``(ndof, error)`` start point per guide; otherwise all guides
    start at the smallest ndof and largest error in the plotted data.
    Set ``xlim`` and ``ylim`` to share axis limits between separate figures;
    None keeps automatic scaling.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
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

    reference_handles = _plot_reference_guides(
        ax, all_ndofs, all_errors, reference_rates, reference_starts,
    )

    ax.legend(handles=method_handles + reference_handles)
    ax.set_xlabel(r"$\mathrm{ndof}$")
    ax.set_ylabel("relative error")
    ax.set_title(title)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    path = save_figure(fig, filename, save_pdf=save_pdf, save_png=save_png)
    plt.close(fig)
    return path
