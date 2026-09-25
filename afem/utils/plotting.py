from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_mesh(mesh, filename: str | Path):
    filename = Path(filename)
    fig = plt.figure(figsize=(6, 5))
    if mesh.p.shape[0] == 2:
        ax = fig.add_subplot(111)
        ax.triplot(mesh.p[0], mesh.p[1], mesh.t.T, linewidth=0.4)
        ax.set_aspect("equal")
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
    ax.set_aspect("equal")
    ax.set_title("P1 solution")
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


def plot_reference_error_comparison(runs: list[dict], filename: str | Path):
    """Compare relative H1 seminorm (solid) and L2 (dashed) errors.

    Each run supplies a history, a mathtext label, and a color. Histories may
    have different mesh sizes because each method refines independently.
    """
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    method_handles = []
    for run in runs:
        history = run["history"]
        ndofs = np.array([entry["ndofs"] for entry in history])
        h1_semi = np.array([entry["relative_h1_semi_error_ref"] for entry in history])
        l2 = np.array([entry["relative_l2_error_ref"] for entry in history])
        ax.loglog(ndofs, h1_semi, color=run["color"], linestyle="-", marker="o")
        ax.loglog(ndofs, l2, color=run["color"], linestyle="--", marker="o")
        method_handles.append(Line2D([], [], color=run["color"], label=run["label"]))

    norm_handles = [
        Line2D([], [], color="black", linestyle="-", label=r"$H^1$ norm"),
        Line2D([], [], color="black", linestyle="--", label=r"$L^2$ norm"),
    ]
    ax.legend(handles=method_handles + norm_handles)
    ax.set_xlabel("ndofs")
    ax.set_ylabel("error")
    ax.set_title("Load approximation comparison")
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
