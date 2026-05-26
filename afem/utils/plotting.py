from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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
            "relative_h1_error_ref" in h
            or "relative_h1_semi_error_ref" in h
            or "h1_semi_error_ref" in h
        )
    ]
    if not entries:
        return

    ndofs = np.array([h["ndofs"] for h in entries])

    fig, ax = plt.subplots()
    if "relative_h1_error_ref" in entries[0]:
        h1_error = np.array([h["relative_h1_error_ref"] for h in entries])
        ax.loglog(ndofs, h1_error, marker="o", label=r"relative $H^1$")
        if "relative_l2_error_ref" in entries[0]:
            l2_error = np.array([h["relative_l2_error_ref"] for h in entries])
            ax.loglog(ndofs, l2_error, marker="s", label=r"relative $L^2$")
        ax.set_ylabel("relative error")
        ax.set_title("Relative reference error")
        ax.legend()
    elif "relative_h1_semi_error_ref" in entries[0]:
        h1_error = np.array([h["relative_h1_semi_error_ref"] for h in entries])
        ax.loglog(ndofs, h1_error, marker="o", label=r"relative $H^1$ seminorm")
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
