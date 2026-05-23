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
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel("estimator")
    ax.grid(True, which="both", ls=":")
    ax.set_title("Estimator decay")
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)
