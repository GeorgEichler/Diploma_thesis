from __future__ import annotations
import numpy as np
from skfem import asm
from .forms import quadrature_load_form


def assemble_load_quadrature(basis, rhs):
    return asm(quadrature_load_form(rhs), basis)


def simplex_volumes(mesh) -> np.ndarray:
    p, t = mesh.p, mesh.t
    dim = p.shape[0]
    ne = t.shape[1]
    vols = np.empty(ne)
    fact = 2.0 if dim == 2 else 6.0
    for k in range(ne):
        verts = p[:, t[:, k]]
        B = verts[:, 1:] - verts[:, [0]]
        vols[k] = abs(np.linalg.det(B)) / fact
    return vols


def element_centroids(mesh) -> np.ndarray:
    return mesh.p[:, mesh.t].mean(axis=1)


def sample_points_in_simplices(mesh, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random points in every simplex.

    Returns shape (dim, nelems, n_samples).
    """
    p, t = mesh.p, mesh.t
    dim, ne = p.shape[0], t.shape[1]
    # Exponential normalization gives uniform barycentric coordinates.
    e = rng.exponential(scale=1.0, size=(dim + 1, ne, n_samples))
    lam = e / e.sum(axis=0, keepdims=True)
    pts = np.einsum("vks,dvk->dks", lam, p[:, t])
    return pts


def p0_rhs_by_monte_carlo(mesh, rhs, n_samples: int, seed: int) -> np.ndarray:
    """Approximate element averages of f by Monte Carlo."""
    rng = np.random.default_rng(seed)
    pts = sample_points_in_simplices(mesh, n_samples, rng)
    dim, ne, ns = pts.shape
    flat = pts.reshape(dim, ne * ns)
    values = rhs(flat).reshape(ne, ns)
    return values.mean(axis=1)


def assemble_load_p0(basis, fbar: np.ndarray):
    from skfem import LinearForm

    fbar_q = np.repeat(fbar[:, None], basis.X.shape[-1], axis=1)

    @LinearForm
    def p0_load(v, w):
        return w.fbar * v

    return asm(p0_load, basis, fbar=fbar_q)