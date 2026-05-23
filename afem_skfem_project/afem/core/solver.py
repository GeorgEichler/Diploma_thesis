from __future__ import annotations
import numpy as np
from skfem import Basis, asm, condense, solve
from skfem.element import ElementTriP1, ElementTetP1
from .forms import laplace
from .load import assemble_load_quadrature, assemble_load_p0, p0_rhs_by_monte_carlo


def make_basis(mesh):
    dim = mesh.p.shape[0]
    if dim == 2:
        return Basis(mesh, ElementTriP1())
    if dim == 3:
        return Basis(mesh, ElementTetP1())
    raise ValueError("Only dim=2 or dim=3 supported.")


def solve_poisson(mesh, rhs, load_method: str, mc_samples: int, mc_seed: int):
    basis = make_basis(mesh)
    A = asm(laplace, basis)

    fbar = None
    if load_method == "quadrature":
        b = assemble_load_quadrature(basis, rhs)
    elif load_method == "monte_carlo":
        fbar = p0_rhs_by_monte_carlo(mesh, rhs, mc_samples, mc_seed)
        b = assemble_load_p0(basis, fbar)
    else:
        raise ValueError(f"Unknown load_method: {load_method}")

    D = basis.get_dofs().all()
    x = solve(*condense(A, b, D=D))
    return np.asarray(x), basis, fbar
