from __future__ import annotations
import numpy as np
from skfem import Basis, asm, condense, solve
from skfem.element import ElementTriP1, ElementTetP1
from .forms import laplace
from .load import assemble_load_quadrature, assemble_load_p0, p0_rhs_by_monte_carlo
from afem.core.quadrature import get_quadrature

def make_basis(mesh, quadrature_rule = "default", quadrature_order = None):
    # Possibility to set quadrature order and rule
    dim = mesh.p.shape[0]
    quadrature = get_quadrature(dim, quadrature_rule)
    if dim == 2:
        if quadrature is not None:
            return Basis(mesh, ElementTriP1(), quadrature=quadrature)
        return Basis(mesh, ElementTriP1(), intorder=quadrature_order)
    
    if dim == 3:
        if quadrature is not None:
            return Basis(mesh, ElementTetP1(), quadrature=quadrature)
        return Basis(mesh, ElementTetP1(), intorder=quadrature_order)
    raise ValueError("Only dim=2 or dim=3 supported.")


def solve_poisson(mesh, rhs, load_method: str, mc_samples: int, mc_seed: int,
                  quadrature_rule, quadrature_order):
    basis = make_basis(
        mesh,
        quadrature_rule=quadrature_rule,
        quadrature_order=quadrature_order)
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
