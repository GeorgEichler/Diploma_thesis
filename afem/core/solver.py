from __future__ import annotations
import numpy as np
from skfem import Basis, asm, condense, solve
from skfem.element import ElementTriP1, ElementTriP2, ElementTriP3, ElementTetP1, ElementTetP2
from .forms import laplace
from .load import assemble_load_quadrature, assemble_load_p0, p0_rhs_by_monte_carlo
from afem.core.quadrature import get_quadrature

def make_element(dim: int, order: int):
    if dim == 2:
        if order == 1:
            return ElementTriP1()
        if order == 2:
            return ElementTriP2()
        if order == 3:
            return ElementTriP3()
    
    if dim == 3:
        if order == 1:
            return ElementTetP1()
        if order == 2:
            return ElementTetP2()
    
    raise ValueError(f"Unsupported element order P{order} in dim={dim}")

def make_basis(mesh, element_order: int = 1,
               quadrature_rule = "default", quadrature_order = None):
    # Possibility to set quadrature order and rule
    dim = mesh.p.shape[0]
    element = make_element(dim, element_order)

    if quadrature_rule == "default":
        return Basis(mesh, element, intorder=quadrature_order)
    
    # Use own defined quadrature rule
    quadrature = get_quadrature(dim, quadrature_rule)
    return Basis(mesh, element, quadrature=quadrature)


def solve_poisson(mesh, rhs, load_method: str, mc_samples: int, mc_seed: int,
                  quadrature_rule, quadrature_order, element_order: int = 1):
    basis = make_basis(
        mesh,
        element_order=element_order,
        quadrature_rule=quadrature_rule,
        quadrature_order=quadrature_order)
    A = asm(laplace, basis)

    # fbar is constant across each element for Monte-Carlo method
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