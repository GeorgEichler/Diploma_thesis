from __future__ import annotations

import numpy as np
from skfem import Basis, asm

from .forms import laplace
from .load import assemble_load_quadrature
from .solver import solve_poisson


def default_energy_quadrature_order(reference_order: int) -> int:
    return 2 * reference_order + 4


def _basis_with_quadrature_order(basis, quadrature_order: int | None):
    if quadrature_order is None:
        return basis
    return Basis(basis.mesh, basis.elem, intorder=quadrature_order)


def dirichlet_energy(
    basis,
    u: np.ndarray,
    rhs,
    quadrature_order: int | None = None,
) -> float:
    """Compute J(u) = 1/2 a(u, u) - l(u).

    If quadrature_order is given, the energy is evaluated using that quadrature
    order, independent of the quadrature rule used to compute u.
    """
    energy_basis = _basis_with_quadrature_order(basis, quadrature_order)
    A = asm(laplace, energy_basis)
    b = assemble_load_quadrature(energy_basis, rhs)
    return float(0.5 * u @ (A @ u) - b @ u)


def reference_solution_energy(
    mesh,
    rhs,
    reference_order: int = 3,
    quadrature_order: int | None = None,
) -> dict:
    """Solve one enriched reference problem and return its Dirichlet energy."""
    if quadrature_order is None:
        quadrature_order = default_energy_quadrature_order(reference_order)

    u_ref, basis_ref, _ = solve_poisson(
        mesh,
        rhs,
        load_method="quadrature",
        mc_samples=0,
        mc_seed=0,
        quadrature_rule="default",
        quadrature_order=quadrature_order,
        element_order=reference_order,
    )

    return {
        "reference_order": int(reference_order),
        "reference_ndofs": int(basis_ref.N),
        "reference_quadrature_order": int(quadrature_order),
        "reference_energy": dirichlet_energy(
            basis_ref,
            u_ref,
            rhs,
            quadrature_order=quadrature_order,
        ),
    }


def add_energy_error_to_history(history: list[dict], reference_energy: float) -> None:
    """Append reference energy error data to history entries in-place."""
    for entry in history:
        energy = entry.get("energy")
        if energy is None:
            continue

        # With J(v) = 1/2 a(v, v) - l(v):
        # ||grad(u - v)||^2 = 2 * (J(v) - J(u)).
        error_sq = max(0.0, 2.0 * (float(energy) - reference_energy))
        entry["h1_semi_error_sq_ref"] = error_sq
        entry["h1_semi_error_ref"] = float(np.sqrt(error_sq))
