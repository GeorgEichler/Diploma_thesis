from __future__ import annotations

import numpy as np
from skfem import Basis, asm

from .forms import laplace
from .load import assemble_load_quadrature
from .solver import make_basis, solve_poisson


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


def _integrate_squared(values: np.ndarray, dx: np.ndarray) -> float:
    return float(np.sum(values * values * dx))


def _relative_error(error: float, reference_norm: float) -> float:
    if reference_norm > np.finfo(float).eps:
        return float(error / reference_norm)
    return 0.0 if error <= np.finfo(float).eps else float("inf")


def _prolong_p1_to_mesh(source_mesh, source_u: np.ndarray, target_mesh) -> np.ndarray:
    """Evaluate a nested P1 function at the vertices of target_mesh."""
    source_basis = make_basis(
        source_mesh,
        element_order=1,
        quadrature_rule="default",
        quadrature_order=None,
    )
    return np.asarray(source_basis.interpolator(source_u)(target_mesh.p))


def reference_solution_direct_errors(
    mesh,
    rhs,
    snapshots: list[dict],
    reference_order: int = 3,
    quadrature_order: int | None = None,
) -> dict:
    """Solve an enriched reference problem and add relative direct errors.

    The AFEM iterates are P1 functions on meshes produced by refinement of the
    final mesh lineage.  They are therefore prolonged exactly by evaluating the
    old P1 function at the final mesh vertices and integrating both functions on
    the final mesh.
    """
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
    basis_p1 = make_basis(
        mesh,
        element_order=1,
        quadrature_rule="default",
        quadrature_order=quadrature_order,
    )

    ref_field = basis_ref.interpolate(u_ref)
    ref_l2_norm = np.sqrt(_integrate_squared(ref_field.value, basis_ref.dx))
    ref_h1_semi_norm = np.sqrt(_integrate_squared(ref_field.grad, basis_ref.dx))
    ref_h1_norm = np.sqrt(ref_l2_norm * ref_l2_norm + ref_h1_semi_norm * ref_h1_semi_norm)

    for snapshot in snapshots:
        u_p1_on_ref_mesh = _prolong_p1_to_mesh(snapshot["mesh"], snapshot["u"], mesh)
        uh_field = basis_p1.interpolate(u_p1_on_ref_mesh)

        l2_error = np.sqrt(_integrate_squared(ref_field.value - uh_field.value, basis_ref.dx))
        h1_semi_error = np.sqrt(_integrate_squared(ref_field.grad - uh_field.grad, basis_ref.dx))
        h1_error = np.sqrt(l2_error * l2_error + h1_semi_error * h1_semi_error)

        entry = snapshot["history_entry"]
        entry["relative_l2_error_ref"] = _relative_error(l2_error, ref_l2_norm)
        entry["relative_h1_error_ref"] = _relative_error(h1_error, ref_h1_norm)
        entry["relative_h1_semi_error_ref"] = _relative_error(
            h1_semi_error,
            ref_h1_semi_norm,
        )

    return {
        "reference_error_method": "direct",
        "reference_order": int(reference_order),
        "reference_ndofs": int(basis_ref.N),
        "reference_quadrature_order": int(quadrature_order),
        "reference_l2_norm": float(ref_l2_norm),
        "reference_h1_norm": float(ref_h1_norm),
        "reference_h1_semi_norm": float(ref_h1_semi_norm),
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
