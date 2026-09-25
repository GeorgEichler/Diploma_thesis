from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from afem.utils.plotting import (
    plot_history,
    plot_mesh,
    plot_reference_error_history,
    plot_solution_2d,
)

from .config import AFEMConfig
from .energy_error_norms import (
    add_energy_error_to_history,
    default_energy_quadrature_order,
    dirichlet_energy,
    reference_solution_direct_errors,
    reference_solution_energy,
    reference_solution_energy_errors,
)
from .estimator import residual_estimator
from .marking import doerfler_marking
from .mesh_factory import make_mesh
from .solver import solve_poisson


def _refine_marked(mesh, marked: np.ndarray):
    if marked.size == 0:
        return mesh
    # scikit-fem supports local refinement by element indices for simplex meshes.
    return mesh.refined(marked)


def run_afem(config: AFEMConfig, rhs):
    """Run P1 FEM with the configured refinement strategy and error evaluation."""
    if config.refinement_strategy not in ("adaptive", "uniform"):
        raise ValueError(f"Unknown refinement_strategy: {config.refinement_strategy}")
    if config.max_iterations < 1:
        raise ValueError("max_iterations must be at least 1")
    out = Path(config.output_dir)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    mesh = make_mesh(config.domain, config.initial_refinements)
    history: list[dict] = []
    reference_snapshots: list[dict] = []
    use_energy_reference_error = (
        config.compute_reference_error
        and config.reference_error_method == "energy"
    )
    energy_quadrature_order = config.reference_quadrature_order
    if energy_quadrature_order is None and (config.compute_energy or config.compute_reference_error):
        energy_quadrature_order = default_energy_quadrature_order(config.reference_order)

    for level in range(config.max_iterations):
        # Important for reproducible MC: use same seed policy for a given level.
        seed_l = config.mc_seed + level
        u, basis, fbar = solve_poisson(
            mesh, rhs, config.load_method, config.mc_samples_per_element, seed_l,
            quadrature_rule=config.quadrature_rule, quadrature_order=config.quadrature_order
        )
        eta = residual_estimator(
            mesh, u, rhs, fbar=fbar,
            quadrature_rule=config.quadrature_rule,
            quadrature_order=(
                config.estimator_quadrature_order
                if config.estimator_quadrature_order is not None
                else config.quadrature_order
            ),
        )
        estimator = float(np.linalg.norm(eta))
        energy = (
            dirichlet_energy(basis, u, rhs, quadrature_order=energy_quadrature_order)
            if config.compute_energy or use_energy_reference_error
            else None
        )
        ndofs = int(basis.N)
        nelems = int(mesh.t.shape[1])
        marked = (
            np.arange(nelems, dtype=int)
            if config.refinement_strategy == "uniform"
            else doerfler_marking(eta, config.theta)
        )

        entry = {
            "level": level,
            "ndofs": ndofs,
            "nelems": nelems,
            "estimator": estimator,
            "nmarked": int(marked.size),
        }
        if energy is not None:
            entry["energy"] = energy
        history.append(entry)
        if config.compute_reference_error:
            reference_snapshots.append({
                "mesh": mesh,
                "u": u.copy(),
                "history_entry": entry,
            })

        msg = (f"level={level:02d} ndofs={ndofs:7d} nelems={nelems:7d} "
               f"eta={estimator:.4e} marked={marked.size}")
        if energy is not None:
            msg += f" J={energy:.4e}"
        print(msg)

        if config.save_plots and (level % config.plot_every == 0 or level == config.max_iterations - 1):
            plot_mesh(mesh, plots / f"mesh_l{level:02d}.png")
            if mesh.p.shape[0] == 2:
                plot_solution_2d(mesh, u, plots / f"solution_l{level:02d}.png")

        if level < config.max_iterations - 1:
            mesh = (
                mesh.refined()
                if config.refinement_strategy == "uniform"
                else _refine_marked(mesh, marked)
            )

    # Compute reference solution on final mesh and append selected error data.
    reference_data = None
    if config.compute_reference_error:
        if config.reference_error_method == "direct":
            reference_data = reference_solution_direct_errors(
                mesh,
                rhs,
                reference_snapshots,
                reference_order=config.reference_order,
                quadrature_order=energy_quadrature_order,
            )
        elif config.reference_error_method == "energy":
            # Use either reference_solution_energy -> compute energy on coarse mesh
            # reference_solution_energy_errors -> prolong to final mesh
            reference_data = reference_solution_energy_errors(
                mesh,
                rhs,
                reference_snapshots,
                reference_order=config.reference_order,
                quadrature_order=energy_quadrature_order,
            )
            #reference_data = reference_solution_energy(mesh, rhs,
            #                                           reference_order=config.reference_order,
            #                                           quadrature_order=config.quadrature_order)
            #reference_data["reference_error_method"] = "energy"
            #add_energy_error_to_history(history, reference_data["reference_energy"])
        else:
            raise ValueError(f"Unknown reference_error_method: {config.reference_error_method}")

    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": asdict(config),
                "reference": reference_data,
                "history": history,
            },
            f,
            indent=2,
            default=str,
        )
    if config.save_history_plots:
        plot_history(history, plots / "estimator_vs_ndofs.png")
    if config.save_history_plots and config.compute_reference_error:
        plot_reference_error_history(history, plots / "reference_error_vs_ndofs.png")
    return mesh, u, history
