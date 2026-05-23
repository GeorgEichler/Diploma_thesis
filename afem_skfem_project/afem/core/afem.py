from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .config import AFEMConfig
from .mesh_factory import make_mesh
from .solver import solve_poisson
from .estimator import residual_estimator
from .marking import doerfler_marking
from afem.utils.plotting import plot_mesh, plot_solution_2d, plot_history


def _refine_marked(mesh, marked: np.ndarray):
    if marked.size == 0:
        return mesh
    # scikit-fem supports local refinement by element indices for simplex meshes.
    return mesh.refined(marked)


def run_afem(config: AFEMConfig, rhs):
    out = Path(config.output_dir)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    mesh = make_mesh(config.domain, config.initial_refinements)
    history: list[dict] = []

    for level in range(config.max_iterations):
        # Important for reproducible MC: use same seed policy for a given level.
        seed_l = config.mc_seed + level
        u, basis, fbar = solve_poisson(
            mesh, rhs, config.load_method, config.mc_samples_per_element, seed_l
        )
        eta = residual_estimator(mesh, u, rhs, fbar=fbar)
        estimator = float(np.linalg.norm(eta))
        ndofs = int(basis.N)
        nelems = int(mesh.t.shape[1])
        marked = doerfler_marking(eta, config.theta)

        history.append({
            "level": level,
            "ndofs": ndofs,
            "nelems": nelems,
            "estimator": estimator,
            "nmarked": int(marked.size),
        })
        print(f"level={level:02d} ndofs={ndofs:7d} nelems={nelems:7d} "
              f"eta={estimator:.4e} marked={marked.size}")

        if config.save_plots and (level % config.plot_every == 0 or level == config.max_iterations - 1):
            plot_mesh(mesh, plots / f"mesh_l{level:02d}.png")
            if mesh.p.shape[0] == 2:
                plot_solution_2d(mesh, u, plots / f"solution_l{level:02d}.png")

        if level < config.max_iterations - 1:
            mesh = _refine_marked(mesh, marked)

    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump({"config": asdict(config), "history": history}, f, indent=2, default=str)
    plot_history(history, plots / "estimator_vs_ndofs.png")
    return mesh, u, history
