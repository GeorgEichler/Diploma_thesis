"""Repeated Monte Carlo experiments; run as python -m experiments.run_mc_ensemble."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from afem.core.afem import run_afem
from afem.core.config import AFEMConfig
from afem.problems.rhs import high_oscillation
from afem.utils.ensemble import aggregate_histories, plot_ensemble
from afem.utils.plotting import plot_mesh

RHS = high_oscillation
NUM_RUNS = 10
SUMMARY = "mean"  # "mean", "median", or "none"
SHOW_INDIVIDUAL = True  # Faint error/estimator trajectories, not solution fields.
SHOW_BAND = True  # Pointwise 25–75% spread across runs.
SAVE_FINAL_MESHES = False
CONFIG = AFEMConfig(
    domain="lshape",
    dim=2,
    initial_refinements=2,
    max_iterations=8,
    theta=0.5,
    load_method="monte_carlo",
    mc_samples_per_element=20,
    mc_seed=123,  # Base seed for the reproducible ensemble.
    refinement_strategy="adaptive",  # "uniform" is also supported.
    compute_reference_error=True,
    reference_error_method="direct",
    reference_order=3,
    reference_quadrature_order=19,
    save_plots=False,
    save_history_plots=False,
    output_dir=Path("results/mc_ensemble"),
)


def _write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def plot_saved_ensemble(output_dir, *, summary="median", show_individual=True, show_band=True):
    """Use only completed runs listed in this ensemble's manifest, not stale folders."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "ensemble.json").read_text(encoding="utf-8"))
    histories = [json.loads((output_dir / run["directory"] / "history.json").read_text(encoding="utf-8"))["history"]
                 for run in manifest["completed_runs"]]
    if not histories:
        raise ValueError("No completed runs available to plot")
    stats = aggregate_histories(histories)
    paths = plot_ensemble(histories, stats, output_dir, summary=summary,
                          show_individual=show_individual, show_band=show_band)
    _write_json(output_dir / "ensemble_statistics.json", stats)
    return paths


def run_ensemble(config, rhs, *, num_runs, save_final_meshes=False,
                 summary="median", show_individual=True, show_band=True):
    if num_runs < 1 or config.max_iterations < 1:
        raise ValueError("num_runs and max_iterations must be at least 1")
    if summary not in ("mean", "median", "none"):
        raise ValueError("summary must be 'mean', 'median', or 'none'")
    if summary == "none" and not show_individual and not show_band:
        raise ValueError("Enable individual curves, a summary, or the spread band")
    if config.mc_seed < 0:
        raise ValueError("mc_seed must be nonnegative")
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "requested_runs": num_runs,
        "base_seed": config.mc_seed,
        "seed_stride": config.max_iterations,
        "seed_policy": "base + run_index * max_iterations + level; no reused level seeds",
        "rhs_module": getattr(rhs, "__module__", None),
        "rhs_name": getattr(rhs, "__qualname__", type(rhs).__name__),
        "reference_strategy": "separate enriched final-mesh reference for each run",
        "completed_runs": [],
    }
    _write_json(out / "ensemble.json", manifest)
    for index in range(num_runs):
        directory = f"run_{index + 1:03d}"
        # A stride of one would reuse seeds between different runs and levels.
        seed = config.mc_seed + index * config.max_iterations
        cfg = replace(config, load_method="monte_carlo", quadrature_rule="default",
                      mc_seed=seed, compute_reference_error=True, reference_error_method="direct",
                      save_plots=False, save_history_plots=False, output_dir=out / directory)
        print(f"\nMonte Carlo run {index + 1}/{num_runs}, seed={seed}", flush=True)
        mesh, u, _ = run_afem(cfg, rhs)
        np.savez_compressed(cfg.output_dir / "final_solution.npz", p=mesh.p, t=mesh.t, u=u)
        manifest["completed_runs"].append({"directory": directory, "seed": seed})
        _write_json(out / "ensemble.json", manifest)
        if save_final_meshes:
            plot_mesh(mesh, cfg.output_dir / "plots" / "final_mesh.png")
        print(f"Saved {directory}", flush=True)
        del mesh, u
    paths = plot_saved_ensemble(out, summary=summary, show_individual=show_individual, show_band=show_band)
    for path in paths:
        print(f"Saved ensemble plot to {path}", flush=True)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=NUM_RUNS)
    parser.add_argument("--summary", choices=("mean", "median", "none"), default=SUMMARY)
    parser.add_argument("--show-individual", action=argparse.BooleanOptionalAction, default=SHOW_INDIVIDUAL)
    parser.add_argument("--show-band", action=argparse.BooleanOptionalAction, default=SHOW_BAND)
    parser.add_argument("--save-final-meshes", action=argparse.BooleanOptionalAction, default=SAVE_FINAL_MESHES)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    options = dict(summary=args.summary, show_individual=args.show_individual, show_band=args.show_band)  # noqa: C408
    if args.plot_only:
        for path in plot_saved_ensemble(CONFIG.output_dir, **options):
            print(path)
    else:
        run_ensemble(CONFIG, RHS, num_runs=args.runs, save_final_meshes=args.save_final_meshes, **options)
