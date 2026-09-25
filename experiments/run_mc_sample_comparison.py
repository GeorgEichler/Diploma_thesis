"""Compare fixed Monte Carlo sample counts for the same RHS.

Run from the project root:
    python -m experiments.run_mc_sample_comparison
    python -m experiments.run_mc_sample_comparison --samples 1 2 5 10 20
    python -m experiments.run_mc_sample_comparison --plot-only
"""

import argparse
import json
from dataclasses import replace
from numbers import Integral
from pathlib import Path

import numpy as np

from afem.core.afem import run_afem
from afem.core.config import AFEMConfig
from afem.problems.rhs import high_oscillation
from afem.utils.sample_comparison import plot_sample_comparison

RHS = high_oscillation
# One full run per entry; each count is fixed throughout that run's AFEM levels.
SAMPLE_COUNTS = [1, 5, 20]
CONFIG = AFEMConfig(
    domain="lshape",
    dim=2,
    initial_refinements=2,
    max_iterations=8,
    theta=0.5,
    load_method="monte_carlo",
    mc_seed=123,
    refinement_strategy="adaptive",  # Uniform refinement is also supported.
    compute_reference_error=True,
    reference_error_method="direct",
    reference_order=3,
    reference_quadrature_order=19,
    save_plots=False,
    save_history_plots=False,
    output_dir=Path("results/mc_sample_comparison"),
)


def _validate_counts(sample_counts):
    counts = list(sample_counts)
    if not counts or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n < 1
                         for n in counts):
        raise ValueError("sample_counts must be a nonempty sequence of positive integers")
    counts = [int(n) for n in counts]
    if len(set(counts)) != len(counts):
        raise ValueError("sample_counts must not contain duplicates")
    return counts


def _write_manifest(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def plot_saved_comparison(output_dir):
    """Replot only completed runs listed in the current study manifest."""
    out = Path(output_dir)
    manifest = json.loads((out / "sample_comparison.json").read_text(encoding="utf-8"))
    runs = []
    for run in manifest["completed_runs"]:
        data = json.loads((out / run["directory"] / "history.json").read_text(encoding="utf-8"))
        runs.append({"samples": run["samples"], "history": data["history"]})
    return plot_sample_comparison(runs, out)


def run_comparison(config: AFEMConfig, rhs, *, sample_counts):
    counts = _validate_counts(sample_counts)
    if config.max_iterations < 1 or config.mc_seed < 0:
        raise ValueError("max_iterations must be positive and mc_seed nonnegative")
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sample_counts": counts,
        "base_seed": config.mc_seed,
        "seed_stride": config.max_iterations,
        "seed_policy": "base + sample_count * max_iterations + level; independent of list order",
        "rhs_module": getattr(rhs, "__module__", None),
        "rhs_name": getattr(rhs, "__qualname__", type(rhs).__name__),
        "reference_strategy": "separate enriched final-mesh reference per sample count",
        "completed_runs": [],
    }
    _write_manifest(out / "sample_comparison.json", manifest)
    for samples in counts:
        # Disjoint level-seed ranges, unchanged if counts are reordered or added.
        seed = config.mc_seed + samples * config.max_iterations
        directory = f"samples_{samples}"
        cfg = replace(
            config, load_method="monte_carlo", quadrature_rule="default",
            mc_samples_per_element=samples, mc_seed=seed,
            compute_reference_error=True, reference_error_method="direct",
            save_plots=False, save_history_plots=False, output_dir=out / directory,
        )
        print(f"\nMonte Carlo: {samples} samples per element, seed={seed}", flush=True)
        mesh, u, _ = run_afem(cfg, rhs)
        np.savez_compressed(cfg.output_dir / "final_solution.npz", p=mesh.p, t=mesh.t, u=u)
        manifest["completed_runs"].append({"samples": samples, "seed": seed, "directory": directory})
        _write_manifest(out / "sample_comparison.json", manifest)
        print(f"Saved results to {cfg.output_dir}", flush=True)
        del mesh, u
    paths = plot_saved_comparison(out)
    for path in paths:
        print(f"Saved comparison to {path}", flush=True)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, nargs="+", default=SAMPLE_COUNTS)
    parser.add_argument("--plot-only", action="store_true", help="Replot saved data without solving.")
    args = parser.parse_args()
    if args.plot_only:
        for path in plot_saved_comparison(CONFIG.output_dir):
            print(path)
    else:
        run_comparison(CONFIG, RHS, sample_counts=args.samples)
