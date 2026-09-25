"""Compare adaptive and uniform refinement for one selected load method.

Run from the project root:
    python -m experiments.run_refinement_comparison
    python -m experiments.run_refinement_comparison --method monte_carlo
    python -m experiments.run_refinement_comparison --method monte_carlo --plot-only

The error calculation uses a separate enriched final-mesh reference for each
strategy. The plotted values are relative H1 seminorm and L2 errors, with the
same shortened labels as the load comparison.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from afem.core.afem import run_afem
from afem.core.config import AFEMConfig
from afem.core.ufem import run_ufem
from afem.problems.rhs import high_oscillation
from afem.utils.plotting import plot_reference_error_comparison

METHOD = "monte_carlo"  # "midpoint", "monte_carlo", or "quadrature"
RHS = high_oscillation
UNIFORM_ITERATIONS = 5  # Includes initial solve; triangle count grows by 4 each step.
CONFIG = AFEMConfig(
    domain="lshape",
    dim=2,
    initial_refinements=2, #reduced so we do not get that much elements for uniform refinement
    max_iterations=9,  # Adaptive iterations; independent of UNIFORM_ITERATIONS.
    theta=0.5,
    quadrature_order=2,
    estimator_quadrature_order=10,
    mc_samples_per_element=20,
    mc_seed=123,
    compute_reference_error=True,
    reference_error_method="direct",
    reference_order=3,
    reference_quadrature_order=19,
    save_plots=True,
    plot_every=2,
    output_dir=Path("results/refinement_comparison"),
)

METHODS = {
    "midpoint": ("quadrature", "midpoint", r"$\Pi_0 f$"),
    "monte_carlo": ("monte_carlo", "default", r"$\hat{\Pi}_0 f$"),
    "quadrature": ("quadrature", "default", r"$f$"),
}
STRATEGIES = (
    ("adaptive", "AFEM", "tab:blue"),
    ("uniform", "Uniform refinement", "tab:orange"),
)


def _method_settings(method: str):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {tuple(METHODS)}")
    return METHODS[method]


def plot_saved_comparison(output_dir: Path, method: str):
    """Replot the completed strategies using only their saved histories."""
    _, _, method_label = _method_settings(method)
    method_dir = Path(output_dir) / method
    runs = []
    for strategy, label, color in STRATEGIES:
        history_path = method_dir / strategy / "history.json"
        if history_path.exists():
            data = json.loads(history_path.read_text(encoding="utf-8"))
            runs.append({"history": data["history"], "label": label, "color": color})
    if not runs:
        raise FileNotFoundError(f"No refinement histories found in {method_dir}")
    plot_path = method_dir / "refinement_errors_comparison.png"
    plot_reference_error_comparison(
        runs, plot_path, title=f"Adaptive vs. uniform refinement: {method_label}",
    )
    return plot_path


def run_comparison(config: AFEMConfig, rhs, *, method: str, uniform_iterations: int):
    """Save the full adaptive run before starting the uniform run."""
    load_method, quadrature_rule, _ = _method_settings(method)
    if config.max_iterations < 1 or uniform_iterations < 1:
        raise ValueError("Adaptive and uniform iteration counts must be at least 1")
    method_dir = Path(config.output_dir) / method
    for strategy, label, _ in STRATEGIES:
        run_config = replace(
            config,
            load_method=load_method,
            quadrature_rule=quadrature_rule,
            refinement_strategy=strategy,
            max_iterations=(config.max_iterations if strategy == "adaptive" else uniform_iterations),
            compute_reference_error=True,
            reference_error_method="direct",
            output_dir=method_dir / strategy,
        )
        print(f"\nStarting {label}: {method}, {run_config.max_iterations} iterations", flush=True)
        runner = run_afem if strategy == "adaptive" else run_ufem
        mesh, u, _ = runner(run_config, rhs)
        # The runner has already saved history.json, reference data, and plots.
        np.savez_compressed(run_config.output_dir / "final_solution.npz", p=mesh.p, t=mesh.t, u=u)
        metadata = {
            "method": method,
            "refinement_strategy": strategy,
            "rhs_module": getattr(rhs, "__module__", None),
            "rhs_name": getattr(rhs, "__qualname__", type(rhs).__name__),
            "error_norms": ["relative_h1_seminorm", "relative_l2"],
            "reference_strategy": "enriched final mesh of each refinement strategy",
        }
        (run_config.output_dir / "experiment.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8",
        )
        print(f"Saved {label} results to {run_config.output_dir}", flush=True)
        del mesh, u

    plot_path = plot_saved_comparison(config.output_dir, method)
    print(f"Saved comparison to {plot_path}", flush=True)
    return plot_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=tuple(METHODS), default=METHOD)
    parser.add_argument("--plot-only", action="store_true", help="Replot saved results without solving.")
    args = parser.parse_args()
    if args.plot_only:
        print(plot_saved_comparison(CONFIG.output_dir, args.method))
    else:
        run_comparison(CONFIG, RHS, method=args.method, uniform_iterations=UNIFORM_ITERATIONS)
