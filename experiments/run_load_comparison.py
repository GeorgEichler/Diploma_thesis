"""Sequential midpoint, Monte Carlo, and scikit-fem quadrature comparison.

Run from the project root with:
    python -m experiments.run_load_comparison
Recreate the combined plot from saved histories with:
    python -m experiments.run_load_comparison --plot-only

Each method uses its own enriched final-mesh reference, as in run_afem.
The labels Pi_0 f and f denote midpoint and standard quadrature here;
neither midpoint cell averages nor integration of arbitrary f are exact.
"""

import argparse  # noqa: I001
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from afem.core.afem import run_afem
from afem.core.config import AFEMConfig
from afem.problems.rhs import high_oscillation
from afem.utils.plotting import plot_reference_error_comparison


# Select the same RHS and common settings for all three methods here.
RHS = high_oscillation
CONFIG = AFEMConfig(
    domain="lshape",
    dim=2,
    initial_refinements=3,
    max_iterations=8,  # Monte Carlo and standard quadrature
    theta=0.5,
    mc_samples_per_element=20,
    mc_seed=123,
    quadrature_order=10,  # 10 is the default value for the quadrature
    estimator_quadrature_order=10,  # integrate f**2; ignored for midpoint/MC
    compute_reference_error=True,
    reference_quadrature_order=19, # 10 is the default value for the quadrature
    reference_error_method="direct",
    save_plots=True,
    plot_every=3,
    output_dir=Path("results/load_comparison_high_quadrature"),
)

# Use more adaptive levels for midpoint; None uses CONFIG.max_iterations.
MIDPOINT_ITERATIONS = 14

# Tuple order is also execution order.
METHODS = (
    ("midpoint", "quadrature", "midpoint", r"$\Pi_0 f$", "tab:blue"),
    ("monte_carlo", "monte_carlo", "default", r"$\hat{\Pi}_0 f$", "tab:orange"),
    ("quadrature", "quadrature", "default", r"$f$", "tab:green"),
)


def plot_saved_comparison(output_dir: Path):
    """Replot completed methods without rerunning any numerical solves."""
    output_dir = Path(output_dir)
    runs = []
    for name, _, _, label, color in METHODS:
        history_path = output_dir / name / "history.json"
        if history_path.exists():
            data = json.loads(history_path.read_text(encoding="utf-8"))
            runs.append({"history": data["history"], "label": label, "color": color})
    if not runs:
        raise FileNotFoundError(f"No method histories found in {output_dir}")
    plot_path = output_dir / "relative_errors_comparison.png"
    plot_reference_error_comparison(runs, plot_path)
    return plot_path


def run_comparison(config: AFEMConfig, rhs, *, midpoint_iterations: int | None = None):
    """Persist each method in sequence, optionally running midpoint longer."""
    if midpoint_iterations is not None and midpoint_iterations < 1:
        raise ValueError("midpoint_iterations must be at least 1")
    output_dir = Path(config.output_dir)
    for name, load_method, quadrature_rule, _, _ in METHODS:
        method_config = replace(
            config,
            load_method=load_method,
            quadrature_rule=quadrature_rule,
            max_iterations=(
                midpoint_iterations
                if name == "midpoint" and midpoint_iterations is not None
                else config.max_iterations
            ),
            compute_reference_error=True,
            reference_error_method="direct",
            output_dir=output_dir / name,
        )
        print(f"\nStarting {name} ({method_config.max_iterations} iterations)", flush=True)
        # run_afem writes history.json, including reference errors, before returning.
        mesh, u, _ = run_afem(method_config, rhs)
        np.savez_compressed(
            method_config.output_dir / "final_solution.npz",
            p=mesh.p, t=mesh.t, u=u,
        )
        metadata = {
            "method": name,
            "rhs_module": getattr(rhs, "__module__", None),
            "rhs_name": getattr(rhs, "__qualname__", type(rhs).__name__),
            "error_norms": ["relative_h1_seminorm", "relative_l2"],
            "reference_strategy": "enriched final mesh of each method",
        }
        (method_config.output_dir / "experiment.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8",
        )
        print(f"Saved {name} results to {method_config.output_dir}", flush=True)
        del mesh, u

    plot_path = plot_saved_comparison(output_dir)
    print(f"Saved comparison to {plot_path}", flush=True)
    return plot_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true", help="Plot saved histories without solving.")
    args = parser.parse_args()
    if args.plot_only:
        print(plot_saved_comparison(CONFIG.output_dir))
    else:
        run_comparison(CONFIG, RHS, midpoint_iterations=MIDPOINT_ITERATIONS)
