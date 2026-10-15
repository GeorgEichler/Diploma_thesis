"""Sequential midpoint, Monte Carlo, and scikit-fem quadrature comparison.

Run from the project root with:
    python -m experiments.run_load_comparison
Recreate the combined plot from saved histories with:
    python -m experiments.run_load_comparison --plot-only
"""

import argparse  # noqa: I001
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from afem.core.afem import run_afem
from afem.core.config import AFEMConfig
from afem.problems.rhs import high_oscillation, low_oscillation
from afem.utils.plotting import plot_estimator_comparison, plot_reference_error_comparison


# Select the same RHS and common settings for all three methods here.
RHS = low_oscillation
CONFIG = AFEMConfig(
    domain="lshape",
    initial_mesh="lshape_four_triangles",
    dim=2,
    initial_refinements=1,
    max_iterations=11,  # Default unless overridden below
    theta=0.5,
    mc_samples_per_element=20,
    mc_seed=123,
    quadrature_order=2,  # 10 is the default value for the quadrature, use 2 for low quadrature
    estimator_quadrature_order=2,  # integrate f**2; ignored for midpoint/MC, chose 2 for low quadrature
    compute_reference_error=True,
    reference_quadrature_order=19, # 10 is the default value for the quadrature
    reference_error_method="direct",
    save_plots=True,
    plot_every=2,
    output_dir=Path("results/load_comparison_low_quadrature_high_oscillation"),
)

# Use more adaptive levels for midpoint; None uses CONFIG.max_iterations.
MIDPOINT_ITERATIONS = 12
# Optional standard quadrature limit; None uses CONFIG.max_iterations.
QUADRATURE_ITERATIONS: int | None = None
# Error-plot convergence guides, e.g. (1.0 / 2.0,); empty means none.
REFERENCE_RATES: tuple[float, ...] = (1.0 / 2.0,)
# Optionally provide one (ndof, error) start point for each rate.
REFERENCE_STARTS: tuple[tuple[float, float], ...] | None = ((33, 0.25),)

# Tuple order is also execution order.
METHODS = (
    ("midpoint", "quadrature", "midpoint", r"$\Pi_0 f$", "tab:blue", "x"),
    ("monte_carlo", "monte_carlo", "default", r"$\hat{\Pi}_0 f$", "tab:orange", "s"),
    ("quadrature", "quadrature", "default", r"$f$", "tab:green", "o"),
)


def plot_saved_comparison(output_dir: Path, *, save_pdf=True, save_png=False):
    """Replot estimators and errors; return the error plot path as before."""
    output_dir = Path(output_dir)
    runs = []
    for name, _, _, label, color, marker in METHODS:
        history_path = output_dir / name / "history.json"
        if history_path.exists():
            data = json.loads(history_path.read_text(encoding="utf-8"))
            runs.append({
                "history": data["history"],
                "label": label,
                "color": color,
                "marker": marker,
            })
    if not runs:
        raise FileNotFoundError(f"No method histories found in {output_dir}")
    plot_estimator_comparison(
        runs, output_dir / "estimators_comparison.pdf",
        save_pdf=save_pdf, save_png=save_png,
    )
    plot_path = output_dir / "relative_errors_comparison.pdf"
    return plot_reference_error_comparison(
        runs, plot_path, reference_rates=REFERENCE_RATES, title="",
        reference_starts=REFERENCE_STARTS, save_pdf=save_pdf, save_png=save_png,
    )


def run_comparison(
    config: AFEMConfig, rhs, *, midpoint_iterations: int | None = None,
    quadrature_iterations: int | None = None,
):
    """Persist each method in sequence with optional per-method iteration limits."""
    if midpoint_iterations is not None and midpoint_iterations < 1:
        raise ValueError("midpoint_iterations must be at least 1")
    if quadrature_iterations is not None and quadrature_iterations < 1:
        raise ValueError("quadrature_iterations must be at least 1")
    iteration_overrides = {
        "midpoint": midpoint_iterations,
        "quadrature": quadrature_iterations,
    }
    output_dir = Path(config.output_dir)
    for name, load_method, quadrature_rule, _, _, _ in METHODS:
        iterations = iteration_overrides.get(name)
        method_config = replace(
            config,
            load_method=load_method,
            quadrature_rule=quadrature_rule,
            max_iterations=(
                iterations
                if iterations is not None
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

    plot_path = plot_saved_comparison(output_dir, save_pdf=config.save_pdf, save_png=config.save_png)
    if plot_path is not None:
        print(f"Saved comparison to {plot_path}", flush=True)
    return plot_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=CONFIG.output_dir,
                        help="Results directory to write or replot (defaults to CONFIG.output_dir).")
    parser.add_argument("--plot-only", action="store_true", help="Plot saved histories without solving.")
    parser.add_argument("--png", action=argparse.BooleanOptionalAction, default=CONFIG.save_png,
                        help="Save 600 dpi PNG images (disabled by default).")
    parser.add_argument("--pdf", action=argparse.BooleanOptionalAction, default=CONFIG.save_pdf,
                        help="Save vector PDFs (enabled by default).")
    args = parser.parse_args()
    if args.plot_only:
        path = plot_saved_comparison(args.output_dir, save_pdf=args.pdf, save_png=args.png)
        if path is not None:
            print(path)
    else:
        run_comparison(replace(CONFIG, output_dir=args.output_dir, save_pdf=args.pdf, save_png=args.png), RHS,
                       midpoint_iterations=MIDPOINT_ITERATIONS,
                       quadrature_iterations=QUADRATURE_ITERATIONS)
