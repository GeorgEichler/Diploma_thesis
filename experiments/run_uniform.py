"""Run with python -m experiments.run_uniform from the project root."""

from pathlib import Path  # noqa: I001

from afem.core.config import AFEMConfig
from afem.core.ufem import run_ufem
from afem.problems.rhs import high_oscillation


RHS = high_oscillation
CONFIG = AFEMConfig(
    domain="lshape",
    dim=2,
    initial_refinements=2,
    max_iterations=4,  # Includes the initial solve; each refinement quadruples triangles.
    load_method="quadrature",  # "monte_carlo" uses the sampling settings below.
    quadrature_rule="default",  # "midpoint" selects centroid assembly instead.
    quadrature_order=2,
    estimator_quadrature_order=10,
    mc_samples_per_element=20,
    mc_seed=123,
    compute_reference_error=True,
    reference_error_method="direct",
    reference_order=3,
    reference_quadrature_order=19,
    save_plots=True,
    plot_every=1,
    output_dir=Path("results/uniform_quadrature"),
)


if __name__ == "__main__":
    run_ufem(CONFIG, RHS)
