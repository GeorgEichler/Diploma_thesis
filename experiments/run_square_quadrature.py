from pathlib import Path
from afem.core.config import AFEMConfig
from afem.core.afem import run_afem
from afem.problems.rhs import manufactured_sine_2d, high_oscillation

if __name__ == "__main__":
    cfg = AFEMConfig(
        domain="unit_square",
        dim=2,
        initial_refinements=2,
        max_iterations=12,
        theta=0.5,
        load_method="quadrature",
        output_dir=Path("results/square_quadrature"),
        quadrature_rule="midpoint",
        plot_every=3,
        compute_reference_error=True,
    )
    run_afem(cfg, high_oscillation)
