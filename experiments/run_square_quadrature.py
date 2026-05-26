from pathlib import Path
from afem.core.config import AFEMConfig
from afem.core.afem import run_afem
from afem.problems.rhs import high_oscillation

if __name__ == "__main__":
    cfg = AFEMConfig(
        domain="lshape",
        dim=2,
        initial_refinements=3,
        max_iterations=12,
        theta=0.5,
        load_method="quadrature",
        output_dir=Path("results/lshape_midpoint"),
        quadrature_rule="midpoint",
        plot_every=3,
        compute_reference_error=True,
        reference_error_method="direct",
    )
    run_afem(cfg, high_oscillation)