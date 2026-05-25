from pathlib import Path
from afem.core.config import AFEMConfig
from afem.core.afem import run_afem
from afem.problems.rhs import high_oscillation

if __name__ == "__main__":
    cfg = AFEMConfig(
        domain="unit_square",
        dim=2,
        initial_refinements=3,
        max_iterations=10,
        theta=0.5,
        load_method="monte_carlo",
        mc_samples_per_element=40,
        mc_seed=2026,
        output_dir=Path("results/square_mc"),
        plot_every=3,
        compute_reference_error=True,
    )
    run_afem(cfg, high_oscillation)
