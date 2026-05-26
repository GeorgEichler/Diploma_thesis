from pathlib import Path
from afem.core.config import AFEMConfig
from afem.core.afem import run_afem
from afem.problems.rhs import high_oscillation

if __name__ == "__main__":
    cfg = AFEMConfig(
        domain="lshape",
        dim=2,
        initial_refinements=3,
        max_iterations=7,
        theta=0.5,
        load_method="monte_carlo",
        mc_samples_per_element=20,
        mc_seed=2026,
        compute_reference_error=True,
        reference_error_method="direct",
        save_plots=True,
        output_dir=Path("results/lshape_mc"),
        plot_every=3,
    )
    run_afem(cfg, high_oscillation)
