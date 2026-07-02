from pathlib import Path
from afem.core.config import AFEMConfig
from afem.core.afem import run_afem
from afem.problems.rhs import high_oscillation, localized_step_2d

if __name__ == "__main__":
    cfg = AFEMConfig(
        domain="unit_square",
        dim=2,
        initial_refinements=3,
        max_iterations=8,
        theta=0.5,
        load_method="monte_carlo",
        mc_samples_per_element=20,
        mc_seed=12345,
        compute_reference_error=True,
        reference_error_method="energy", #either energy or direct
        save_plots=True,
        output_dir=Path("results/step_function_square_mc_energy"),
        plot_every=3,
    )
    run_afem(cfg, localized_step_2d)
