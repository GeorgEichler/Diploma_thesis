from pathlib import Path
from afem.core.config import AFEMConfig
from afem.core.afem import run_afem
from afem.problems.rhs import constant_one

if __name__ == "__main__":
    cfg = AFEMConfig(
        domain="lshape",
        dim=2,
        initial_refinements=1,
        max_iterations=20,
        theta=0.5,
        load_method="monte_carlo",
        mc_samples_per_element=64,
        mc_seed=2026,
        output_dir=Path("results/lshape_mc"),
        plot_every=5,
    )
    run_afem(cfg, constant_one)
