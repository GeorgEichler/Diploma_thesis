from pathlib import Path
from afem.core.config import AFEMConfig
from afem.core.afem import run_afem
from afem.problems.rhs import manufactured_sine_3d

if __name__ == "__main__":
    cfg = AFEMConfig(
        domain="unit_cube",
        dim=3,
        initial_refinements=1,
        max_iterations=8,
        theta=0.5,
        load_method="quadrature",
        output_dir=Path("results/cube_quadrature"),
        plot_every=2,
    )
    run_afem(cfg, manufactured_sine_3d)
