from pathlib import Path
from afem.core.config import AFEMConfig
from afem.core.afem import run_afem
from afem.problems.rhs import manufactured_sine_2d

"""
Test of AFEM method for the Dirichlet problem
- \Delta u = 2*pi^2*sin(pi*x)*sin(pi*y)   in [0,1]^2
         u = 0                            on \partial [0,1]^2
with exact solution u(x,y) = sin(pi*x)*sin(pi*y) and
Dirichlet energy J(u) = pi^2/4
"""

if __name__ == "__main__":
    cfg = AFEMConfig(
        domain="unit_square",
        dim=2,
        initial_refinements=2,
        max_iterations=12,
        theta=0.5,
        load_method="quadrature",
        output_dir=Path("testing/test_square_quadrature"),
        quadrature_rule="midpoint",
        plot_every=3,
        compute_reference_error=True,
    )
    run_afem(cfg, manufactured_sine_2d)