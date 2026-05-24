from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LoadMethod = Literal["quadrature", "monte_carlo"]
# Default method uses scikit fems quadrature rule for computing integrals
QuadratureRule = Literal["default", "midpoint"]
DomainName = Literal["unit_square", "lshape", "unit_cube"]

@dataclass(frozen=True)
class AFEMConfig:
    domain: DomainName = "unit_square"
    dim: int = 2
    initial_refinements: int = 2
    max_iterations: int = 10
    theta: float = 0.5 # for Dörfler marrking

    load_method: LoadMethod = "quadrature"
    quadrature_rule: QuadratureRule = "default"
    quadrature_order: int | None = None

    mc_samples_per_element: int = 32
    mc_seed: int = 12345
    save_plots: bool = True
    plot_every: int = 5
    output_dir: Path = Path("results")
    solver: Literal["direct"] = "direct"

    # Store J(u_h) in the history without necessarily solving a reference problem.
    compute_energy: bool = False
    compute_reference_error: bool = False
    # Order of the Lagrange polynomial space
    reference_order: int = 3
    # Order of quadrature rule for solving energy integral of reference solution
    reference_quadrature_order: int | None = None
