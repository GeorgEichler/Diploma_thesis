from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

LoadMethod = Literal["quadrature", "monte_carlo"]
DomainName = Literal["unit_square", "lshape", "unit_cube"]

@dataclass(frozen=True)
class AFEMConfig:
    domain: DomainName = "unit_square"
    dim: int = 2
    initial_refinements: int = 2
    max_iterations: int = 25
    theta: float = 0.5
    load_method: LoadMethod = "quadrature"
    mc_samples_per_element: int = 32
    mc_seed: int = 12345
    save_plots: bool = True
    plot_every: int = 5
    output_dir: Path = Path("results")
    solver: Literal["direct"] = "direct"
