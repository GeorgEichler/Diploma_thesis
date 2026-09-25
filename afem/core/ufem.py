"""Uniform-refinement counterpart of the adaptive FEM runner."""

from dataclasses import replace

from .afem import run_afem
from .config import AFEMConfig


def run_ufem(config: AFEMConfig, rhs):
    """Solve, estimate, and uniformly refine using the existing AFEM settings.

    All elements are refined between solves, even if the estimator is zero.
    max_iterations counts solved levels, including the initial mesh. The
    marking parameter theta is ignored. Both reference-error methods, energy
    evaluation, plots, and history.json use the same implementation as AFEM.

    Returns (final_mesh, final_solution, history), just like run_afem.
    The supplied frozen configuration is not modified; the saved configuration
    records refinement_strategy="uniform".
    """
    return run_afem(replace(config, refinement_strategy="uniform"), rhs)
