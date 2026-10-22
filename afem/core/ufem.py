"""Uniform-refinement counterpart of the adaptive FEM runner."""

from dataclasses import replace

from .afem import run_afem
from .config import AFEMConfig


def run_ufem(config: AFEMConfig, rhs):
    """Solve, estimate, and uniformly refine during each step."""
    return run_afem(replace(config, refinement_strategy="uniform"), rhs)
