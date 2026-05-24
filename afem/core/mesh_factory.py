from __future__ import annotations
from skfem import MeshTri, MeshTet


def make_mesh(domain: str, initial_refinements: int = 0):
    """Create initial mesh.

    Supported:
      - unit_square: triangular mesh of [0,1]^2
      - lshape: built-in L-shaped triangular mesh if available
      - unit_cube: tetrahedral mesh of [0,1]^3
    """
    if domain == "unit_square":
        mesh = MeshTri().refined(initial_refinements)
    elif domain == "lshape":
        if not hasattr(MeshTri, "init_lshaped"):
            raise RuntimeError("Your scikit-fem version has no MeshTri.init_lshaped().")
        mesh = MeshTri.init_lshaped().refined(initial_refinements)
    elif domain == "unit_cube":
        mesh = MeshTet().refined(initial_refinements)
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return mesh
