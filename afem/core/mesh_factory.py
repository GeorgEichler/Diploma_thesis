from __future__ import annotations

import numpy as np
from skfem import MeshTet, MeshTri


def _lshape_four_triangles():
    """Split each of three unit squares into four triangles around its centre."""
    points = np.array([
        [-1, -1], [0, -1], [1, -1],
        [-1, 0], [0, 0], [1, 0],
        [-1, 1], [0, 1],
        [-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5],
    ], dtype=float).T
    # Shared corner indices give a conforming mesh across the square interfaces.
    squares = (
        ((0, 1, 4, 3), 8),
        ((3, 4, 7, 6), 9),
        ((1, 2, 5, 4), 10),
    )
    triangles = np.array([
        (corners[i], corners[(i + 1) % 4], centre)
        for corners, centre in squares
        for i in range(4)
    ], dtype=int).T
    return MeshTri(points, triangles)


def make_mesh(domain: str, initial_refinements: int = 0, *, initial_mesh: str = "default"):
    """Create a base mesh, then apply uniform initial refinements.

    Supported:
      - unit_square: triangular mesh of [0,1]^2
      - lshape: built-in L-shaped mesh, or four triangles per unit square
      - unit_cube: tetrahedral mesh of [0,1]^3
    """
    if initial_mesh not in ("default", "lshape_four_triangles"):
        raise ValueError(f"Unknown initial_mesh: {initial_mesh}")
    if initial_mesh == "lshape_four_triangles" and domain != "lshape":
        raise ValueError("initial_mesh='lshape_four_triangles' requires domain='lshape'")

    if domain == "unit_square":
        mesh = MeshTri()
    elif domain == "lshape":
        if initial_mesh == "lshape_four_triangles":
            mesh = _lshape_four_triangles()
        else:
            if not hasattr(MeshTri, "init_lshaped"):
                raise RuntimeError("Your scikit-fem version has no MeshTri.init_lshaped().")
            mesh = MeshTri.init_lshaped()
    elif domain == "unit_cube":
        mesh = MeshTet()
    else:
        raise ValueError(f"Unknown domain: {domain}")
    return mesh.refined(initial_refinements)
