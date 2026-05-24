from __future__ import annotations

import numpy as np


def midpoint_quadrature(dim: int):
    """
    Return midpoint quadrature on the reference simplex.

    For triangles:
        reference triangle has area 1/2,
        midpoint is (1/3, 1/3).

    For tetrahedra:
        reference tetrahedron has volume 1/6,
        midpoint is (1/4, 1/4, 1/4).
    """
    if dim == 2:
        X = np.array([[1.0 / 3.0],
                      [1.0 / 3.0]])
        W = np.array([0.5])
        return X, W

    if dim == 3:
        X = np.array([[1.0 / 4.0],
                      [1.0 / 4.0],
                      [1.0 / 4.0]])
        W = np.array([1.0 / 6.0])
        return X, W

    raise ValueError(f"Unsupported dimension for midpoint quadrature: {dim}")


def get_quadrature(dim: int, rule: str):
    if rule == "default":
        return None

    if rule == "midpoint":
        return midpoint_quadrature(dim)

    raise ValueError(f"Unknown quadrature rule: {rule}")