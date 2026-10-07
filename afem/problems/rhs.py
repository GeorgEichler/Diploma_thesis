from __future__ import annotations

import numpy as np

Array = np.ndarray

def constant_one(x: Array) -> Array:
    """f(x)=1. x has shape (dim, npoints)."""
    return np.ones_like(x[0])

def oscillatory_2d(x: Array, k: float = 12.0) -> Array:
    """Smooth oscillatory right-hand side on 2D domains."""
    return np.sin(k * np.pi * x[0]) * np.sin(k * np.pi * x[1])

def manufactured_sine_2d(x: Array) -> Array:
    """RHS for exact u=sin(pi x) sin(pi y) on unit square."""
    return 2.0 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

def manufactured_sine_3d(x: Array) -> Array:
    """RHS for exact u=prod_i sin(pi x_i) on unit cube."""
    return 3.0 * np.pi**2 * np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2])

def high_oscillation(x: Array) -> Array:
    """f(x,y) = |sin(2^l * 3 * pi)|"""
    return np.abs(np.sin(np.pi * 2**5 * 3.0 * x[0]))

def low_oscillation(x: Array) -> Array:
    """f(x,y) = |sin(2^l * 3 * pi)|"""
    return np.abs(np.sin(np.pi * 2 * 3.0 * x[0]))

def rhs_lshape(x):
    """
    Right-hand side f = -Delta u_exact for the L-shaped domain

        Omega = (-1,1)^2 \\ ([0,1] x [0,1])

    with

        u_exact = r^(2/3) sin(2 theta / 3)
                  (1-x^2)(1-y^2).

    x can have shape (2, npoints) or
    (2, nelements, npoints), as in scikit-fem.
    """

    X = x[0]
    Y = x[1]

    r = np.sqrt(X**2 + Y**2)

    # Angle measured counterclockwise from the positive y-axis.
    # On the L-shaped domain: theta in [0, 3*pi/2].
    theta = np.mod(
        np.arctan2(Y, X) - np.pi / 2.0,
        2.0 * np.pi
    )

    # Avoid division by zero at the reentrant corner.
    r_safe = np.where(r > 0.0, r, 1.0)

    alpha = 2.0 / 3.0

    sin_term = np.sin(alpha * theta)
    cos_term = np.cos(alpha * theta)

    # Singular harmonic function
    s = r_safe**alpha * sin_term

    # Its Cartesian derivatives
    sx = (
        alpha
        * r_safe**(alpha - 2.0)
        * (X * sin_term - Y * cos_term)
    )

    sy = (
        alpha
        * r_safe**(alpha - 2.0)
        * (Y * sin_term + X * cos_term)
    )

    f = (
        2.0 * (2.0 - r**2) * s
        + 4.0 * X * (1.0 - Y**2) * sx
        + 4.0 * Y * (1.0 - X**2) * sy
    )

    # f -> 0 as r -> 0, so define its value at the corner by continuity.
    f = np.where(r > 0.0, f, 0.0)

    return f

def localized_step_2d(x: Array) -> Array:
    """Indicator RHS: one on a small square subset of the unit square."""
    inside = (
        (0.45 <= x[0]) & (x[0] <= 0.46) &
        (0.45 <= x[1]) & (x[1] <= 0.46)
    )
    return inside.astype(float)
