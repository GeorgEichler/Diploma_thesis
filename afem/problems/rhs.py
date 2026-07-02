from __future__ import annotations
import numpy as np

Array = np.ndarray

def constant_one(x: Array) -> Array:
    """f(x)=1. x has shape (dim, npoints)."""
    return np.ones(x.shape[1])

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

def localized_step_2d(x: Array) -> Array:
    """Indicator RHS: one on a small square subset of the unit square."""
    inside = (
        (0.45 <= x[0]) & (x[0] <= 0.46) &
        (0.45 <= x[1]) & (x[1] <= 0.46)
    )
    return inside.astype(float)
