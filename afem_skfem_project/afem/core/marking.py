from __future__ import annotations
import numpy as np


def doerfler_marking(eta: np.ndarray, theta: float) -> np.ndarray:
    """Minimal-cardinality bulk marking by descending indicators.

    Returns sorted element indices M such that sum_{K in M} eta_K^2 >= theta sum eta_K^2.
    """
    if not (0.0 < theta <= 1.0):
        raise ValueError("theta must satisfy 0 < theta <= 1")
    eta2 = np.asarray(eta, dtype=float) ** 2
    total = eta2.sum()
    if total <= 0.0:
        return np.array([], dtype=int)
    order = np.argsort(eta2)[::-1]
    csum = np.cumsum(eta2[order])
    nmark = int(np.searchsorted(csum, theta * total, side="left") + 1)
    return np.sort(order[:nmark])
