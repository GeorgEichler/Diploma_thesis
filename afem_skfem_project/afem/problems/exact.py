from __future__ import annotations
import numpy as np

Array = np.ndarray

def sine_2d(x: Array) -> Array:
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1])

def sine_3d(x: Array) -> Array:
    return np.sin(np.pi * x[0]) * np.sin(np.pi * x[1]) * np.sin(np.pi * x[2])
