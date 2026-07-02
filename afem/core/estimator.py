from __future__ import annotations
import numpy as np
from .geometry import element_geometry, facet_measure, interior_facets

# Uses midpoint quadrature for volume terms
def residual_estimator(mesh, u: np.ndarray, rhs, fbar: np.ndarray | None = None) -> np.ndarray:
    """Residual estimator for P1 Poisson solution.

    eta_K^2 = h_K^2 ||f + Delta u_h||^2_K
              + 1/2 sum_{E interior, E subset dK} h_E ||[grad u_h . n]||^2_E.

    For P1 elements Delta u_h = 0 elementwise.  If fbar is supplied, it is used
    as an elementwise P0 approximation of f; otherwise f is evaluated at centroids.
    """
    vols, hs, grad_u, grad_lam = element_geometry(mesh, u)
    #ne = mesh.t.shape[1]

    if fbar is None:
        centroids = mesh.p[:, mesh.t].mean(axis=1)
        fvals = rhs(centroids)
    else:
        fvals = fbar

    eta2 = hs**2 * vols * fvals**2

    for face, (k0, opp0), (k1, opp1) in interior_facets(mesh):
        face_vertices = mesh.p[:, list(face)]
        meas = facet_measure(face_vertices)
        h_face = meas if mesh.p.shape[0] == 2 else np.sqrt(meas)

        n0 = -grad_lam[:, opp0, k0]
        n0 = n0 / np.linalg.norm(n0)
        n1 = -grad_lam[:, opp1, k1]
        n1 = n1 / np.linalg.norm(n1)

        jump = float(np.dot(grad_u[:, k0], n0) + np.dot(grad_u[:, k1], n1))
        contrib = 0.5 * h_face * meas * jump**2
        eta2[k0] += contrib
        eta2[k1] += contrib

    return np.sqrt(np.maximum(eta2, 0.0))
