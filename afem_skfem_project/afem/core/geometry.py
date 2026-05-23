from __future__ import annotations
import itertools
import numpy as np


def element_geometry(mesh, u: np.ndarray | None = None):
    """Element volumes, diameters, P1 gradients, barycentric gradients.

    If u is None, gradients of u are returned as None.
    """
    p, t = mesh.p, mesh.t
    dim, ne = p.shape[0], t.shape[1]
    vols = np.empty(ne)
    hs = np.empty(ne)
    grad_u = None if u is None else np.empty((dim, ne))
    grad_lam = np.empty((dim, dim + 1, ne))
    denom = 2.0 if dim == 2 else 6.0

    for k in range(ne):
        ids = t[:, k]
        verts = p[:, ids]
        B = verts[:, 1:] - verts[:, [0]]
        vols[k] = abs(np.linalg.det(B)) / denom
        hs[k] = max(np.linalg.norm(verts[:, i] - verts[:, j])
                    for i, j in itertools.combinations(range(dim + 1), 2))
        M = np.vstack([np.ones(dim + 1), verts]).T
        coeffs = np.linalg.solve(M, np.eye(dim + 1))
        grad_lam[:, :, k] = coeffs[1:, :]
        if u is not None:
            c = np.linalg.solve(M, u[ids])
            grad_u[:, k] = c[1:]
    return vols, hs, grad_u, grad_lam


def facet_measure(vertices: np.ndarray) -> float:
    """Length in 2D or area in 3D of a simplex facet."""
    dim = vertices.shape[0]
    if dim == 2:
        return float(np.linalg.norm(vertices[:, 1] - vertices[:, 0]))
    if dim == 3:
        a = vertices[:, 1] - vertices[:, 0]
        b = vertices[:, 2] - vertices[:, 0]
        return 0.5 * float(np.linalg.norm(np.cross(a, b)))
    raise ValueError("Only dim=2 or dim=3 supported.")


def interior_facets(mesh):
    """Build interior facet adjacency independent of scikit-fem internals.

    Yields tuples: (face_vertices_global, (elem0, local_opposite0), (elem1, local_opposite1)).
    """
    t = mesh.t
    nloc, ne = t.shape
    faces: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for k in range(ne):
        for opposite in range(nloc):
            face = tuple(sorted(np.delete(t[:, k], opposite).tolist()))
            faces.setdefault(face, []).append((k, opposite))
    for face, owners in faces.items():
        if len(owners) == 2:
            yield face, owners[0], owners[1]
