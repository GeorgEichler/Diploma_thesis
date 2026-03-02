import numpy as np

def sample_uniform_points_in_triangles(mesh, m_samples, rng=None, return_barycentric=False):
    """
    Return samples using reflection trick on triangles
    """

    if rng is None:
        rng = np.random.default_rng()

    # triangle vertex coordinates
    p = mesh.p.T # (n_vertices, 2)
    t = mesh.t.T # (n_elements, 3)
    v0 = p[t[:, 0]]
    v1 = p[t[:, 1]]
    v2 = p[t[:, 2]]

    # reflection trick on reference triangle
    u = rng.random((t.shape[0], m_samples))
    v = rng.random((t.shape[0], m_samples))
    mask = (u + v) > 1.0
    u[mask] = 1.0 - u[mask]
    v[mask] = 1.0 - v[mask]

    # barycentric weights on reference triangle are (0,0), (1,0) and (0,1)
    w0 = 1.0 - u - v
    w1 = u
    w2 = v 

    # affine map to other triangles
    pts = (
        w0[..., None] * v0[:, None, :]
        + w1[..., None] * v1[:, None, :]
        + w2[..., None] * v2[:, None, :]
    )
    if return_barycentric:
        return pts, (w0, w1, w2)
    return pts  # (n_elements, m_samples, 2)