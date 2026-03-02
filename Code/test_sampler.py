from triangle_sampler import sample_uniform_points_in_triangles
import numpy as np
import matplotlib.pyplot as plt
from skfem.visuals.matplotlib import draw
from skfem import MeshTri

def visualize_samples(mesh, pts, max_elems=30, s=6):
    fig, ax = plt.subplots()
    draw(mesh, ax=ax)

    ne = mesh.nelements
    # get only a subset of triangles
    #take = np.arange(min(ne, max_elems))
    #X = pts[take].reshape(-1, 2)

    X = pts.reshape(-1, 2) # get all triangles
    ax.scatter(X[:, 0], X[:, 1], s=s)
    ax.set_aspect('equal')
    ax.set_title("Uniform samples in triangles")
    plt.show()

m = MeshTri.init_sqsymmetric().refined(2)
rng = np.random.default_rng(0)

pts = sample_uniform_points_in_triangles(m, m_samples=50, rng=rng)

# 1) sanity checks
print("pts shape:", pts.shape)  # should be (nelements, m_samples, 2)

# 3) visualize
visualize_samples(m, pts, max_elems=40)