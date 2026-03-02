from skfem import *
from skfem.models.poisson import laplace # bilinear form for \int_\Omega \nabla u \nabla v dx
from skfem.helpers import grad 
from triangle_sampler import sample_uniform_points_in_triangles
import numpy as np



# Create triangular mesh of [0,1]^2 and do two uniform refinements
m = MeshTri.init_sqsymmetric().refined(2)
# use P1 Lagrange elements (continuous, piecewise linear)
e = ElementTriP1()

#Define RHS
def load_func(x, y):
    return 1.0

def triangle_areas(mesh):
    p = mesh.p.T
    t = mesh.t.T
    a = p[t[:, 0]]
    b = p[t[:, 1]]
    c = p[t[:, 2]]
    # area = 0.5 * |det(b-a, c-a)|
    return 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) -
                        (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

def assemble_load_mc_P1(mesh, load_func, m_samples=10, rng=None):
    """
    Monte Carlo assembly of F_i = ∫ f φ_i dx for P1 on triangles.
    Returns global load vector F of length n_vertices.
    """
    if rng is None:
        rng = np.random.default_rng()

    pts, (w0, w1, w2) = sample_uniform_points_in_triangles(
        mesh, m_samples, rng=rng, return_barycentric=True
    )
    x = pts[..., 0]
    y = pts[..., 1]

    vals = load_func(x, y)
    vals = np.asarray(vals)
    if vals.ndim == 0:
        vals = np.full_like(x, float(vals), dtype=float)  # handle constant f

    #ne = mesh.nelements
    areas = triangle_areas(mesh)  # (ne,)

    # local contributions for each triangle vertex
    scale = areas / m_samples
    b0 = scale * np.sum(vals * w0, axis=1)   # (ne,)
    b1 = scale * np.sum(vals * w1, axis=1)
    b2 = scale * np.sum(vals * w2, axis=1)

    # scatter-add into global vector at vertex indices
    F = np.zeros(mesh.p.shape[1], dtype=float)
    t = mesh.t  # (3, ne) vertex indices

    np.add.at(F, t[0], b0)
    np.add.at(F, t[1], b1)
    np.add.at(F, t[2], b2)

    return F


# Residual estimator with input mesh m and approximation u
def eval_estimator(m, u):
    # interior residual
    # Basis elements contain the mesh and the discretization space
    basis = Basis(m, e)

    # implement residual
    @Functional
    def interior_residual(w):
        h = w.h
        x, y = w.x
        return h**2 * load_func(x, y)**2
    
    # return residual for each element depending on basis and the interpolation of u
    eta_K = interior_residual.elemental(basis, w = basis.interpolate(u))

    # get the solution values of bith sides of a triangle from each side (may jump)
    fbasis = [InteriorFacetBasis(m, e, side=i) for i in [0,1]]
    w = {'u' + str(i + 1): fbasis[i].interpolate(u) for i in [0, 1]}

    # Compute jump term  η_e^2​≈h_e​∫_e​((∇u_h​∣K1​​−∇u_h​∣K2​​)⋅n)^2 ds
    @Functional
    def edge_jump(w):
        h = w.h
        n = w.n
        dw1 = grad(w['u1'])
        dw2 = grad(w['u2'])
        return h * ((dw1[0] - dw2[0]) * n[0] +
                    (dw1[1] - dw2[1]) * n[1])**2

    # get the values per interior facet
    eta_E = edge_jump.elemental(fbasis[0], **w)

    # as each interior facet belongs to two elements, add half its conribution to each adjacent element
    tmp = np.zeros(m.facets.shape[1])
    np.add.at(tmp, fbasis[0].find, eta_E)
    eta_E = np.sum(0.5 * tmp[m.t2f], axis=0)

    return eta_K + eta_E

"""
Implement the Dörfler marking strategy
"""
def dorfler_marking(eta2, theta = 0.5):
    """
    Returns element indices to refine
    Assumes the squared residual error
    """
    eta2 = np.asarray(eta2)
    total = eta2.sum()

    if total <= 0:
        return np.array([], dtype = int)
    
    order = np.argsort(eta2)[::-1] # sort elements by decreasing error value
    csum = np.cumsum(eta2[order]) # compute the sum cumulative sums
    target = theta * total

    k = np.searchsorted(csum, target) + 1 # find first index where csum is larger than target
    marked = order[:max(1, k)] # take first k element indices for the marking
    return marked

if __name__ == "__main__":
    from skfem.visuals.matplotlib import draw, plot
    n_refinements = 10
    m_samples = 10
    rng = np.random.default_rng(0)

    # run 6 adaptive steps
    for itr in reversed(range(n_refinements)):
        # build FE space on current mesh
        basis = Basis(m, e)

        # assemble stiffeness matrix and load vector
        K = asm(laplace, basis)
        f = assemble_load_mc_P1(m, load_func, m_samples=m_samples, rng=rng)

        # Homogeneous Dirichlet u=0 on boundary:
        # solve for interior nodes only
        I = m.interior_nodes() # return DOFs not on the boundary
        u = solve(*condense(K, f, I=I)) # condense forms reduced linear system only on interior nodes

        # refine
        if itr > 0:
            # eval estimator computes errors/residuals for each element
            # adaptive theta computes the maximum error \eta_max
            # by default all elements with \eta_K > 0.5* \eta_max will be marked
            # marked elements will then be refined
            # smoothening improves mesh (e.g. to get shape regularity) by moving vertices of the triangles
            
            eta2 = eval_estimator(m, u)
            marked = dorfler_marking(eta2, theta=0.5)
            m = m.refined(marked).smoothed()
            #m = m.refined(adaptive_theta(eval_estimator(m, u))).smoothed()

    # visualize final mesh + solution
    ax = draw(m)
    plot(m, u, ax=ax, shading='gouraud', colorbar=True).show()