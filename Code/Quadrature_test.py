from skfem import *
from skfem.models.poisson import laplace
from skfem.helpers import grad
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# Mesh + element
# -----------------------------------------------------------
m = MeshTri.init_sqsymmetric().refined()
m = MeshTri()
e = ElementTriP1()

# -----------------------------------------------------------
# Manufactured exact solution and RHS (Dirichlet 0)
# u = sin(pi x) sin(pi y)
# f = -Δu = 2*pi^2*sin(pi x) sin(pi y)
# -----------------------------------------------------------
def u_exact(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def load_func(x, y):
    return 2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# -----------------------------------------------------------
# Deterministic quadrature load assembly
# -----------------------------------------------------------
@LinearForm
def load(v, w):
    x, y = w.x
    return load_func(x, y) * v

# -----------------------------------------------------------
# Residual estimator (deterministic RHS)
# -----------------------------------------------------------
def eval_estimator(m, u):
    basis = Basis(m, e)

    @Functional
    def interior_residual(w):
        h = w.h
        x, y = w.x
        return h**2 * load_func(x, y)**2

    eta_K = interior_residual.elemental(basis, w=basis.interpolate(u))

    fbasis = [InteriorFacetBasis(m, e, side=i) for i in [0, 1]]
    wdict = {'u' + str(i + 1): fbasis[i].interpolate(u) for i in [0, 1]}

    @Functional
    def edge_jump(w):
        h = w.h
        n = w.n
        dw1 = grad(w['u1'])
        dw2 = grad(w['u2'])
        return h * ((dw1[0] - dw2[0]) * n[0] +
                    (dw1[1] - dw2[1]) * n[1])**2

    eta_E = edge_jump.elemental(fbasis[0], **wdict)

    tmp = np.zeros(m.facets.shape[1])
    np.add.at(tmp, fbasis[0].find, eta_E)
    eta_E = np.sum(0.5 * tmp[m.t2f], axis=0)

    return eta_K + eta_E

# -----------------------------------------------------------
# Dörfler marking
# -----------------------------------------------------------
def dorfler_marking(eta2, theta=0.5):
    eta2 = np.asarray(eta2)
    total = eta2.sum()
    if total <= 0:
        return np.array([], dtype=int)

    order = np.argsort(eta2)[::-1]
    csum = np.cumsum(eta2[order])
    k = np.searchsorted(csum, theta * total) + 1
    return order[:max(1, k)]

# -----------------------------------------------------------
# Error computation against exact solution
# -----------------------------------------------------------
def compute_errors(basis, u_vec):
    uh = basis.interpolate(u_vec)

    @Functional
    def l2_err_sq(w):
        x, y = w.x
        return (w.uh - u_exact(x, y))**2

    @Functional
    def h1_semi_err_sq(w):
        x, y = w.x
        duh = grad(w.uh)
        ux = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
        uy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
        return (duh[0] - ux)**2 + (duh[1] - uy)**2

    L2 = np.sqrt(l2_err_sq.assemble(basis, uh=uh))
    H1 = np.sqrt(h1_semi_err_sq.assemble(basis, uh=uh))
    return L2, H1

# -----------------------------------------------------------
# AFEM loop (deterministic quadrature)
# -----------------------------------------------------------
if __name__ == "__main__":
    from skfem.visuals.matplotlib import draw, plot

    n_refinements = 10

    for it in range(n_refinements):
        basis = Basis(m, e)
        K = asm(laplace, basis)
        f = asm(load, basis)

        I = m.interior_nodes()
        u = solve(*condense(K, f, I=I))

        L2, H1 = compute_errors(basis, u)
        print(f"it={it+1:2d}  ndof={basis.N:6d}  L2={L2:.3e}  H1_semi={H1:.3e}")

        if it < n_refinements - 1:
            eta2 = eval_estimator(m, u)
            marked = dorfler_marking(eta2, theta=0.5)
            m = m.refined(marked).smoothed()

    # exact solution
    xv, yv = m.p[0], m.p[1]
    u_ex_vec = u_exact(xv, yv)

    vmin = min(u.min(), u_ex_vec.min())
    vmax = max(u.max(), u_ex_vec.max())

    fig1, ax1 = plt.subplots()
    draw(m, ax=ax1)
    plot(m, u, ax=ax1, shading='gouraud', colorbar=True, vmin=vmin, vmax=vmax)
    ax1.set_title("AFEM solution (quadrature)")


    fig2, ax2 = plt.subplots()
    draw(m, ax=ax2)
    plot(m, u_ex_vec, ax=ax2, shading='gouraud', colorbar=True, vmin=vmin, vmax=vmax)
    ax2.set_title("Exact solution (quadrature)")

    error_vec = u - u_ex_vec
    fig3, ax3 = plt.subplots()
    draw(m, ax=ax3)
    plot(m, error_vec, ax=ax3, shading='gouraud',colorbar=True)
    ax3.set_title("Error (quadrature)")

    plt.show()