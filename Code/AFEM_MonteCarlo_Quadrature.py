"""
AFEM comparison for

    -Delta u = |sin(pi * 2^5 * 3 * x)|   in (0, 1)^2,
          u = 0                          on boundary.

Implemented load approximations:

    1. method="midpoint"
       One-point centroid approximation:
           f_T = f(x_T)

    2. method="quadrature"
       Deterministic higher-order quadrature approximation of the cell average:
           f_T ≈ (1 / |T|) int_T f dx

    3. method="mc"
       Monte Carlo cell average:
           f_T = (1/N) sum_i f(X_i^T)

The AFEM loop is:

    SOLVE -> ESTIMATE -> MARK -> REFINE

with Dörfler marking.

"""

from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import matplotlib.pyplot as plt

from skfem import MeshTri, Basis, ElementTriP1, asm, solve, condense
from skfem.models.poisson import laplace


# ============================================================
# Problem data
# ============================================================

L_OSC = 5


def rhs_f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Right-hand side

        f(x, y) = |sin(pi * 2^5 * 3 * x)| = |sin(96 pi x)|.

    y is unused but included for a standard 2D interface.
    """
    return np.abs(np.sin(np.pi * (2**L_OSC) * 3.0 * x))


# ============================================================
# Mesh helpers
# ============================================================

def orient_triangles_positively(mesh: MeshTri) -> MeshTri:
    """
    Ensure that all triangles have positive orientation.

    This is useful after optional jittering of the initial mesh.
    """
    p = mesh.p.copy()
    t = mesh.t.copy()

    p0 = p[:, t[0]]
    p1 = p[:, t[1]]
    p2 = p[:, t[2]]

    signed_twice_area = (
        (p1[0] - p0[0]) * (p2[1] - p0[1])
        - (p1[1] - p0[1]) * (p2[0] - p0[0])
    )

    bad = signed_twice_area < 0.0
    if np.any(bad):
        t[[1, 2], bad] = t[[2, 1], bad]

    return MeshTri(p, t)


def jitter_interior_nodes(
    mesh: MeshTri,
    amount: float = 0.10,
    seed: int = 123,
) -> MeshTri:
    """
    Slightly perturb only interior nodes of the initial mesh.

    This is optional. It is useful for this very special oscillatory RHS because
    midpoint quadrature on a perfectly structured mesh can sample at many zeros
    of |sin(96 pi x)|.

    Boundary nodes are fixed, so the domain remains the unit square.
    """
    rng = np.random.default_rng(seed)

    p = mesh.p.copy()
    boundary = np.zeros(p.shape[1], dtype=bool)
    boundary[mesh.boundary_nodes()] = True
    interior = ~boundary

    xs = np.unique(np.round(p[0], 14))
    ys = np.unique(np.round(p[1], 14))

    hx = np.min(np.diff(xs)) if len(xs) > 1 else 1.0
    hy = np.min(np.diff(ys)) if len(ys) > 1 else 1.0
    h = min(hx, hy)

    perturbation = amount * h * rng.uniform(-1.0, 1.0, size=(2, interior.sum()))
    p[:, interior] += perturbation

    p[0, interior] = np.clip(p[0, interior], 1e-12, 1.0 - 1e-12)
    p[1, interior] = np.clip(p[1, interior], 1e-12, 1.0 - 1e-12)

    return orient_triangles_positively(MeshTri(p, mesh.t.copy()))


def triangle_geometry(mesh: MeshTri):
    """
    Vectorized geometry for all triangles.

    Returns
    -------
    p0, p1, p2:
        Vertex coordinates, each of shape (2, nelems).
    area:
        Triangle areas, shape (nelems,).
    hT:
        Element diameters, shape (nelems,).
    centroid:
        Triangle centroids, shape (2, nelems).
    """
    p = mesh.p
    t = mesh.t

    p0 = p[:, t[0]]
    p1 = p[:, t[1]]
    p2 = p[:, t[2]]

    e01 = p1 - p0
    e12 = p2 - p1
    e20 = p0 - p2

    signed_twice_area = e01[0] * (p2[1] - p0[1]) - e01[1] * (p2[0] - p0[0])
    area = 0.5 * np.abs(signed_twice_area)

    l01 = np.linalg.norm(e01, axis=0)
    l12 = np.linalg.norm(e12, axis=0)
    l20 = np.linalg.norm(e20, axis=0)

    hT = np.maximum.reduce([l01, l12, l20])
    centroid = (p0 + p1 + p2) / 3.0

    return p0, p1, p2, area, hT, centroid


# ============================================================
# P0 load approximations
# ============================================================

def p0_load_midpoint(mesh: MeshTri) -> np.ndarray:
    """
    One-point midpoint/centroid approximation.

    On each triangle T:

        f_T = f(centroid of T).

    This is cheap but can badly alias highly oscillatory functions.
    """
    _, _, _, _, _, centroid = triangle_geometry(mesh)
    return rhs_f(centroid[0], centroid[1])


def p0_load_mc(
    mesh: MeshTri,
    n_samples: int = 20,
    seed: int | None = None,
) -> np.ndarray:
    """
    Monte Carlo approximation of the P0 cell average.

    On every triangle T:

        f_T = 1/N sum_{i=1}^N f(X_i^T),

    where X_i^T are uniformly distributed in T.

    Sampling uses the reflection trick on the reference triangle.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    rng = np.random.default_rng(seed)

    p0, p1, p2, _, _, _ = triangle_geometry(mesh)
    nelems = mesh.t.shape[1]

    d10 = p1 - p0
    d20 = p2 - p0

    fK = np.zeros(nelems)

    for _ in range(n_samples):
        r = rng.random(nelems)
        s = rng.random(nelems)

        outside = r + s > 1.0
        r[outside] = 1.0 - r[outside]
        s[outside] = 1.0 - s[outside]

        x = p0[0] + r * d10[0] + s * d20[0]
        y = p0[1] + r * d10[1] + s * d20[1]

        fK += rhs_f(x, y)

    return fK / float(n_samples)


def p0_load_gauss_duffy(
    mesh: MeshTri,
    n_gauss: int = 12,
) -> np.ndarray:
    """
    Deterministic higher-order quadrature approximation of the P0 cell average.

    On every triangle T, approximate

        f_T = (1 / |T|) int_T f(x, y) dx dy.

    We use a tensor-product Gauss-Legendre rule on [0, 1]^2 and map it to
    the reference triangle by the Duffy transform:

        r = xi,
        s = (1 - xi) eta,

    where xi, eta in [0, 1]. The Jacobian of this transform is (1 - xi).

    The affine map from the reference triangle to T is

        F_T(r, s) = p0 + r (p1 - p0) + s (p2 - p0).

    Because |T| = |det(B)| / 2, the normalized cell average contains a factor 2:

        f_T ≈ 2 * sum_i sum_j w_i w_j (1 - xi_i)
                  f(F_T(xi_i, (1 - xi_i) eta_j)).
    """
    if n_gauss < 1:
        raise ValueError("n_gauss must be at least 1.")

    # Gauss-Legendre nodes and weights on [-1, 1].
    nodes, weights = np.polynomial.legendre.leggauss(n_gauss)

    # Transform nodes and weights to [0, 1].
    xi_nodes = 0.5 * (nodes + 1.0)
    xi_weights = 0.5 * weights

    eta_nodes = xi_nodes
    eta_weights = xi_weights

    p0, p1, p2, _, _, _ = triangle_geometry(mesh)

    d10 = p1 - p0
    d20 = p2 - p0

    nelems = mesh.t.shape[1]
    fK = np.zeros(nelems)

    # Loops over quadrature points; vectorized over all elements.
    for xi, wxi in zip(xi_nodes, xi_weights):
        one_minus_xi = 1.0 - xi

        for eta, weta in zip(eta_nodes, eta_weights):
            r = xi
            s = one_minus_xi * eta

            # Normalized cell-average weight.
            w = 2.0 * wxi * weta * one_minus_xi

            x = p0[0] + r * d10[0] + s * d20[0]
            y = p0[1] + r * d10[1] + s * d20[1]

            fK += w * rhs_f(x, y)

    return fK


def compute_load(
    mesh: MeshTri,
    method: str,
    n_mc_samples: int,
    seed: int,
    n_gauss: int,
) -> np.ndarray:
    """
    Compute one P0 load value per triangle.
    """
    if method == "midpoint":
        return p0_load_midpoint(mesh)

    if method == "quadrature":
        return p0_load_gauss_duffy(mesh, n_gauss=n_gauss)

    if method == "mc":
        return p0_load_mc(mesh, n_samples=n_mc_samples, seed=seed)

    raise ValueError("method must be 'midpoint', 'quadrature', or 'mc'.")


def assemble_p0_load_vector(mesh: MeshTri, fK: np.ndarray) -> np.ndarray:
    """
    Assemble

        b_i = int_D f_h phi_i dx,

    where f_h is P0 with values fK and phi_i are P1 nodal basis functions.

    Since f_h is constant on T,

        int_T f_T phi_i dx = f_T |T| / 3

    for each local vertex.
    """
    _, _, _, area, _, _ = triangle_geometry(mesh)

    nvertices = mesh.p.shape[1]
    b = np.zeros(nvertices)

    local = fK * area / 3.0

    np.add.at(b, mesh.t[0], local)
    np.add.at(b, mesh.t[1], local)
    np.add.at(b, mesh.t[2], local)

    return b


# ============================================================
# FEM solve
# ============================================================

def solve_poisson_p1(mesh: MeshTri, fK: np.ndarray) -> np.ndarray:
    """
    Solve the P1 FEM problem

        int_D grad u_h · grad v_h dx = int_D f_h v_h dx

    with homogeneous Dirichlet boundary conditions.
    """
    basis = Basis(mesh, ElementTriP1())

    A = asm(laplace, basis)
    b = assemble_p0_load_vector(mesh, fK)

    D = mesh.boundary_nodes()

    u = solve(*condense(A, b, D=D))
    return u


# ============================================================
# Residual estimator
# ============================================================

def p1_element_gradients(mesh: MeshTri, u: np.ndarray) -> np.ndarray:
    """
    Vectorized computation of the constant P1 gradient on every triangle.

    Returns
    -------
    grads:
        Array of shape (nelems, 2).
    """
    p = mesh.p
    t = mesh.t

    x0, y0 = p[:, t[0]]
    x1, y1 = p[:, t[1]]
    x2, y2 = p[:, t[2]]

    u0 = u[t[0]]
    u1 = u[t[1]]
    u2 = u[t[2]]

    twice_area_signed = (
        (x1 - x0) * (y2 - y0)
        - (y1 - y0) * (x2 - x0)
    )

    grad_phi0_x = (y1 - y2) / twice_area_signed
    grad_phi0_y = (x2 - x1) / twice_area_signed

    grad_phi1_x = (y2 - y0) / twice_area_signed
    grad_phi1_y = (x0 - x2) / twice_area_signed

    grad_phi2_x = (y0 - y1) / twice_area_signed
    grad_phi2_y = (x1 - x0) / twice_area_signed

    grads = np.empty((mesh.t.shape[1], 2))

    grads[:, 0] = (
        u0 * grad_phi0_x
        + u1 * grad_phi1_x
        + u2 * grad_phi2_x
    )
    grads[:, 1] = (
        u0 * grad_phi0_y
        + u1 * grad_phi1_y
        + u2 * grad_phi2_y
    )

    return grads


def residual_estimator_p1(mesh: MeshTri, u: np.ndarray, fK: np.ndarray) -> np.ndarray:
    """
    Compute local residual indicators eta_T^2.

    For P1 FEM, Delta u_h = 0 inside each triangle. Therefore the volume
    residual is just f_h.

    Estimator:

        eta_T^2 =
            h_T^2 ||f_h||_{L2(T)}^2
            + 1/2 sum_{E subset dT interior}
                  h_E ||[[grad u_h · n_E]]||_{L2(E)}^2.

    Since grad u_h is constant on each element, the edge jump is constant.
    """
    p = mesh.p

    _, _, _, area, hT, _ = triangle_geometry(mesh)

    # Volume residual:
    #
    #   h_T^2 ||f_h||^2_{L2(T)}
    #   = h_T^2 |T| f_T^2.
    eta2 = hT**2 * area * fK**2

    grads = p1_element_gradients(mesh, u)

    # scikit-fem facet connectivity:
    #
    #   mesh.facets has shape (2, nfacets)
    #   mesh.f2t has shape (2, nfacets)
    #
    # For boundary facets, mesh.f2t[1, facet] == -1.
    facets = mesh.facets
    f2t = mesh.f2t

    interior_facets = np.where(f2t[1] != -1)[0]

    if interior_facets.size == 0:
        return eta2

    elems_left = f2t[0, interior_facets]
    elems_right = f2t[1, interior_facets]

    a = facets[0, interior_facets]
    b = facets[1, interior_facets]

    pa = p[:, a]
    pb = p[:, b]

    edge_vec = pb - pa
    edge_len = np.linalg.norm(edge_vec, axis=0)

    normal = np.empty_like(edge_vec)
    normal[0] = edge_vec[1] / edge_len
    normal[1] = -edge_vec[0] / edge_len

    grad_jump = grads[elems_left] - grads[elems_right]
    jump_normal = grad_jump[:, 0] * normal[0] + grad_jump[:, 1] * normal[1]

    # h_E ||jump||^2_{L2(E)}
    # = edge_len * (edge_len * jump^2)
    # = edge_len^2 * jump^2.
    #
    # Split half to each neighboring element.
    edge_contribution = 0.5 * edge_len**2 * jump_normal**2

    np.add.at(eta2, elems_left, edge_contribution)
    np.add.at(eta2, elems_right, edge_contribution)

    return eta2


# ============================================================
# Dörfler marking
# ============================================================

def doerfler_marking(eta2: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """
    Dörfler marking.

    Choose a set M such that

        sum_{T in M} eta_T^2 >= theta sum_T eta_T^2.

    This implementation sorts eta_T^2 in descending order and selects the
    smallest prefix satisfying the criterion.
    """
    if not (0.0 < theta <= 1.0):
        raise ValueError("theta must satisfy 0 < theta <= 1.")

    total = float(np.sum(eta2))

    if total <= 0.0 or not np.isfinite(total):
        return np.array([], dtype=np.int64)

    order = np.argsort(eta2)[::-1]
    cumsum = np.cumsum(eta2[order])

    nmarked = np.searchsorted(cumsum, theta * total) + 1

    return order[:nmarked]


# ============================================================
# AFEM loop
# ============================================================

@dataclass
class AFEMHistory:
    method: str
    ndof: list[int]
    nelems: list[int]
    estimator: list[float]
    nmarked: list[int]
    solve_time: list[float]
    estimator_time: list[float]
    refine_time: list[float]


def make_initial_mesh(
    initial_refinements: int = 3,
    jitter: bool = False,
    jitter_amount: float = 0.10,
    seed: int = 123,
    domain: str = "square",
) -> MeshTri:
    """
    Create the initial mesh.

    domain="square":
        unit square (0,1)^2.

    domain="lshape":
        built-in scikit-fem L-shaped domain.
    """
    if domain == "square":
        mesh = MeshTri().refined(initial_refinements)

        if jitter:
            mesh = jitter_interior_nodes(mesh, amount=jitter_amount, seed=seed)

        return mesh

    if domain == "lshape":
        # Built-in L-shaped domain from scikit-fem.
        mesh = MeshTri.init_lshaped().refined(initial_refinements)
        return mesh

    raise ValueError("domain must be either 'square' or 'lshape'.")


def run_afem(
    method: str,
    max_iter: int = 20,
    theta: float = 0.5,
    n_mc_samples: int = 20,
    n_gauss: int = 12,
    seed: int = 12345,
    initial_refinements: int = 3,
    jitter_initial_mesh: bool = False,
    jitter_amount: float = 0.10,
    max_ndof: int | None = None,
    verbose: bool = True,
    domain: str = "square",
) -> tuple[AFEMHistory, MeshTri, np.ndarray]:
    """
    Run the AFEM loop for one load approximation method.

    Parameters
    ----------
    method:
        "midpoint", "quadrature", or "mc".
    max_iter:
        Maximum number of adaptive iterations.
    theta:
        Dörfler marking parameter.
    n_mc_samples:
        Number of Monte Carlo samples per element for method="mc".
    n_gauss:
        Number of Gauss points per coordinate direction for method="quadrature".
        The total number of function evaluations per element is n_gauss^2.
    seed:
        Random seed.
    initial_refinements:
        Number of initial uniform refinements.
    jitter_initial_mesh:
        Whether to perturb interior nodes of the initial mesh.
    max_ndof:
        Optional stopping criterion.
    verbose:
        Print iteration data if True.
    """
    mesh = make_initial_mesh(
        initial_refinements=initial_refinements,
        jitter=jitter_initial_mesh,
        jitter_amount=jitter_amount,
        seed=seed,
        domain=domain,
    )

    hist = AFEMHistory(
        method=method,
        ndof=[],
        nelems=[],
        estimator=[],
        nmarked=[],
        solve_time=[],
        estimator_time=[],
        refine_time=[],
    )

    u = np.zeros(mesh.p.shape[1])

    for it in range(max_iter):
        # ----------------------------
        # SOLVE
        # ----------------------------
        t0 = time.perf_counter()

        fK = compute_load(
            mesh,
            method=method,
            n_mc_samples=n_mc_samples,
            seed=seed + 1000 * it,
            n_gauss=n_gauss,
        )

        u = solve_poisson_p1(mesh, fK)

        solve_t = time.perf_counter() - t0

        # ----------------------------
        # ESTIMATE
        # ----------------------------
        t0 = time.perf_counter()

        eta2 = residual_estimator_p1(mesh, u, fK)
        eta = float(np.sqrt(np.sum(eta2)))

        estimator_t = time.perf_counter() - t0

        # ----------------------------
        # MARK
        # ----------------------------
        marked = doerfler_marking(eta2, theta=theta)

        hist.ndof.append(mesh.p.shape[1])
        hist.nelems.append(mesh.t.shape[1])
        hist.estimator.append(eta)
        hist.nmarked.append(marked.size)
        hist.solve_time.append(solve_t)
        hist.estimator_time.append(estimator_t)

        if verbose:
            print(
                f"{method:>10s} | it={it:02d} | "
                f"ndof={hist.ndof[-1]:7d} | "
                f"nelems={hist.nelems[-1]:7d} | "
                f"marked={marked.size:6d} | "
                f"eta={eta:.6e} | "
                f"solve={solve_t:.3f}s | "
                f"estimate={estimator_t:.3f}s"
            )

        if marked.size == 0:
            if verbose:
                print("No elements marked. Stopping.")
            hist.refine_time.append(0.0)
            break

        if max_ndof is not None and mesh.p.shape[1] >= max_ndof:
            if verbose:
                print("Reached max_ndof. Stopping.")
            hist.refine_time.append(0.0)
            break

        if it == max_iter - 1:
            hist.refine_time.append(0.0)
            break

        # ----------------------------
        # REFINE
        # ----------------------------
        t0 = time.perf_counter()

        mesh = mesh.refined(marked)

        refine_t = time.perf_counter() - t0
        hist.refine_time.append(refine_t)

    return hist, mesh, u


# ============================================================
# Plotting
# ============================================================

def add_reference_slope(
    ndof: list[int],
    y_anchor: float,
    exponent: float,
    label: str,
):
    """
    Add a reference line C * ndof^exponent to the current log-log plot.
    """
    ndof_arr = np.asarray(ndof, dtype=float)
    slope = y_anchor * (ndof_arr / ndof_arr[-1]) ** exponent
    plt.loglog(ndof_arr, slope, "k--", linewidth=1.2, label=label)


def plot_afem_histories_three(
    hist_mid: AFEMHistory,
    hist_quad: AFEMHistory,
    hist_mc: AFEMHistory,
    n_mc_samples: int,
    n_gauss: int,
    filename: str = "afem_three_methods_estimator.png",
):
    """
    Plot estimator histories for midpoint, deterministic quadrature, and MC.
    """
    plt.figure(figsize=(8.0, 5.6))

    plt.loglog(
        hist_mid.ndof,
        hist_mid.estimator,
        "o-",
        label=r"midpoint $\Pi_0 f$",
    )

    plt.loglog(
        hist_quad.ndof,
        hist_quad.estimator,
        "^-",
        label=rf"deterministic quadrature $\Pi_0 f$, $n_q={n_gauss}$",
    )

    plt.loglog(
        hist_mc.ndof,
        hist_mc.estimator,
        "s-",
        label=rf"Monte Carlo $\widehat{{\Pi}}_0 f$, $N={n_mc_samples}$",
    )

    add_reference_slope(
        hist_mc.ndof,
        hist_mc.estimator[-1],
        -0.5,
        r"$\mathrm{ndof}^{-1/2}$",
    )

    plt.xlabel("degrees of freedom")
    plt.ylabel(r"total estimator $\eta_\ell$")
    plt.title("AFEM: midpoint vs deterministic quadrature vs Monte Carlo")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


def plot_marked_elements(
    hist_mid: AFEMHistory,
    hist_quad: AFEMHistory,
    hist_mc: AFEMHistory,
    filename: str = "afem_marked_elements.png",
):
    """
    Plot number of marked elements per adaptive step.
    """
    plt.figure(figsize=(8.0, 5.2))

    plt.semilogy(hist_mid.nmarked, "o-", label="midpoint")
    plt.semilogy(hist_quad.nmarked, "^-", label="deterministic quadrature")
    plt.semilogy(hist_mc.nmarked, "s-", label="Monte Carlo")

    plt.xlabel("adaptive iteration")
    plt.ylabel("number of marked elements")
    plt.title("Dörfler marking: number of marked elements")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


def plot_runtime_breakdown(
    hist_mid: AFEMHistory,
    hist_quad: AFEMHistory,
    hist_mc: AFEMHistory,
    filename: str = "afem_runtime_breakdown.png",
):
    """
    Plot cumulative solve + estimator time for each method.

    This is useful because high-order quadrature can be more expensive than MC.
    """
    plt.figure(figsize=(8.0, 5.2))

    def cumulative_total_time(hist: AFEMHistory):
        return np.cumsum(
            np.asarray(hist.solve_time)
            + np.asarray(hist.estimator_time)
            + np.asarray(hist.refine_time[: len(hist.solve_time)])
        )

    plt.plot(cumulative_total_time(hist_mid), "o-", label="midpoint")
    plt.plot(cumulative_total_time(hist_quad), "^-", label="deterministic quadrature")
    plt.plot(cumulative_total_time(hist_mc), "s-", label="Monte Carlo")

    plt.xlabel("adaptive iteration")
    plt.ylabel("cumulative time [s]")
    plt.title("Runtime comparison")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


def plot_mesh(mesh: MeshTri, title: str = "Final adaptive mesh"):
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    mesh.draw(ax=ax)
    ax.set_aspect("equal")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


# ============================================================
# Main experiment
# ============================================================

if __name__ == "__main__":
    THETA = 0.5
    MAX_ITER = 20

    # Monte Carlo samples per element.
    N_MC = 20

    # Higher-order deterministic quadrature parameter.
    # Total function evaluations per element are N_GAUSS^2.
    #
    # Reasonable values to test:
    #   4, 8, 12, 16, 24, 32
    N_GAUSS = 4

    SEED = 12345

    INITIAL_REFINEMENTS = 3

    # Set this to False if you want to clearly expose the midpoint aliasing
    # on the structured initial mesh.
    #
    # Set this to True if you want to reduce that special-grid artifact.
    JITTER_INITIAL_MESH = False
    JITTER_AMOUNT = 0.10

    MAX_NDOF = 200_000
    DOMAIN = 'lshape'

    print("\nRunning AFEM with midpoint P0 load\n")
    hist_mid, mesh_mid, u_mid = run_afem(
        method="midpoint",
        max_iter=MAX_ITER,
        theta=THETA,
        n_mc_samples=N_MC,
        n_gauss=N_GAUSS,
        seed=SEED,
        initial_refinements=INITIAL_REFINEMENTS,
        jitter_initial_mesh=JITTER_INITIAL_MESH,
        jitter_amount=JITTER_AMOUNT,
        max_ndof=MAX_NDOF,
        verbose=True,
        domain=DOMAIN,
    )

    print("\nRunning AFEM with deterministic higher-order quadrature P0 load\n")
    hist_quad, mesh_quad, u_quad = run_afem(
        method="quadrature",
        max_iter=MAX_ITER,
        theta=THETA,
        n_mc_samples=N_MC,
        n_gauss=N_GAUSS,
        seed=SEED,
        initial_refinements=INITIAL_REFINEMENTS,
        jitter_initial_mesh=JITTER_INITIAL_MESH,
        jitter_amount=JITTER_AMOUNT,
        max_ndof=MAX_NDOF,
        verbose=True,
    )

    print("\nRunning AFEM with Monte Carlo P0 load\n")
    hist_mc, mesh_mc, u_mc = run_afem(
        method="mc",
        max_iter=MAX_ITER,
        theta=THETA,
        n_mc_samples=N_MC,
        n_gauss=N_GAUSS,
        seed=SEED,
        initial_refinements=INITIAL_REFINEMENTS,
        jitter_initial_mesh=JITTER_INITIAL_MESH,
        jitter_amount=JITTER_AMOUNT,
        max_ndof=MAX_NDOF,
        verbose=True,
    )

    plot_afem_histories_three(
        hist_mid,
        hist_quad,
        hist_mc,
        n_mc_samples=N_MC,
        n_gauss=N_GAUSS,
        filename="afem_three_methods_estimator.png",
    )

    plot_marked_elements(
        hist_mid,
        hist_quad,
        hist_mc,
        filename="afem_marked_elements.png",
    )

    plot_runtime_breakdown(
        hist_mid,
        hist_quad,
        hist_mc,
        filename="afem_runtime_breakdown.png",
    )

    plot_mesh(mesh_mid, title="Final adaptive mesh: midpoint load")
    #plot_mesh(mesh_quad, title="Final adaptive mesh: deterministic quadrature load")
    #plot_mesh(mesh_mc, title="Final adaptive mesh: Monte Carlo load")