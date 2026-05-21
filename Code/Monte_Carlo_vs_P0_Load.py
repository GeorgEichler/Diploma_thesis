"""
AFEM comparison for

    -Delta u = |sin(pi * 2^5 * 3 * x)|   in Omega,
          u = 0                          on boundary.

Implemented load approximations:

    1. method="midpoint"
       P0 load with centroid value:
           f_T = f(x_T)

    2. method="mc"
       Monte Carlo P0 load:
           f_T = (1/N) sum_i f(X_i^T)

No high-order deterministic quadrature is included.

Key idea for error computation:

    For each method, we run AFEM for more iterations than we want to plot.
    The final mesh is a refinement of all previous meshes from the same run.

    Therefore, we can prolong all previous P1 solutions to the final mesh and
    compute errors there directly, without element_finder and without chunking.

Requirements:

    pip install scikit-fem numpy matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass, replace
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

    The function does not depend on y.
    """
    return np.abs(np.sin(np.pi * (2**L_OSC) * 3.0 * x))


# ============================================================
# Mesh helpers
# ============================================================

def orient_triangles_positively(mesh: MeshTri) -> MeshTri:
    """
    Ensure all triangles have positive orientation.
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
    Slightly perturb interior nodes of the initial mesh.

    Optional. It can reduce midpoint aliasing for the oscillatory right-hand side.
    Boundary nodes are fixed.
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


def make_initial_mesh(
    initial_refinements: int = 3,
    jitter: bool = False,
    jitter_amount: float = 0.10,
    seed: int = 123,
    domain: str = "square",
) -> MeshTri:
    """
    Create initial mesh.

    domain="square":
        unit square.

    domain="lshape":
        scikit-fem built-in L-shaped domain.
    """
    if domain == "square":
        mesh = MeshTri().refined(initial_refinements)
    elif domain == "lshape":
        mesh = MeshTri.init_lshaped().refined(initial_refinements)
    else:
        raise ValueError("domain must be 'square' or 'lshape'.")

    if jitter:
        mesh = jitter_interior_nodes(mesh, amount=jitter_amount, seed=seed)

    return mesh


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
    One-point centroid approximation.

    On each triangle T:

        f_T = f(centroid of T).
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

        f_T = 1/N sum_i f(X_i^T),

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


def compute_load(
    mesh: MeshTri,
    method: str,
    n_mc_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Compute one P0 load value per triangle.
    """
    if method == "midpoint":
        return p0_load_midpoint(mesh)

    if method == "mc":
        return p0_load_mc(mesh, n_samples=n_mc_samples, seed=seed)

    raise ValueError("method must be 'midpoint' or 'mc'.")


def assemble_p0_load_vector(mesh: MeshTri, fK: np.ndarray) -> np.ndarray:
    """
    Assemble

        b_i = int_D f_h phi_i dx,

    where f_h is P0 with values fK and phi_i are P1 basis functions.

    Since f_h is constant on T,

        int_T f_T phi_i dx = f_T |T| / 3.
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
# P1 gradients and residual estimator
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

    For P1 FEM, Delta u_h = 0 inside each triangle.

    Estimator:

        eta_T^2 =
            h_T^2 ||f_h||_{L2(T)}^2
            + 1/2 sum_{E subset dT interior}
                  h_E ||[[grad u_h · n_E]]||_{L2(E)}^2.
    """
    p = mesh.p

    _, _, _, area, hT, _ = triangle_geometry(mesh)

    eta2 = hT**2 * area * fK**2

    grads = p1_element_gradients(mesh, u)

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

    Choose M such that

        sum_{T in M} eta_T^2 >= theta sum_T eta_T^2.
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
# Prolongation to refined mesh
# ============================================================

def coordinate_key(point: np.ndarray, decimals: int = 13) -> tuple[float, float]:
    """
    Coordinate key for matching old vertices and edge midpoints.

    Rounding is used to avoid tiny floating point mismatch.
    """
    return (round(float(point[0]), decimals), round(float(point[1]), decimals))


def prolong_p1_to_refined_mesh(
    mesh_old: MeshTri,
    u_old: np.ndarray,
    mesh_new: MeshTri,
    decimals: int = 13,
) -> np.ndarray:
    """
    Prolong a P1 function from mesh_old to its refined mesh mesh_new.

    Assumption:
        mesh_new was created by refining mesh_old.

    Since the space is P1, values at old vertices are copied and values at
    new edge-midpoint vertices are averages of endpoint values.

    No element_finder is used.
    """
    p_old = mesh_old.p
    p_new = mesh_new.p

    u_new = np.empty(p_new.shape[1], dtype=float)

    # Map old vertex coordinates to old vertex indices.
    old_vertex = {
        coordinate_key(p_old[:, i], decimals): i
        for i in range(p_old.shape[1])
    }

    # Map old edge-midpoint coordinates to endpoint pairs.
    midpoint_to_edge: dict[tuple[float, float], tuple[int, int]] = {}

    for e in range(mesh_old.facets.shape[1]):
        a = int(mesh_old.facets[0, e])
        b = int(mesh_old.facets[1, e])

        midpoint = 0.5 * (p_old[:, a] + p_old[:, b])
        key = coordinate_key(midpoint, decimals)

        midpoint_to_edge[key] = (a, b)

    missing = []

    for i in range(p_new.shape[1]):
        key = coordinate_key(p_new[:, i], decimals)

        if key in old_vertex:
            u_new[i] = u_old[old_vertex[key]]
        elif key in midpoint_to_edge:
            a, b = midpoint_to_edge[key]
            u_new[i] = 0.5 * (u_old[a] + u_old[b])
        else:
            missing.append(i)

    if missing:
        raise RuntimeError(
            f"Could not prolong {len(missing)} new vertices. "
            "This usually means mesh_new is not a direct refinement of mesh_old, "
            "or coordinate rounding tolerance is too strict. "
            "Try lowering decimals, e.g. decimals=12."
        )

    return u_new


# ============================================================
# Exact same-mesh L2 and H1 errors
# ============================================================

def l2_norm_sq_p1(mesh: MeshTri, v: np.ndarray) -> float:
    """
    Exact L2 norm squared of a P1 function on a triangular mesh.

    Uses the local P1 mass matrix

        |T| / 12 * [[2,1,1],[1,2,1],[1,1,2]].
    """
    _, _, _, area, _, _ = triangle_geometry(mesh)

    t = mesh.t
    v0 = v[t[0]]
    v1 = v[t[1]]
    v2 = v[t[2]]

    local = area / 6.0 * (
        v0**2 + v1**2 + v2**2
        + v0 * v1 + v1 * v2 + v2 * v0
    )

    return float(np.sum(local))


def h1_seminorm_sq_p1(mesh: MeshTri, v: np.ndarray) -> float:
    """
    Exact H1 seminorm squared of a P1 function on a triangular mesh.
    """
    _, _, _, area, _, _ = triangle_geometry(mesh)

    grads = p1_element_gradients(mesh, v)
    local = area * np.sum(grads**2, axis=1)

    return float(np.sum(local))


def compute_errors_on_final_mesh(
    mesh_final: MeshTri,
    solutions_on_final_mesh: list[np.ndarray],
) -> tuple[list[float], list[float]]:
    """
    Compute relative L2 and H1-seminorm errors against the final solution.

    All solutions must already be represented on mesh_final.
    """
    u_ref = solutions_on_final_mesh[-1]

    l2_ref_sq = l2_norm_sq_p1(mesh_final, u_ref)
    h1_ref_sq = h1_seminorm_sq_p1(mesh_final, u_ref)

    rel_l2 = []
    rel_h1 = []

    for u in solutions_on_final_mesh:
        e = u_ref - u

        l2_e_sq = l2_norm_sq_p1(mesh_final, e)
        h1_e_sq = h1_seminorm_sq_p1(mesh_final, e)

        rel_l2.append(float(np.sqrt(l2_e_sq / l2_ref_sq)))
        rel_h1.append(float(np.sqrt(h1_e_sq / h1_ref_sq)))

    return rel_l2, rel_h1


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
    rel_l2_error: list[float]
    rel_h1_error: list[float]


def run_afem_nested_reference(
    method: str,
    max_iter: int = 20,
    theta: float = 0.5,
    n_mc_samples: int = 20,
    seed: int = 12345,
    initial_refinements: int = 3,
    jitter_initial_mesh: bool = False,
    jitter_amount: float = 0.10,
    max_ndof: int | None = None,
    verbose: bool = True,
    domain: str = "square",
) -> tuple[AFEMHistory, MeshTri, np.ndarray, list[np.ndarray]]:
    """
    Run one long AFEM computation.

    The final iterate is used as reference. All earlier solutions are prolonged
    along the same refinement sequence to the final mesh.

    This avoids element_finder and guarantees that Monte Carlo reference and
    plotted iterations use the same seeds and same refinement path.
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
        rel_l2_error=[],
        rel_h1_error=[],
    )

    # At every stage, these arrays are represented on the current mesh.
    stored_solutions_on_current_mesh: list[np.ndarray] = []

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

        # Store current solution on current mesh.
        stored_solutions_on_current_mesh.append(u.copy())

        if verbose:
            print(
                f"{method:>10s} | it={it:02d} | "
                f"ndof={hist.ndof[-1]:8d} | "
                f"nelems={hist.nelems[-1]:8d} | "
                f"marked={marked.size:8d} | "
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
        # REFINE and PROLONG stored solutions
        # ----------------------------
        t0 = time.perf_counter()

        mesh_new = mesh.refined(marked)

        stored_solutions_on_new_mesh = []
        for u_stored in stored_solutions_on_current_mesh:
            stored_solutions_on_new_mesh.append(
                prolong_p1_to_refined_mesh(mesh, u_stored, mesh_new)
            )

        mesh = mesh_new
        stored_solutions_on_current_mesh = stored_solutions_on_new_mesh

        refine_t = time.perf_counter() - t0
        hist.refine_time.append(refine_t)

    # Now all stored solutions are represented on the final mesh.
    mesh_final = mesh
    solutions_on_final_mesh = stored_solutions_on_current_mesh
    u_final = solutions_on_final_mesh[-1]

    # Compute relative errors against final solution.
    rel_l2, rel_h1 = compute_errors_on_final_mesh(
        mesh_final,
        solutions_on_final_mesh,
    )

    hist.rel_l2_error = rel_l2
    hist.rel_h1_error = rel_h1

    if verbose:
        print(
            f"\nFinished method={method}:\n"
            f"  final ndof   = {mesh_final.p.shape[1]}\n"
            f"  final elems  = {mesh_final.t.shape[1]}\n"
            f"  final eta    = {hist.estimator[-1]:.6e}\n"
            f"  final L2 err = {hist.rel_l2_error[-1]:.6e}\n"
            f"  final H1 err = {hist.rel_h1_error[-1]:.6e}\n"
        )

    return hist, mesh_final, u_final, solutions_on_final_mesh


def truncate_history(hist: AFEMHistory, n: int) -> AFEMHistory:
    """
    Take the first n entries of a history.

    This lets us run more iterations for the reference but only plot the
    first MAIN_ITER iterations as the main convergence history.
    """
    return AFEMHistory(
        method=hist.method,
        ndof=hist.ndof[:n],
        nelems=hist.nelems[:n],
        estimator=hist.estimator[:n],
        nmarked=hist.nmarked[:n],
        solve_time=hist.solve_time[:n],
        estimator_time=hist.estimator_time[:n],
        refine_time=hist.refine_time[:n],
        rel_l2_error=hist.rel_l2_error[:n],
        rel_h1_error=hist.rel_h1_error[:n],
    )


# ============================================================
# Plotting
# ============================================================

def positive_arrays(x: list[float], y: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove zero/non-finite values for log-log plotting.
    """
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya) & (xa > 0.0) & (ya > 0.0)
    return xa[mask], ya[mask]


def add_reference_slope(
    ndof: list[int],
    y_values: list[float],
    exponent: float,
    label: str,
):
    """
    Add C * ndof^exponent using the last positive y-value as anchor.
    """
    x, y = positive_arrays(ndof, y_values)

    if len(x) == 0:
        return

    slope = y[-1] * (x / x[-1]) ** exponent
    plt.loglog(x, slope, "k--", linewidth=1.2, label=label)


def plot_estimator_histories(
    hist_mid: AFEMHistory,
    hist_mc: AFEMHistory,
    n_mc_samples: int,
    filename: str = "afem_estimator_midpoint_vs_mc.png",
):
    plt.figure(figsize=(8.0, 5.6))

    x, y = positive_arrays(hist_mid.ndof, hist_mid.estimator)
    plt.loglog(x, y, "o-", label=r"midpoint $\Pi_0 f$")

    x, y = positive_arrays(hist_mc.ndof, hist_mc.estimator)
    plt.loglog(x, y, "s-", label=rf"Monte Carlo $\widehat{{\Pi}}_0 f$, $N={n_mc_samples}$")

    add_reference_slope(
        hist_mc.ndof,
        hist_mc.estimator,
        -0.5,
        r"$\mathrm{ndof}^{-1/2}$",
    )

    plt.xlabel("degrees of freedom")
    plt.ylabel(r"total estimator $\eta_\ell$")
    plt.title("AFEM estimator: midpoint vs Monte Carlo")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


def plot_l2_h1_errors(
    hist_mid: AFEMHistory,
    hist_mc: AFEMHistory,
    filename: str = "afem_l2_h1_errors_midpoint_vs_mc.png",
):
    plt.figure(figsize=(8.4, 5.8))

    x, y = positive_arrays(hist_mid.ndof, hist_mid.rel_h1_error)
    plt.loglog(x, y, "o-", label=r"midpoint, $H^1$")

    x, y = positive_arrays(hist_mid.ndof, hist_mid.rel_l2_error)
    plt.loglog(x, y, "o:", label=r"midpoint, $L^2$")

    x, y = positive_arrays(hist_mc.ndof, hist_mc.rel_h1_error)
    plt.loglog(x, y, "s-", label=r"Monte Carlo, $H^1$")

    x, y = positive_arrays(hist_mc.ndof, hist_mc.rel_l2_error)
    plt.loglog(x, y, "s:", label=r"Monte Carlo, $L^2$")

    add_reference_slope(
        hist_mc.ndof,
        hist_mc.rel_h1_error,
        -0.5,
        r"$\mathrm{ndof}^{-1/2}$",
    )

    add_reference_slope(
        hist_mc.ndof,
        hist_mc.rel_l2_error,
        -1.0,
        r"$\mathrm{ndof}^{-1}$",
    )

    plt.xlabel("degrees of freedom")
    plt.ylabel("relative error")
    plt.title(r"AFEM errors against final nested reference solution")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


def plot_runtime_breakdown(
    hist_mid: AFEMHistory,
    hist_mc: AFEMHistory,
    filename: str = "afem_runtime_midpoint_vs_mc.png",
):
    plt.figure(figsize=(8.0, 5.2))

    def cumulative_total_time(hist: AFEMHistory):
        n = len(hist.solve_time)
        refine = np.asarray(hist.refine_time[:n])
        if len(refine) < n:
            refine = np.pad(refine, (0, n - len(refine)))

        return np.cumsum(
            np.asarray(hist.solve_time)
            + np.asarray(hist.estimator_time)
            + refine
        )

    plt.plot(cumulative_total_time(hist_mid), "o-", label="midpoint")
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
    # --------------------------------------------------------
    # Domain and AFEM parameters
    # --------------------------------------------------------
    DOMAIN = "square"      # "square" or "lshape"

    THETA = 0.5

    # We run REF_ITER iterations, use the final one as reference,
    # and plot only the first MAIN_ITER entries.
    MAIN_ITER = 10
    REF_ITER = 12

    # Optional stopping criterion. The final reference may stop earlier if this is hit.
    MAX_NDOF = 300_000

    # Monte Carlo samples.
    #
    # Important:
    # The reference and the plotted MC iterations use the same N_MC and the same seed,
    # because they are from one single long run.
    N_MC = 20

    SEED = 12345

    INITIAL_REFINEMENTS = 3

    JITTER_INITIAL_MESH = False
    JITTER_AMOUNT = 0.10

    # --------------------------------------------------------
    # Run midpoint long computation
    # --------------------------------------------------------
    print("\nRunning long AFEM with midpoint P0 load\n")

    hist_mid_full, mesh_mid_final, u_mid_ref, _ = run_afem_nested_reference(
        method="midpoint",
        max_iter=REF_ITER,
        theta=THETA,
        n_mc_samples=N_MC,
        seed=SEED,
        initial_refinements=INITIAL_REFINEMENTS,
        jitter_initial_mesh=JITTER_INITIAL_MESH,
        jitter_amount=JITTER_AMOUNT,
        max_ndof=MAX_NDOF,
        verbose=True,
        domain=DOMAIN,
    )

    # --------------------------------------------------------
    # Run Monte Carlo long computation
    # --------------------------------------------------------
    print("\nRunning long AFEM with Monte Carlo P0 load\n")

    hist_mc_full, mesh_mc_final, u_mc_ref, _ = run_afem_nested_reference(
        method="mc",
        max_iter=REF_ITER,
        theta=THETA,
        n_mc_samples=N_MC,
        seed=SEED,
        initial_refinements=INITIAL_REFINEMENTS,
        jitter_initial_mesh=JITTER_INITIAL_MESH,
        jitter_amount=JITTER_AMOUNT,
        max_ndof=MAX_NDOF,
        verbose=True,
        domain=DOMAIN,
    )

    # --------------------------------------------------------
    # Plot only the first MAIN_ITER iterations
    # --------------------------------------------------------
    n_mid = min(MAIN_ITER, len(hist_mid_full.ndof))
    n_mc = min(MAIN_ITER, len(hist_mc_full.ndof))

    hist_mid = truncate_history(hist_mid_full, n_mid)
    hist_mc = truncate_history(hist_mc_full, n_mc)

    plot_estimator_histories(
        hist_mid,
        hist_mc,
        n_mc_samples=N_MC,
        filename="afem_estimator_midpoint_vs_mc.png",
    )

    plot_l2_h1_errors(
        hist_mid,
        hist_mc,
        filename="afem_l2_h1_errors_midpoint_vs_mc.png",
    )

    plot_runtime_breakdown(
        hist_mid_full,
        hist_mc_full,
        filename="afem_runtime_midpoint_vs_mc.png",
    )

    #plot_mesh(mesh_mid_final, title="Final reference mesh: midpoint load")
    #plot_mesh(mesh_mc_final, title="Final reference mesh: Monte Carlo load")