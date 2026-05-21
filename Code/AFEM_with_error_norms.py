"""
AFEM comparison for

    -Delta u = |sin(pi * 2^5 * 3 * x)|   in Omega,
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

AFEM loop:

    SOLVE -> ESTIMATE -> MARK -> REFINE

The code also computes relative errors against a precomputed reference solution:

    ||u_ref - u_h||_L2 / ||u_ref||_L2

and

    ||grad(u_ref - u_h)||_L2 / ||grad u_ref||_L2.

Error integration is performed on the reference mesh. Reference quadrature data
are precomputed once, then errors are computed on the fly during AFEM iterations.

Requirements:

    pip install scikit-fem numpy matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
import os
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

    This is optional. It can reduce midpoint aliasing for the oscillatory RHS.
    Boundary nodes remain fixed.
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


# ============================================================
# Reference solution save/load
# ============================================================

def save_reference_solution(filename: str, mesh: MeshTri, u: np.ndarray) -> None:
    """
    Save reference mesh and solution to a .npz file.
    """
    np.savez(
        filename,
        p=mesh.p,
        t=mesh.t,
        u=u,
    )


def load_reference_solution(filename: str) -> tuple[MeshTri, np.ndarray]:
    """
    Load reference mesh and solution from a .npz file.
    """
    data = np.load(filename)
    mesh = MeshTri(data["p"], data["t"])
    u = data["u"]
    return mesh, u


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

    On each triangle T:

        f_T = (1 / |T|) int_T f(x, y) dx dy.

    Uses a tensor-product Gauss-Legendre rule on [0,1]^2 and the Duffy map

        r = xi,
        s = (1 - xi) eta

    from the unit square to the reference triangle.
    """
    if n_gauss < 1:
        raise ValueError("n_gauss must be at least 1.")

    nodes, weights = np.polynomial.legendre.leggauss(n_gauss)

    xi_nodes = 0.5 * (nodes + 1.0)
    xi_weights = 0.5 * weights

    eta_nodes = xi_nodes
    eta_weights = xi_weights

    p0, p1, p2, _, _, _ = triangle_geometry(mesh)

    d10 = p1 - p0
    d20 = p2 - p0

    nelems = mesh.t.shape[1]
    fK = np.zeros(nelems)

    for xi, wxi in zip(xi_nodes, xi_weights):
        one_minus_xi = 1.0 - xi

        for eta, weta in zip(eta_nodes, eta_weights):
            r = xi
            s = one_minus_xi * eta

            # Weight for the normalized cell average.
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

    For P1 FEM, Delta u_h = 0 inside each triangle. Therefore the volume
    residual is just f_h.

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

    Choose a set M such that

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
# Reference error computation on reference mesh
# ============================================================

def eval_p1_at_points_chunked(
    mesh: MeshTri,
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    chunk_size: int = 30_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a P1 finite element function and its gradient at many points
    in chunks.

    This avoids huge memory allocations inside skfem's element_finder.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    values = np.empty_like(x, dtype=float)
    grads = np.empty((x.size, 2), dtype=float)

    all_grads = p1_element_gradients(mesh, u)
    finder = mesh.element_finder()

    p = mesh.p
    t = mesh.t

    for start in range(0, x.size, chunk_size):
        end = min(start + chunk_size, x.size)

        xc = x[start:end]
        yc = y[start:end]

        elems = finder(xc, yc)

        if np.any(elems < 0):
            nbad = int(np.sum(elems < 0))
            raise ValueError(f"{nbad} quadrature points were not found in the mesh.")

        elems = elems.astype(np.int64)

        verts = t[:, elems]

        P0 = p[:, verts[0]]
        P1 = p[:, verts[1]]
        P2 = p[:, verts[2]]

        b11 = P1[0] - P0[0]
        b12 = P2[0] - P0[0]
        b21 = P1[1] - P0[1]
        b22 = P2[1] - P0[1]

        detB = b11 * b22 - b12 * b21

        dx = xc - P0[0]
        dy = yc - P0[1]

        r = (b22 * dx - b12 * dy) / detB
        s = (-b21 * dx + b11 * dy) / detB

        lam0 = 1.0 - r - s
        lam1 = r
        lam2 = s

        u0 = u[verts[0]]
        u1 = u[verts[1]]
        u2 = u[verts[2]]

        values[start:end] = lam0 * u0 + lam1 * u1 + lam2 * u2
        grads[start:end] = all_grads[elems]

    return values, grads


class ReferenceErrorComputer:
    """
    Precompute quadrature data on a reference mesh and use it to compute
    relative L2 and H1-seminorm errors for AFEM iterates.

    The integration is always performed over the reference mesh.

    Reference-side quantities are computed once:
        x_q, y_q, w_q, u_ref(x_q), grad u_ref(x_q).

    During each AFEM iteration, only u_h and grad u_h are evaluated at
    the precomputed reference quadrature points.
    """

    def __init__(
        self,
        mesh_ref: MeshTri,
        u_ref: np.ndarray,
        n_gauss_error: int = 3,
    ):
        self.mesh_ref = mesh_ref
        self.u_ref = u_ref
        self.n_gauss_error = n_gauss_error

        t0 = time.perf_counter()

        (
            self.xq,
            self.yq,
            self.wq,
            self.uref_q,
            self.grad_ref_q,
        ) = self._precompute_reference_quadrature()

        self.l2_ref_sq = float(np.sum(self.wq * self.uref_q**2))
        self.h1_ref_sq = float(
            np.sum(self.wq * np.sum(self.grad_ref_q**2, axis=1))
        )

        elapsed = time.perf_counter() - t0

        print(
            "\nReference quadrature data precomputed:\n"
            f"  number of quadrature points = {self.xq.size}\n"
            f"  L2 reference norm squared   = {self.l2_ref_sq:.6e}\n"
            f"  H1 reference seminorm sq.   = {self.h1_ref_sq:.6e}\n"
            f"  time                         = {elapsed:.2f}s\n"
        )

    def _precompute_reference_quadrature(self):
        """
        Precompute quadrature points, weights, reference values, and
        reference gradients on the reference mesh.
        """
        nodes, weights = np.polynomial.legendre.leggauss(self.n_gauss_error)

        xi_nodes = 0.5 * (nodes + 1.0)
        xi_weights = 0.5 * weights

        eta_nodes = xi_nodes
        eta_weights = xi_weights

        P0, P1, P2, area_ref, _, _ = triangle_geometry(self.mesh_ref)

        d10 = P1 - P0
        d20 = P2 - P0

        tref = self.mesh_ref.t
        grad_ref_elem = p1_element_gradients(self.mesh_ref, self.u_ref)

        xq_all = []
        yq_all = []
        wq_all = []
        uref_all = []
        grad_ref_all = []

        for xi, wxi in zip(xi_nodes, xi_weights):
            one_minus_xi = 1.0 - xi

            for eta, weta in zip(eta_nodes, eta_weights):
                r = xi
                s = one_minus_xi * eta

                w_phys = 2.0 * area_ref * wxi * weta * one_minus_xi

                xq = P0[0] + r * d10[0] + s * d20[0]
                yq = P0[1] + r * d10[1] + s * d20[1]

                lam0 = 1.0 - r - s
                lam1 = r
                lam2 = s

                uref_q = (
                    lam0 * self.u_ref[tref[0]]
                    + lam1 * self.u_ref[tref[1]]
                    + lam2 * self.u_ref[tref[2]]
                )

                xq_all.append(xq)
                yq_all.append(yq)
                wq_all.append(w_phys)
                uref_all.append(uref_q)
                grad_ref_all.append(grad_ref_elem)

        xq_all = np.concatenate(xq_all)
        yq_all = np.concatenate(yq_all)
        wq_all = np.concatenate(wq_all)
        uref_all = np.concatenate(uref_all)
        grad_ref_all = np.vstack(grad_ref_all)

        return xq_all, yq_all, wq_all, uref_all, grad_ref_all

    def compute_error(
        self,
        mesh: MeshTri,
        u: np.ndarray,
        chunk_size: int = 30_000,
    ) -> tuple[float, float]:
        """
        Compute relative L2 and H1-seminorm errors of u against u_ref.

        Integration is over the precomputed reference quadrature points.
        """
        u_q, grad_q = eval_p1_at_points_chunked(
            mesh,
            u,
            self.xq,
            self.yq,
            chunk_size=chunk_size,
        )

        diff = self.uref_q - u_q
        grad_diff = self.grad_ref_q - grad_q

        l2_err_sq = float(np.sum(self.wq * diff**2))
        h1_err_sq = float(np.sum(self.wq * np.sum(grad_diff**2, axis=1)))

        rel_l2 = np.sqrt(l2_err_sq / self.l2_ref_sq)
        rel_h1 = np.sqrt(h1_err_sq / self.h1_ref_sq)

        return float(rel_l2), float(rel_h1)


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
    error_time: list[float]
    rel_l2_error: list[float]
    rel_h1_error: list[float]


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
    error_computer: ReferenceErrorComputer | None = None,
    error_chunk_size: int = 30_000,
) -> tuple[AFEMHistory, MeshTri, np.ndarray]:
    """
    Run the AFEM loop for one load approximation method.

    If error_computer is provided, relative L2 and H1 errors are computed
    on the fly after every solve.
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
        error_time=[],
        rel_l2_error=[],
        rel_h1_error=[],
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
        # REFERENCE ERRORS
        # ----------------------------
        if error_computer is not None:
            t0 = time.perf_counter()

            rel_l2, rel_h1 = error_computer.compute_error(
                mesh,
                u,
                chunk_size=error_chunk_size,
            )

            error_t = time.perf_counter() - t0
        else:
            rel_l2 = np.nan
            rel_h1 = np.nan
            error_t = 0.0

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
        hist.error_time.append(error_t)
        hist.rel_l2_error.append(rel_l2)
        hist.rel_h1_error.append(rel_h1)

        if verbose:
            print(
                f"{method:>10s} | it={it:02d} | "
                f"ndof={hist.ndof[-1]:8d} | "
                f"nelems={hist.nelems[-1]:8d} | "
                f"marked={marked.size:8d} | "
                f"eta={eta:.6e} | "
                f"L2={rel_l2:.3e} | "
                f"H1={rel_h1:.3e} | "
                f"solve={solve_t:.3f}s | "
                f"est={estimator_t:.3f}s | "
                f"err={error_t:.3f}s"
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
    plt.title("AFEM estimator: midpoint vs quadrature vs Monte Carlo")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


def plot_l2_h1_errors_three(
    hist_mid: AFEMHistory,
    hist_quad: AFEMHistory,
    hist_mc: AFEMHistory,
    filename: str = "afem_l2_h1_errors.png",
):
    """
    Plot relative H1-seminorm and L2-errors for all three methods.

    Solid lines: H1-seminorm errors.
    Dotted lines: L2-errors.
    """
    plt.figure(figsize=(8.4, 5.8))

    plt.loglog(
        hist_mid.ndof,
        hist_mid.rel_h1_error,
        "o-",
        label=r"midpoint, $H^1$",
    )
    plt.loglog(
        hist_mid.ndof,
        hist_mid.rel_l2_error,
        "o:",
        label=r"midpoint, $L^2$",
    )

    plt.loglog(
        hist_quad.ndof,
        hist_quad.rel_h1_error,
        "^-",
        label=r"quadrature, $H^1$",
    )
    plt.loglog(
        hist_quad.ndof,
        hist_quad.rel_l2_error,
        "^:",
        label=r"quadrature, $L^2$",
    )

    plt.loglog(
        hist_mc.ndof,
        hist_mc.rel_h1_error,
        "s-",
        label=r"Monte Carlo, $H^1$",
    )
    plt.loglog(
        hist_mc.ndof,
        hist_mc.rel_l2_error,
        "s:",
        label=r"Monte Carlo, $L^2$",
    )

    add_reference_slope(
        hist_mc.ndof,
        hist_mc.rel_h1_error[-1],
        -0.5,
        r"$\mathrm{ndof}^{-1/2}$",
    )

    add_reference_slope(
        hist_mc.ndof,
        hist_mc.rel_l2_error[-1],
        -1.0,
        r"$\mathrm{ndof}^{-1}$",
    )

    plt.xlabel("degrees of freedom")
    plt.ylabel("relative error")
    plt.title(r"AFEM errors against reference solution")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(fontsize=9)
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
    Plot cumulative solve + estimator + refine + error time for each method.
    """
    plt.figure(figsize=(8.0, 5.2))

    def cumulative_total_time(hist: AFEMHistory):
        n = len(hist.solve_time)
        refine = np.asarray(hist.refine_time[:n])
        if len(refine) < n:
            refine = np.pad(refine, (0, n - len(refine)))
        return np.cumsum(
            np.asarray(hist.solve_time)
            + np.asarray(hist.estimator_time)
            + np.asarray(hist.error_time)
            + refine
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
    """
    Plot mesh without opening an extra empty figure.
    """
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
    DOMAIN = "lshape"      # "square" or "lshape"

    THETA = 0.5
    MAX_ITER = 10

    N_MC = 20
    N_GAUSS = 4

    SEED = 12345

    INITIAL_REFINEMENTS = 3

    JITTER_INITIAL_MESH = False
    JITTER_AMOUNT = 0.10

    MAX_NDOF = 200_000

    # --------------------------------------------------------
    # Reference solution parameters
    # --------------------------------------------------------
    REFERENCE_FILE = "reference_solution.npz"
    COMPUTE_NEW_REFERENCE = True

    REF_METHOD = "quadrature"
    REF_MAX_ITER = 10
    REF_N_GAUSS = 4
    REF_N_MC = 100
    REF_MAX_NDOF = 250_000

    # Error quadrature over the reference mesh.
    #
    # Warning:
    #   Number of error quadrature points =
    #       n_ref_elements * N_GAUSS_ERROR^2.
    #
    # For large reference meshes, N_GAUSS_ERROR = 2 or 3 is recommended.
    N_GAUSS_ERROR = 2

    # Chunk size for evaluating current AFEM solution at reference quadrature points.
    ERROR_CHUNK_SIZE = 250_000

    # --------------------------------------------------------
    # Compute or load reference solution
    # --------------------------------------------------------
    if COMPUTE_NEW_REFERENCE or not os.path.exists(REFERENCE_FILE):
        print("\nComputing reference solution\n")

        hist_ref, mesh_ref, u_ref = run_afem(
            method=REF_METHOD,
            max_iter=REF_MAX_ITER,
            theta=THETA,
            n_mc_samples=REF_N_MC,
            n_gauss=REF_N_GAUSS,
            seed=SEED + 999_000,
            initial_refinements=INITIAL_REFINEMENTS,
            jitter_initial_mesh=JITTER_INITIAL_MESH,
            jitter_amount=JITTER_AMOUNT,
            max_ndof=REF_MAX_NDOF,
            verbose=True,
            domain=DOMAIN,
            error_computer=None,
            error_chunk_size=ERROR_CHUNK_SIZE,
        )

        save_reference_solution(REFERENCE_FILE, mesh_ref, u_ref)

        print(
            "\nReference solution saved:\n"
            f"  file   = {REFERENCE_FILE}\n"
            f"  method = {REF_METHOD}\n"
            f"  ndof   = {mesh_ref.p.shape[1]}\n"
            f"  elems  = {mesh_ref.t.shape[1]}\n"
        )

    else:
        print(f"\nLoading reference solution from {REFERENCE_FILE}\n")
        mesh_ref, u_ref = load_reference_solution(REFERENCE_FILE)

        print(
            "\nReference solution loaded:\n"
            f"  ndof  = {mesh_ref.p.shape[1]}\n"
            f"  elems = {mesh_ref.t.shape[1]}\n"
        )

    # --------------------------------------------------------
    # Precompute reference quadrature data once
    # --------------------------------------------------------
    error_computer = ReferenceErrorComputer(
        mesh_ref=mesh_ref,
        u_ref=u_ref,
        n_gauss_error=N_GAUSS_ERROR,
    )

    # --------------------------------------------------------
    # Run three AFEM methods with errors computed on the fly
    # --------------------------------------------------------
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
        error_computer=error_computer,
        error_chunk_size=ERROR_CHUNK_SIZE,
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
        domain=DOMAIN,
        error_computer=error_computer,
        error_chunk_size=ERROR_CHUNK_SIZE,
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
        domain=DOMAIN,
        error_computer=error_computer,
        error_chunk_size=ERROR_CHUNK_SIZE,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    plot_afem_histories_three(
        hist_mid,
        hist_quad,
        hist_mc,
        n_mc_samples=N_MC,
        n_gauss=N_GAUSS,
        filename="afem_three_methods_estimator.png",
    )

    plot_l2_h1_errors_three(
        hist_mid,
        hist_quad,
        hist_mc,
        filename="afem_l2_h1_errors.png",
    )

    plot_runtime_breakdown(
        hist_mid,
        hist_quad,
        hist_mc,
        filename="afem_runtime_breakdown.png",
    )

    #plot_mesh(mesh_mid, title="Final adaptive mesh: midpoint load")
    #plot_mesh(mesh_quad, title="Final adaptive mesh: deterministic quadrature load")
    #plot_mesh(mesh_mc, title="Final adaptive mesh: Monte Carlo load")