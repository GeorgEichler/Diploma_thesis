"""
Uniform-refinement experiment for

    -Delta u = |sin(pi * 2^5 * 3 * x)|   in (0, 1)^2,
          u = 0                          on boundary.

The code compares two piecewise-constant load approximations:

    1. midpoint:
         f_K = f(centroid of K)

    2. Monte Carlo:
         f_K = 1/N sum_i f(X_i^K),
         X_i^K uniformly distributed in triangle K.

For each uniform mesh level, the P1 FEM solution is computed with skfem.

Errors are measured against a fine reference solution computed on a mesh with
REF_EXTRA more uniform refinement levels and a Monte Carlo load with many samples.

Plots:
    - relative H1-seminorm error
    - relative L2-error
    - combined plot

"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from skfem import MeshTri, Basis, ElementTriP1, asm, solve, condense
from skfem.models.poisson import laplace


# ---------------------------------------------------------------------
# Problem data
# ---------------------------------------------------------------------

L_OSC = 5


def rhs_f(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Right-hand side

        f(x, y) = |sin(pi * 2^5 * 3 * x)| = |sin(96 pi x)|.

    y is unused, but included for a standard two-dimensional interface
    """
    return np.abs(np.sin(np.pi * (2**L_OSC) * 3.0 * x))


# ---------------------------------------------------------------------
# Structured triangular meshes
# ---------------------------------------------------------------------

def unit_square_structured_tri_mesh(n: int) -> MeshTri:
    """
    Create a structured triangular mesh of the unit square.

    The square is split into n x n smaller squares.
    Each square is divided along the diagonal

        lower-left ---- upper-right.


    Parameters
    ----------
    n:
        Number of square subdivisions per coordinate direction.

    Returns
    -------
    mesh:
        skfem MeshTri.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")

    xs = np.linspace(0.0, 1.0, n + 1)
    ys = np.linspace(0.0, 1.0, n + 1)

    points = []
    for j in range(n + 1):
        for i in range(n + 1):
            points.append([xs[i], ys[j]])
    p = np.array(points, dtype=float).T

    # convert grid (i,j) into global index
    def vid(i: int, j: int) -> int:
        return i + (n + 1) * j

    triangles = []
    # crete triangles
    for j in range(n):
        for i in range(n):
            ll = vid(i, j)
            lr = vid(i + 1, j)
            ur = vid(i + 1, j + 1)
            ul = vid(i, j + 1)

            # Lower triangle: ll -> lr -> ur
            triangles.append([ll, lr, ur])

            # Upper triangle: ll -> ur -> ul
            triangles.append([ll, ur, ul])

    t = np.array(triangles, dtype=int).T

    return MeshTri(p, t)


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------

def triangle_geometry(mesh: MeshTri):
    """
    Return useful geometric quantities:
    end points, area, diameter and the center
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


# ---------------------------------------------------------------------
# Piecewise-constant load approximations
# ---------------------------------------------------------------------

def p0_load_midpoint(mesh: MeshTri) -> np.ndarray:
    """
    Piecewise-constant midpoint approximation.

    On every triangle K,

        f_K = f(x_K),

    where x_K is the triangle centroid.
    """
    _, _, _, _, _, centroid = triangle_geometry(mesh)
    return rhs_f(centroid[0], centroid[1])


def p0_load_mc(
    mesh: MeshTri,
    n_samples: int = 20,
    seed: int | None = None,
) -> np.ndarray:
    """
    Piecewise-constant Monte Carlo approximation.

    On every triangle K,

        f_K = 1/N sum_i f(X_i^K),

    where X_i^K are uniform random samples in K.

    Sampling uses the reflection trick on the reference triangle:
        draw r, s ~ U(0,1);
        if r+s > 1, replace (r,s) by (1-r, 1-s).
    """
    if n_samples < 1:
        raise ValueError("n_samples must be >= 1.")

    rng = np.random.default_rng(seed)

    p0, p1, p2, _, _, _ = triangle_geometry(mesh)
    nelems = mesh.t.shape[1]

    d10 = p1 - p0
    d20 = p2 - p0

    values = np.zeros(nelems)

    # Use reflection trick to sample points in triangles
    for _ in range(n_samples):
        r = rng.random(nelems)
        s = rng.random(nelems)

        outside = r + s > 1.0
        r[outside] = 1.0 - r[outside]
        s[outside] = 1.0 - s[outside]

        x = p0[0] + r * d10[0] + s * d20[0]
        y = p0[1] + r * d10[1] + s * d20[1]

        values += rhs_f(x, y)

    # Outputs one value for every triangle
    return values / float(n_samples)


def assemble_p0_load_vector(mesh: MeshTri, fK: np.ndarray) -> np.ndarray:
    """
    Assemble the load vector

        b_i = int_D f_h phi_i dx,

    where f_h is P0 with element values fK and phi_i are P1 basis functions.

    Since f_h is constant on K and phi_i is linear,

        int_K f_K phi_i dx = f_K |K| / 3

    for each local vertex basis function.
    """
    _, _, _, area, _, _ = triangle_geometry(mesh)

    nvertices = mesh.p.shape[1]
    b = np.zeros(nvertices)

    local_contribution = fK * area / 3.0

    # Assemble RhS of load vector
    for local_vertex in range(3):
        np.add.at(b, mesh.t[local_vertex], local_contribution)

    return b


# ---------------------------------------------------------------------
# FEM solve
# ---------------------------------------------------------------------

def solve_poisson_p1(mesh: MeshTri, fK: np.ndarray) -> np.ndarray:
    """
    Solve the P1 FEM problem with homogeneous Dirichlet boundary condition:

        int_D grad u_h · grad v_h dx = int_D f_h v_h dx.

    Here f_h is a P0 function with values fK.
    """
    basis = Basis(mesh, ElementTriP1())

    A = asm(laplace, basis)
    b = assemble_p0_load_vector(mesh, fK)

    # Homogeneous Dirichlet boundary condition on the whole boundary.
    interior = mesh.interior_nodes()

    u = solve(*condense(A, b, I=interior))
    return u


# ---------------------------------------------------------------------
# P1 gradients
# ---------------------------------------------------------------------

def p1_element_gradients(mesh: MeshTri, u: np.ndarray) -> np.ndarray:
    """
    Compute the constant gradient of a P1 function on every triangle.

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

    twice_area_signed = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

    grad_phi0_x = (y1 - y2) / twice_area_signed
    grad_phi0_y = (x2 - x1) / twice_area_signed

    grad_phi1_x = (y2 - y0) / twice_area_signed
    grad_phi1_y = (x0 - x2) / twice_area_signed

    grad_phi2_x = (y0 - y1) / twice_area_signed
    grad_phi2_y = (x1 - x0) / twice_area_signed

    grads = np.empty((mesh.t.shape[1], 2))
    grads[:, 0] = u0 * grad_phi0_x + u1 * grad_phi1_x + u2 * grad_phi2_x
    grads[:, 1] = u0 * grad_phi0_y + u1 * grad_phi1_y + u2 * grad_phi2_y

    return grads


# ---------------------------------------------------------------------
# Exact evaluation of structured P1 functions on nested meshes
# ---------------------------------------------------------------------

def structured_p1_value_and_grad(
    x: float,
    y: float,
    u: np.ndarray,
    n: int,
    grads: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    Evaluate a P1 function and its gradient on a structured n x n mesh.

    This assumes the mesh was created by unit_square_structured_tri_mesh(n).

    Parameters
    ----------
    x, y:
        Evaluation point in [0, 1]^2.
    u:
        Nodal values on the structured mesh.
    n:
        Number of square subdivisions per coordinate direction.
    grads:
        Element gradients from p1_element_gradients.

    Returns
    -------
    value:
        u_h(x, y).
    grad:
        grad u_h on the triangle containing (x, y).
    """
    # Clamp points lying numerically on x=1 or y=1 into the last cell.
    i = min(max(int(np.floor(n * x)), 0), n - 1)
    j = min(max(int(np.floor(n * y)), 0), n - 1)

    xi = n * x - i
    eta = n * y - j

    # Numerical safety near the boundary.
    xi = min(max(xi, 0.0), 1.0)
    eta = min(max(eta, 0.0), 1.0)

    def vid(ii: int, jj: int) -> int:
        return ii + (n + 1) * jj

    ll = vid(i, j)
    lr = vid(i + 1, j)
    ur = vid(i + 1, j + 1)
    ul = vid(i, j + 1)

    base_elem = 2 * (j * n + i)

    if eta <= xi:
        # Lower triangle: ll, lr, ur
        lam0 = 1.0 - xi
        lam1 = xi - eta
        lam2 = eta

        value = lam0 * u[ll] + lam1 * u[lr] + lam2 * u[ur]
        grad = grads[base_elem]
    else:
        # Upper triangle: ll, ur, ul
        lam0 = 1.0 - eta
        lam1 = xi
        lam2 = eta - xi

        value = lam0 * u[ll] + lam1 * u[ur] + lam2 * u[ul]
        grad = grads[base_elem + 1]

    return float(value), grad


# ---------------------------------------------------------------------
# Error computation against a fine reference solution
# ---------------------------------------------------------------------

def relative_errors_against_reference(
    mesh_ref: MeshTri,
    u_ref: np.ndarray,
    mesh_coarse: MeshTri,
    u_coarse: np.ndarray,
    n_coarse: int,
) -> tuple[float, float]:
    """
    Compute relative H1-seminorm and L2 errors against a reference solution.

    The integration is performed over the fine reference mesh.

    Because all meshes are nested structured meshes with the same diagonal
    convention, the coarse P1 function is still affine on every fine triangle.
    Therefore:

        - the H1 error is integrated exactly elementwise;
        - the L2 error is integrated exactly by a degree-2 triangle rule.

    Returns
    -------
    rel_h1:
        ||grad(u_ref - u_coarse)|| / ||grad u_ref||.
    rel_l2:
        ||u_ref - u_coarse|| / ||u_ref||.
    """
    _, _, _, area_ref, _, _ = triangle_geometry(mesh_ref)

    grads_ref = p1_element_gradients(mesh_ref, u_ref)
    grads_coarse = p1_element_gradients(mesh_coarse, u_coarse)

    # Three-point quadrature rule, exact for quadratic polynomials.
    bary_q = np.array([
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ])

    l2_err_sq = 0.0
    l2_ref_sq = 0.0

    h1_err_sq = 0.0
    h1_ref_sq = 0.0

    p = mesh_ref.p
    t = mesh_ref.t

    for k in range(t.shape[1]):
        verts = t[:, k]
        coords = p[:, verts]
        uvals_ref = u_ref[verts]

        # H1 seminorm contribution. On each fine triangle, grad u_ref is constant.
        centroid = np.mean(coords, axis=1)
        _, grad_coarse = structured_p1_value_and_grad(
            centroid[0],
            centroid[1],
            u_coarse,
            n_coarse,
            grads_coarse,
        )

        grad_diff = grads_ref[k] - grad_coarse

        h1_err_sq += area_ref[k] * float(np.dot(grad_diff, grad_diff))
        h1_ref_sq += area_ref[k] * float(np.dot(grads_ref[k], grads_ref[k]))

        # L2 contribution via degree-2 quadrature.
        for lam in bary_q:
            xq, yq = coords @ lam
            uref_q = float(np.dot(lam, uvals_ref))

            ucoarse_q, _ = structured_p1_value_and_grad(
                xq,
                yq,
                u_coarse,
                n_coarse,
                grads_coarse,
            )

            diff = uref_q - ucoarse_q

            l2_err_sq += area_ref[k] / 3.0 * diff**2
            l2_ref_sq += area_ref[k] / 3.0 * uref_q**2

    rel_h1 = np.sqrt(h1_err_sq / h1_ref_sq)
    rel_l2 = np.sqrt(l2_err_sq / l2_ref_sq)

    return float(rel_h1), float(rel_l2)


# ---------------------------------------------------------------------
# Uniform experiment
# ---------------------------------------------------------------------

@dataclass
class UniformHistory:
    levels: list[int]
    ndof: list[int]
    nelems: list[int]
    rel_h1: list[float]
    rel_l2: list[float]


def compute_load(
    mesh: MeshTri,
    method: str,
    n_mc_samples: int,
    seed: int,
) -> np.ndarray:
    """
    Convenience wrapper for the two load approximations.
    """
    if method == "midpoint":
        return p0_load_midpoint(mesh)

    if method == "mc":
        return p0_load_mc(mesh, n_samples=n_mc_samples, seed=seed)

    raise ValueError("method must be 'midpoint' or 'mc'.")


def run_uniform_experiment(
    max_level: int = 7,
    ref_extra: int = 2,
    n_mc: int = 20,
    n_ref_mc: int = 100,
    seed: int = 12345,
) -> tuple[UniformHistory, UniformHistory]:
    """
    Run the uniform-refinement comparison.

    Mesh level ell means n = 2^ell square subdivisions per direction.

    The reference mesh has level max_level + ref_extra.
    The reference solution uses Monte Carlo P0 load with n_ref_mc samples
    per element.
    """
    ref_level = max_level + ref_extra
    n_ref = 2**ref_level

    print("\nComputing reference solution")
    print(f"  reference level       = {ref_level}")
    print(f"  reference subdivisions= {n_ref} x {n_ref}")
    print(f"  reference MC samples  = {n_ref_mc} per element")

    mesh_ref = unit_square_structured_tri_mesh(n_ref)
    fK_ref = p0_load_mc(mesh_ref, n_samples=n_ref_mc, seed=seed + 999_000)
    u_ref = solve_poisson_p1(mesh_ref, fK_ref)

    histories: dict[str, UniformHistory] = {}

    for method in ["midpoint", "mc"]:
        hist = UniformHistory(
            levels=[],
            ndof=[],
            nelems=[],
            rel_h1=[],
            rel_l2=[],
        )

        print(f"\nRunning method: {method}")

        for level in range(max_level + 1):
            n = 2**level
            mesh = unit_square_structured_tri_mesh(n)

            fK = compute_load(
                mesh,
                method=method,
                n_mc_samples=n_mc,
                seed=seed + 10_000 * level,
            )

            u = solve_poisson_p1(mesh, fK)

            rel_h1, rel_l2 = relative_errors_against_reference(
                mesh_ref=mesh_ref,
                u_ref=u_ref,
                mesh_coarse=mesh,
                u_coarse=u,
                n_coarse=n,
            )

            hist.levels.append(level)
            hist.ndof.append(mesh.p.shape[1])
            hist.nelems.append(mesh.t.shape[1])
            hist.rel_h1.append(rel_h1)
            hist.rel_l2.append(rel_l2)

            print(
                f"  level={level:2d}, "
                f"ndof={hist.ndof[-1]:7d}, "
                f"nelems={hist.nelems[-1]:7d}, "
                f"rel H1={rel_h1:.6e}, "
                f"rel L2={rel_l2:.6e}"
            )

        histories[method] = hist

    return histories["midpoint"], histories["mc"]


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def add_reference_slope(
    ndof: list[int],
    y_anchor: float,
    exponent: float,
    label: str,
):
    """
    Add a reference slope C * ndof^exponent to the current log-log plot.
    """
    ndof_arr = np.asarray(ndof, dtype=float)
    slope = y_anchor * (ndof_arr / ndof_arr[-1]) ** exponent
    plt.loglog(ndof_arr, slope, "k--", linewidth=1.2, label=label)


def plot_separate(hist_mid: UniformHistory, hist_mc: UniformHistory, n_mc: int):
    """
    Produce two separate plots:
        1. relative H1-seminorm error,
        2. relative L2-error.
    """
    plt.figure(figsize=(7.0, 5.0))
    plt.loglog(hist_mid.ndof, hist_mid.rel_h1, "o-", label=r"midpoint $\Pi_0 f$")
    plt.loglog(hist_mc.ndof, hist_mc.rel_h1, "s-", label=rf"Monte Carlo $\widehat{{\Pi}}_0 f$, N={n_mc}")
    add_reference_slope(hist_mc.ndof, hist_mc.rel_h1[-1], -0.5, r"$\mathrm{ndof}^{-1/2}$")
    plt.xlabel("degrees of freedom")
    plt.ylabel(r"relative $H^1$-seminorm error")
    plt.title(r"Uniform refinement: relative $H^1$-seminorm error")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("uniform_H1_error.png", dpi=200)
    plt.show()

    plt.figure(figsize=(7.0, 5.0))
    plt.loglog(hist_mid.ndof, hist_mid.rel_l2, "o-", label=r"midpoint $\Pi_0 f$")
    plt.loglog(hist_mc.ndof, hist_mc.rel_l2, "s-", label=rf"Monte Carlo $\widehat{{\Pi}}_0 f$, N={n_mc}")
    add_reference_slope(hist_mc.ndof, hist_mc.rel_l2[-1], -1.0, r"$\mathrm{ndof}^{-1}$")
    plt.xlabel("degrees of freedom")
    plt.ylabel(r"relative $L^2$ error")
    plt.title(r"Uniform refinement: relative $L^2$ error")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("uniform_L2_error.png", dpi=200)
    plt.show()


def plot_combined_storn_style(
    hist_mid: UniformHistory,
    hist_mc: UniformHistory,
    n_mc: int,
):
    """
    Combined plot similar in spirit to Storn's Figure 1.

    Solid lines:
        relative H1-seminorm errors.

    Dotted lines:
        relative L2-errors.
    """
    plt.figure(figsize=(7.0, 5.0))

    plt.loglog(
        hist_mid.ndof,
        hist_mid.rel_h1,
        "o-",
        label=r"midpoint $\Pi_0 f$, $H^1$",
    )
    plt.loglog(
        hist_mid.ndof,
        hist_mid.rel_l2,
        "o:",
        label=r"midpoint $\Pi_0 f$, $L^2$",
    )

    plt.loglog(
        hist_mc.ndof,
        hist_mc.rel_h1,
        "s-",
        label=rf"MC $\widehat{{\Pi}}_0 f$, N={n_mc}, $H^1$",
    )
    plt.loglog(
        hist_mc.ndof,
        hist_mc.rel_l2,
        "s:",
        label=rf"MC $\widehat{{\Pi}}_0 f$, N={n_mc}, $L^2$",
    )

    add_reference_slope(
        hist_mc.ndof,
        hist_mc.rel_h1[-1],
        -0.5,
        r"$\mathrm{ndof}^{-1/2}$",
    )

    plt.xlabel("degrees of freedom")
    plt.ylabel("relative error")
    plt.title("Uniform refinement: midpoint load vs Monte Carlo load")
    plt.grid(True, which="both", linestyle=":")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("uniform_combined_errors.png", dpi=200)
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    MAX_LEVEL = 7
    REF_EXTRA = 2

    # Use N_MC = 1 to reproduce the left-panel spirit of Figure 1,
    # or N_MC = 20 for the right-panel Monte Carlo setting.
    N_MC = 20

    # Reference solution sample count.
    N_REF_MC = 100

    SEED = 12345

    hist_mid, hist_mc = run_uniform_experiment(
        max_level=MAX_LEVEL,
        ref_extra=REF_EXTRA,
        n_mc=N_MC,
        n_ref_mc=N_REF_MC,
        seed=SEED,
    )

    plot_separate(hist_mid, hist_mc, n_mc=N_MC)
    plot_combined_storn_style(hist_mid, hist_mc, n_mc=N_MC)