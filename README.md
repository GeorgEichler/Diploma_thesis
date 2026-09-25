# AFEM scikit-fem framework

Modular adaptive finite element method framework for Poisson/Laplace problems

```math
-\Delta u = f \quad \text{in }\Omega, \qquad u=0 \quad \text{on }\partial\Omega.
```

Main features:
- P1 FEM on triangular/tetrahedral meshes.
- Dörfler marking.
- Load assembly by standard quadrature or Monte Carlo elementwise P0 approximation.
- Residual estimator with element residual and interior flux-jump term.
- Square, cube, L-shaped 2D, and simple custom mesh hooks.
- Saved plots for mesh, solution, and convergence histories.

## Install

```bash
cd afem_skfem_project
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e .
```

## Run

```bash
python -m experiments.run_square_mc
python -m experiments.run_square_quadrature
```

Run these commands from the project root.

## Compare load approximations

Set `RHS` and the shared `CONFIG` in `experiments/run_load_comparison.py`, then run:

```bash
python -m experiments.run_load_comparison
```

Set `MIDPOINT_ITERATIONS` in that script to run midpoint for more levels
(initially 12). Monte Carlo and standard quadrature use `CONFIG.max_iterations`
(initially 8). Set `MIDPOINT_ITERATIONS = None` to use the shared count for all
methods. Different history lengths are supported by the combined plot.

The experiment runs midpoint, Monte Carlo, and standard scikit-fem quadrature
sequentially. Each method saves its configuration, reference data, and error
history (`history.json`), final mesh and solution (`final_solution.npz`, arrays
`p`, `t`, `u`), RHS identity (`experiment.json`), and plots in its own subdirectory
of `results/load_comparison` before the next method starts.

The combined `relative_errors_comparison.png` uses one color per
method, solid lines for relative H1 seminorm error, and dashed lines for relative
L2 error, plotted against DOFs. Each method uses its own enriched final-mesh
reference solution; these are not a shared exact reference. The requested
labels Pi_0 f, hat(Pi)_0 f, and f identify the three methods; midpoint is an
approximation to the cell average, and standard quadrature is numerical integration.

The volume residual uses centroid values for midpoint, sampled cell averages for
Monte Carlo, and quadrature of `f**2` for standard quadrature. Set
`estimator_quadrature_order` to control the last method's estimator integration
independently of load assembly. With `None`, it follows `quadrature_order`, or
scikit-fem's default if both are `None`. Flux-jump contributions are unchanged.

Recreate the comparison from saved histories (including a partially completed run):

```bash
python -m experiments.run_load_comparison --plot-only
```

Rerunning the experiment overwrites that output directory's method results;
choose a different `CONFIG.output_dir` to retain a previous experiment.
