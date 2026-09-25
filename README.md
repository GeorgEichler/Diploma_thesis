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

## Uniform refinement

Edit `RHS` and `CONFIG` in `experiments/run_uniform.py`, then run:

```bash
python -m experiments.run_uniform
```

To reuse any existing experiment with uniform refinement, replace its runner:

```python
from afem.core.ufem import run_ufem

mesh, u, history = run_ufem(cfg, rhs)
```

It accepts the same `AFEMConfig` and RHS as `run_afem`, supports midpoint,
Monte Carlo, and standard quadrature, and saves the same history and plots.
Both `direct` and `energy` reference errors remain available. The reference
is computed on the final uniform mesh with the configured reference degree
and quadrature. `theta` is ignored; every element is refined, even for a zero
estimator. Saved configuration records `refinement_strategy="uniform"`.

Alternatively, set `refinement_strategy="uniform"` in an existing experiment's
configuration, including `run_load_comparison.py`. The default is `"adaptive"`.
Use a separate output directory and fewer iterations: every uniform 2D
refinement quadruples the triangle count. `max_iterations=4` means four solves
and three refinements after the initial mesh. For the load comparison, also
set `MIDPOINT_ITERATIONS = None` to use the same short uniform run for all methods.

## Compare adaptive and uniform refinement

To compare **adaptive versus uniform refinement** for a single load method,
edit `RHS`, `METHOD`, and `CONFIG` in `experiments/run_refinement_comparison.py`:

```bash
python -m experiments.run_refinement_comparison --method quadrature
```

The method choices are `midpoint`, `monte_carlo`, and `quadrature`; omitting
`--method` uses `METHOD` in the file. `CONFIG.max_iterations` controls adaptive
levels; `UNIFORM_ITERATIONS` controls uniform levels. Both start with the same
mesh and use the same load, sampling, and quadrature settings.

AFEM finishes and saves before uniform refinement starts. Results are stored
under `CONFIG.output_dir/<method>/adaptive/` and `uniform/`, including histories,
reference data, final meshes and solutions. The combined
`<method>/refinement_errors_comparison.png` uses a color for each strategy,
solid H1 and dashed L2 curves against DOFs. These are relative H1 seminorm and
L2 errors despite the shortened labels. Each strategy uses its own enriched
final-mesh reference, not a common reference solution.

Replot without solving using the same selected method:

```bash
python -m experiments.run_refinement_comparison --method quadrature --plot-only
```

Different methods use separate folders; rerunning the same method overwrites
its results. Choose another `CONFIG.output_dir` to retain earlier runs.

## Compare load methods with the same refinement strategy

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
