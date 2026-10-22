# Adaptive Finite Element Experiments

A Python codebase for studying adaptive finite element methods for the Poisson
problem with homogeneous Dirichlet boundary conditions:

```math
-\Delta u = f \quad \text{in } \Omega, \qquad u = 0 \quad \text{on } \partial\Omega.
```

Built on scikit-fem, it focuses on how mesh refinement and load approximation
affect convergence.

## Main capabilities

- Piecewise linear (P1) elements on triangular and tetrahedral meshes, with square,
  L-shaped, and cube domains.
- Adaptive refinement using a residual error estimator and Dörfler marking,
  alongside uniform refinement.
- Load assembly using standard quadrature, midpoint quadrature, or Monte Carlo
  sampling.
- Experiments comparing refinement strategies, load methods, Monte Carlo sample
  counts, and repeated random realizations.
- Optional reference-solution error evaluation, saved numerical results, and plots
  of meshes, solutions, and convergence.

## Installation

Requires Python 3.10 or newer. From the repository root, install the package and
its dependencies, preferably in a virtual environment:

```bash
python -m pip install -e .
```

## Running experiments

Run experiment modules from the repository root. For example:

```bash
python -m experiments.run_uniform
python -m experiments.run_refinement_comparison --method quadrature
```

Experiment settings live in the corresponding scripts under `experiments/`.
Adjust the right-hand side and `AFEMConfig` to choose the domain, refinement,
load approximation, iteration count, and output directory. Available settings
are defined in [`afem/core/config.py`](afem/core/config.py).

Results are written to the configured directory under `results/` by default.
Runs save configuration and convergence histories; comparison studies also save
final meshes and solutions. Plots use PDF by default, with optional PNG export.
Comparison scripts support `--plot-only` to regenerate figures from saved data.
Choose a new output directory to keep results from earlier runs.

## Code layout

| Directory | Contents |
| --- | --- |
| `afem/core/` | Configuration, mesh construction, assembly, solvers, error estimation, and refinement loops. |
| `afem/problems/` | Example right-hand sides and exact solutions. |
| `afem/utils/` | Plotting, figure export, and Monte Carlo study utilities. |
| `experiments/` | Experiment scripts and tools for working with saved results. |
| `testing/` | Automated tests and exploratory scripts. |
| `results/` | Simulation outputs and figures. |

## Tests

Run the automated tests from the repository root:

```bash
python -m unittest discover -s testing -p "test_*.py"
```
