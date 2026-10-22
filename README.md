# Adaptive Finite Element Experiments for approximated loads

A Python codebase for studying adaptive finite element methods for the Poisson
problem with homogeneous Dirichlet boundary conditions:

```math
-\Delta u = f \quad \text{in } D, \qquad u = 0 \quad \text{on } \partial D.
```

Built on ideas from [Storn (2026)]([https://arxiv.org/abs/2602.04490]) to study the convergence behavior
of the AFEM algorithm with approximated right-hand side load functions.
These include on the one hand the deterministic $L^2$-orthogonal projection for a triangulation $\mathcal{T}$

$$
\Pi_0 f \colon L^2(D) \rightarrow P_0(\mathcal{T}), \quad \\Pi_0 f|_T = \frac{1}{|T|} \int_T f  dx.
$$

and for a randomized approximation operator with given sample number $N \in \mathbb N$

$$
\hat{\Pi}_0 f \colon L^2(D) \rightarrow P_0(\mathcal{T}), \quad \hat{\Pi}_0 f|_T = \frac{1}{N} \sum_{i=1}^N f(X_i^T), 
$$

where $X_i^T$ are uniformly distributed random variables on the simplex T.
We study if the AFEM with randomized load function still shows the typical convergence advantages when using adaptive refinement
instead of uniform refinement on a L-shape benchmark problem and compare its convergence behavior for a highly oscillating right-hand side.

## Main capabilities

- Implementation of Finite Element Method utilises the scikit-fem library
- Piecewise linear (P1) elements on triangular and tetrahedral meshes, with square,
  L-shaped, and cube domains.
- Adaptive refinement using a residual error estimator and Dörfler marking,
  alongside uniform refinement.
- Load assembly using standard quadrature, midpoint quadrature, or Monte Carlo
  sampling.
- Experiments comparing refinement strategies, load methods, Monte Carlo sample
  counts, and repeated random realizations.

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
