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
python experiments/run_lshape_mc.py
python experiments/run_square_quadrature.py
```
