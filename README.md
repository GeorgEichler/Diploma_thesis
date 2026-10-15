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

## Figures for papers

Mesh figures use separate settings in `afem/utils/publication_style.py`:
`MESH_PANEL_WIDTH_CM=3.2`, `MESH_PANEL_HEIGHT_CM=3.2`, and
`MESH_LINEWIDTH=0.15` pt. They have no axes, tick labels, or titles and
use a small margin, allowing four panels per row within a 13.7 cm text
width. Put refinement levels and element counts in LaTeX subcaptions
and include each mesh PDF with `width=3.2cm`.

Existing vector mesh PDFs can be reformatted without rerunning the
simulations. The optional utility requires PyMuPDF (`pip install pymupdf`):

```bash
python -m experiments.reformat_mesh_pdfs results/my_study
```

It rewrites only `mesh_l*.pdf` files, using their saved vector mesh
edges. Add `--png` to regenerate PNG copies as well. Numerical results
and other figures are preserved. The following comparison and error
figure settings retain their separate 6.5 cm panel size.

The refinement comparison uses matching `PLOT_XLIM` and `PLOT_YLIM`
settings in `experiments/run_refinement_comparison.py` for panels shown
side by side. The current ranges are 10 to 100,000 DOFs and relative
errors from 0.0001 to 1. Set either value to `None` for automatic scaling,
or adjust it to include the error range of another study.

All plotting functions save only a vector PDF by default.
`AFEMConfig` controls both formats independently: `save_pdf=True` and
`save_png=False`. Disabling both saves no plot files.
Optional PNG exports use 600 dpi. PDF exports embed TrueType fonts and preserve
vector curves and text; their quality does not depend on the PNG pixel count.
Export defaults live in `afem/utils/figure_export.py`.
The exporter suppresses the known FontTools `created`/`modified` timestamp
messages from Computer Modern font metadata while saving. Other font
warnings remain visible, and logging outside export is unchanged.

Figures are designed at their printed size: each panel is 6.5 cm wide
and 6.5 cm high, fitting two panels within a 13.7 cm text width. Shared
settings in `afem/utils/publication_style.py` use Computer Modern serif
text and mathematics, 10 pt axis labels, 8.5 pt ticks, thin curves, inward
ticks on all four sides, and a pale background without grid lines.
This follows the visual style of Figures 1 and 2 of Storn's
*Randomized Projection Operators onto Piecewise Polynomial Spaces*.
The fonts ship with Matplotlib; plotting does not require a LaTeX installation.
Adjust `PANEL_WIDTH_CM`, `PANEL_HEIGHT_CM`, and `PAPER_RC` there for another
manuscript layout. Settings apply locally and do not change the caller's style.

Use the PDF files in LaTeX, for example with `\usepackage{graphicx}`:

```latex
\includegraphics[width=6.5cm]{results/mc_sample_comparison/sample_errors.pdf}
\hfill
\includegraphics[width=6.5cm]{results/mc_ensemble/ensemble_errors.pdf}
```

Both Monte Carlo comparison types keep identical canvas dimensions and
data axes. The sample color key sits below the axes; the ensemble uses
that space for its individual-run and spread legend. Exports retain the
full canvas so placing them at equal widths preserves alignment. Include
them at 6.5 cm to preserve the designed font sizes. Enlarging or shrinking
the PDF scales its text and line widths too.
`results/paper_figures/comparison_preview.tex` demonstrates both pairs
inside a document with a 13.7 cm text width.
To regenerate the comparison figures from existing results without solving:

```bash
python -m experiments.run_mc_sample_comparison --plot-only
python -m experiments.run_mc_ensemble --plot-only
```

To also generate PNG copies, add `--png` to any comparison command,
including when using `--plot-only`:

```bash
python -m experiments.run_mc_sample_comparison --plot-only --png
python -m experiments.run_mc_ensemble --plot-only --png
python -m experiments.run_load_comparison --plot-only --png
python -m experiments.run_refinement_comparison --plot-only --png
```

For other experiments, set `save_png=True` in their `AFEMConfig` to apply
the option to mesh, solution, error-history, and estimator plots. Direct
plotting calls accept both format switches:

```python
plot_history(history, "estimator.pdf", save_png=True)
plot_history(history, "estimator.pdf", save_pdf=False, save_png=True)  # PNG only
```

Comparison commands also accept `--pdf` / `--no-pdf` and `--png` / `--no-png`:

```bash
python -m experiments.run_mc_ensemble --plot-only --no-pdf --png
```

Plotting functions return PDF paths when PDF is enabled, otherwise PNG
paths when PNG is enabled. With both disabled, a single-plot function
returns `None` and a multi-plot function returns an empty tuple. Previously
saved files are retained when disabling a format.

## Initial meshes

`AFEMConfig.initial_mesh="default"` uses the existing scikit-fem mesh for
the selected domain. For an L-shaped domain, choose:

```python
domain="lshape",
initial_mesh="lshape_four_triangles",
initial_refinements=0,
```

This base mesh consists of the three unit squares `[-1,0]^2`,
`[-1,0] x [0,1]`, and `[0,1] x [-1,0]`. Each square's centre connects
to its four corners, giving 12 triangles and 11 vertices in total.
Shared square edges match, so the mesh is conforming.
`initial_refinements` is applied afterwards: values 0, 1, and 2 give
12, 48, and 192 triangles respectively before the first solve.
This option works for both adaptive and uniform runs and is recorded
in `history.json`. It requires `domain="lshape"`.

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

## Monte Carlo sample-count study

Set `SAMPLE_COUNTS = [1, 2, 5, 10, 20]`, `RHS`, and `CONFIG` in
`experiments/run_mc_sample_comparison.py`, then run:

```bash
python -m experiments.run_mc_sample_comparison
python -m experiments.run_mc_sample_comparison --samples 1 5 20 100
python -m experiments.run_mc_sample_comparison --plot-only
```

Each array entry runs a separate AFEM experiment with that fixed number of
samples per element at every level. Runs execute in array order and save before
the next starts. All use the same RHS, initial mesh, refinement settings, and
reference settings. `CONFIG.refinement_strategy="uniform"` is also supported.

The study creates PDFs in `results/mc_sample_comparison`: `sample_errors`
and `sample_estimator`. Both use progressively darker shades of seaborn's
`crest` colormap for larger sample
counts, identified by a continuous logarithmic colorbar with ticks at the actual
sample counts (up to eight labels). The error plot has solid H1 and dashed
L2 curves and a "relative error" axis. Values remain relative H1 seminorm and L2
errors, despite the shortened H1 label. Curves are plotted at each run's own DOFs;
they are not averaged or interpolated. No titles, solution-field images,
mesh images, or per-run convergence plots are generated.

`samples_<count>/history.json` saves configuration, reference data, and history;
`final_solution.npz` stores the final mesh and solution arrays. The manifest
`sample_comparison.json` lists completed runs, allowing plotting after an
interruption and excluding stale run folders. Seeds use
`base + sample_count * max_iterations + level`, giving disjoint level-seed ranges
and reproducibility independent of list order. Rerunning overwrites the current
study; use a different output directory to retain earlier results.

This is one random realization per sample count, not an ensemble average.
Each count also uses its own enriched final-mesh reference; neither monotone
errors with increasing samples nor a common exact reference are assumed.

## Multiple Monte Carlo realizations

Edit `RHS`, `CONFIG`, and `NUM_RUNS` in `experiments/run_mc_ensemble.py` or run:

```bash
python -m experiments.run_mc_ensemble --runs 20
python -m experiments.run_mc_ensemble --plot-only --summary mean --no-show-individual
```

The first command runs AFEM sequentially with Monte Carlo loads. Each run's
history, configuration, reference data, and final mesh/solution arrays are
saved in `results/mc_ensemble/run_001/`, `run_002/`, etc., before starting the
next. `ensemble.json` records the base seed and completed runs. Seeds follow
`base_seed + run_index * max_iterations + level`, avoiding reused seeds across
different runs and levels. The ensemble is reproducible for the same settings.

Two PDFs are saved: `ensemble_errors` (with the axis labelled
"relative error") and `ensemble_estimator`. Options can be set in the file
or on the command line:

- `--summary median`, `mean`, or `none` selects the central curve.
- `--show-individual` / `--no-show-individual` controls faint individual curves.
- `--show-band` / `--no-show-band` controls the pointwise 25–75% spread band.
- `--save-final-meshes` optionally saves one final mesh image per run. No
  solution-field images, intermediate meshes, or per-run convergence plots
  are generated. Final numerical mesh/solution arrays are always saved.
- `--plot-only` changes curve presentation without solving, and can plot the
  completed runs of an interrupted ensemble. It does not regenerate mesh images.

Adaptive meshes differ between runs. To summarize errors at comparable sizes,
we interpolate each run's error/estimator linearly against log(DOFs) on 100
log-spaced points in the range reached by **every** run, without extrapolation.
For repeated DOF counts the last observation is used for statistics; individual
curves retain all observations. A single common DOF count is also supported.
Means, medians, and quartiles are saved to `ensemble_statistics.json`.
The band describes variation between runs, not uncertainty in the mean.
Zeros remain in saved data but are omitted on logarithmic plots.

These are averages of convergence curves, not averages of solution fields.
Error values remain relative H1 seminorm and L2 errors despite shortened plot
labels. Each realization uses its own enriched final-mesh reference; reference
integration errors can affect the apparent spread. Uniform refinement is also
available through `CONFIG.refinement_strategy`, using fewer levels.

Rerunning in the same output folder replaces the ensemble manifest and run
results. Only runs in the new manifest are included, even if old run folders
remain. Choose a new `CONFIG.output_dir` to preserve an earlier experiment.

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
`<method>/refinement_errors_comparison.pdf` uses a color for each strategy,
solid H1 and dashed L2 curves against DOFs. These are relative H1 seminorm and
L2 errors despite the shortened labels. Each strategy uses its own enriched
final-mesh reference, not a common reference solution.

Replot without solving using the same selected method:

```bash
python -m experiments.run_refinement_comparison --method quadrature --plot-only
```

Both running and replotting also save `<method>/refinement_estimators_comparison.pdf`,
overlaying the adaptive and uniform residual estimators against DOFs. Estimator
plots share the error plot's x limits and use automatic y limits.
Use `--output-dir PATH` to select an existing comparison directory without editing
the script, for example:

```bash
python -m experiments.run_refinement_comparison --method quadrature --plot-only --output-dir results/refinement_comparison_quadrature_lshape_one_initial_refinement
```

Different methods use separate folders; rerunning the same method overwrites
its results. Choose another `CONFIG.output_dir` to retain earlier runs.

Generate a 3D surface from a saved final P1 solution without solving again:

```bash
python -m experiments.plot_saved_solution results/refinement_comparison_Monte_Carlo_lshape_one_initial_refinement/monte_carlo/adaptive/final_solution.npz
```

This saves `adaptive/plots/final_solution_3d.pdf` as a vector figure using the
saved mesh connectivity, preserving the L-shaped domain. Add `--png` for a
600 dpi PNG, `--no-pdf` to disable PDF export, or `--output` to choose another
filename. `--elev` and `--azim` set the camera angles in degrees.

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

The combined `relative_errors_comparison.pdf` uses one color per
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

Both running and replotting also save `estimators_comparison.pdf`, overlaying
the three methods' residual estimators against DOFs with the same colors and
markers as the error comparison. All saved levels are included. Estimators are
the recorded values for each method's load approximation and integration rule;
they are not normalized errors against a common reference.
Use `--output-dir PATH` to replot another saved experiment:

```bash
python -m experiments.run_load_comparison --plot-only --output-dir results/load_comparison_low_quadrature_high_oscillation
```

Both comparison scripts honor `--png` and `--no-pdf` for estimator plots too.

In `run_load_comparison.py`, configure the relative-error slope guide:

```python
REFERENCE_RATES = (1.0 / 2.0,)
REFERENCE_STARTS = ((300.0, 0.1),)  # (ndof, relative error)
```

The guide follows `error = 0.1 * (ndof / 300)**(-1/2)` from the start point
to the largest plotted DOF. Use `None` for automatic placement or rates `()`
to hide guides. Estimator plots have no slope guides. Replot with `--plot-only`
after changing these settings.

Rerunning the experiment overwrites that output directory's method results;
choose a different `CONFIG.output_dir` to retain a previous experiment.
