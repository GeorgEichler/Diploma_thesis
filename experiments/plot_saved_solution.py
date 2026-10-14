"""Generate a 3D surface from final_solution.npz without rerunning a simulation.

Usage:
    python -m experiments.plot_saved_solution results/my_run/final_solution.npz
Add --png for a 600 dpi PNG alongside the default vector PDF.
"""

import argparse
from pathlib import Path

import numpy as np
from skfem import MeshTri

from afem.utils.plotting import plot_solution_surface_2d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution", type=Path, help="Saved final_solution.npz")
    parser.add_argument("--output", type=Path, help="Output filename (default: plots/final_solution_3d.pdf)")
    parser.add_argument("--pdf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--png", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--elev", type=float, default=30, help="Camera elevation in degrees")
    parser.add_argument("--azim", type=float, default=45, help="Camera azimuth in degrees")
    args = parser.parse_args()
    with np.load(args.solution, allow_pickle=False) as saved:
        p, t, u = saved["p"], saved["t"], saved["u"]
    if p.ndim != 2 or p.shape[0] != 2 or t.ndim != 2 or t.shape[0] != 3:
        parser.error("The saved solution must use a two-dimensional triangular mesh.")
    if u.shape != (p.shape[1],):
        parser.error("The saved solution must contain one P1 value per mesh vertex.")
    mesh = MeshTri(p, t)
    output = args.output or args.solution.parent / "plots" / "final_solution_3d.pdf"
    path = plot_solution_surface_2d(mesh, u, output, save_pdf=args.pdf,
                                    save_png=args.png, elev=args.elev, azim=args.azim)
    if path is not None:
        print(f"Saved surface: {path}")
        if args.pdf and args.png:
            print(f"Saved PNG: {output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
