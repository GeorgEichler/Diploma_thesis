"""Reformat saved vector mesh PDFs without rerunning FEM simulations.

Requires optional PyMuPDF. Usage:
    python -m experiments.reformat_mesh_pdfs results/my_study

Only mesh_l*.pdf files are rewritten. Other figures and numerical data are
untouched. Use --png to also regenerate PNG copies at the new panel size.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from afem.utils.comparison_layout import mesh_figure
from afem.utils.figure_export import save_figure
from afem.utils.publication_style import MESH_LINEWIDTH, paper_style


@paper_style
def reformat_mesh_pdf(filename: Path, *, save_png=False):
    """Preserve the saved mesh edges while replacing its figure layout."""
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError("Reformatting existing PDFs requires PyMuPDF: pip install pymupdf") from exc

    filename = Path(filename)
    with pymupdf.open(filename) as document:
        if len(document) != 1:
            raise ValueError(f"Expected a single-page mesh PDF: {filename}")
        drawings = document[0].get_drawings()
        strokes = [p for p in drawings if p["type"] == "s" and p.get("color") is not None
                   and p["items"] and all(item[0] == "l" for item in p["items"])]
        if not strokes:
            raise ValueError(f"No vector mesh edges found in {filename}")
        # triplot's mesh is the longest line path; its colour distinguishes
        # it from the axis spines and ticks in the original PDFs.
        mesh_colour = max(strokes, key=lambda p: len(p["items"]))["color"]
        segments = np.array([
            [tuple(item[1]), tuple(item[2])]
            for path in strokes if path["color"] == mesh_colour
            for item in path["items"]
        ])
    # PDF coordinates increase downwards; Matplotlib coordinates increase upwards.
    segments[:, :, 1] *= -1
    bounds_min = segments.min(axis=(0, 1))
    bounds_max = segments.max(axis=(0, 1))
    if np.any(bounds_max <= bounds_min):
        raise ValueError(f"Degenerate mesh bounds in {filename}")
    fig, ax = mesh_figure()
    ax.add_collection(LineCollection(segments, colors=[mesh_colour], linewidths=MESH_LINEWIDTH))
    ax.set_xlim(bounds_min[0], bounds_max[0])
    ax.set_ylim(bounds_min[1], bounds_max[1])
    temporary = filename.with_name(f".{filename.stem}.mesh_panel.pdf")
    try:
        save_figure(fig, temporary, save_png=save_png)
        temporary.replace(filename)
        if save_png:
            temporary.with_suffix(".png").replace(filename.with_suffix(".png"))
    finally:
        plt.close(fig)
    return len(segments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+", help="Folders containing saved mesh PDFs.")
    parser.add_argument("--png", action="store_true", help="Also regenerate 600 dpi PNG copies.")
    args = parser.parse_args()
    for directory in args.directories:
        paths = sorted(directory.rglob("mesh_l*.pdf"))
        if not paths:
            raise FileNotFoundError(f"No mesh_l*.pdf files found in {directory}")
        for path in paths:
            count = reformat_mesh_pdf(path, save_png=args.png)
            print(f"Reformatted {path} ({count} mesh edges)", flush=True)
