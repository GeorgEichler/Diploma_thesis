"""Publication exports shared by all plotting functions."""

from pathlib import Path

import matplotlib as mpl


RASTER_DPI = 600


def save_figure(fig, filename: str | Path, *, save_pdf: bool = True,
                save_png: bool = False, dpi: int = RASTER_DPI) -> Path | None:
    """Export enabled formats; return PDF, otherwise PNG, or None if disabled.

    Keep the full canvas so matching comparison layouts stay aligned in LaTeX.
    DPI controls raster output and any rasterized artists inside vector files.
    The filename supplies the base name; even a legacy PNG filename defaults
    to PDF only. Each format can be enabled independently.
    """
    if not save_pdf and not save_png:
        return None
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = filename.with_suffix(".pdf")
    with mpl.rc_context({"pdf.fonttype": 42, "savefig.bbox": None}):
        if save_pdf:
            fig.savefig(pdf_path, dpi=dpi, facecolor="white", transparent=False)
        if save_png:
            fig.savefig(filename.with_suffix(".png"), dpi=dpi,
                        facecolor="white", transparent=False)
    return pdf_path if save_pdf else filename.with_suffix(".png")
