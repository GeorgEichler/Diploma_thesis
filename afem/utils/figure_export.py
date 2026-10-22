"""Publication exports shared by all plotting functions."""

import logging
from pathlib import Path

import matplotlib as mpl

RASTER_DPI = 600


class _FontTimestampFilter(logging.Filter):
    """Hide known Computer Modern timestamp messages, preserving other warnings."""

    def filter(self, record):
        return record.getMessage() not in {
            "'created' timestamp seems very low; regarding as unix timestamp",
            "'modified' timestamp seems very low; regarding as unix timestamp",
        }


def save_figure(fig, filename: str | Path, *, save_pdf: bool = True,
                save_png: bool = False, dpi: int = RASTER_DPI) -> Path | None:
    """Export enabled formats; return PDF, otherwise PNG, or None if disabled."""
    if not save_pdf and not save_png:
        return None
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = filename.with_suffix(".pdf")
    font_logger = logging.getLogger("fontTools.ttLib.tables._h_e_a_d")
    timestamp_filter = _FontTimestampFilter()
    font_logger.addFilter(timestamp_filter)
    try:
        with mpl.rc_context({"pdf.fonttype": 42, "savefig.bbox": None}):
            if save_pdf:
                fig.savefig(pdf_path, dpi=dpi, facecolor="white", transparent=False)
            if save_png:
                fig.savefig(filename.with_suffix(".png"), dpi=dpi,
                            facecolor="white", transparent=False)
    finally:
        font_logger.removeFilter(timestamp_filter)
    return pdf_path if save_pdf else filename.with_suffix(".png")
