"""
pipeline/pdf_converter.py
--------------------------
Converts PDF files to a list of image paths using pdf2image (poppler).

Each page becomes a separate PNG in a temp directory.
The caller is responsible for cleaning up temp files.

Supports:
  - Single-page PDFs  (most attendance sheets)
  - Multi-page PDFs   (returns one image per page)
"""

import os
import tempfile
from pathlib import Path


# ── Try pdf2image first, fall back to pdftoppm subprocess ────────────────────

def _convert_with_pdf2image(pdf_path: str, dpi: int, output_dir: str) -> list:
    from pdf2image import convert_from_path
    pages = convert_from_path(pdf_path, dpi=dpi, output_folder=output_dir,
                               fmt="png", output_file="page")
    paths = []
    for i, page in enumerate(pages):
        out = os.path.join(output_dir, f"page_{i+1:03d}.png")
        page.save(out, "PNG")
        paths.append(out)
    return paths


def _convert_with_pdftoppm(pdf_path: str, dpi: int, output_dir: str) -> list:
    import subprocess, glob
    prefix = os.path.join(output_dir, "page")
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-png", pdf_path, prefix],
        check=True, capture_output=True,
    )
    paths = sorted(glob.glob(f"{prefix}-*.png") + glob.glob(f"{prefix}*.png"))
    return paths


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list:
    """
    Convert every page of a PDF to a PNG image.

    Parameters
    ----------
    pdf_path : str   — path to the .pdf file
    dpi      : int   — resolution (300 recommended for OCR)

    Returns
    -------
    list of str  — absolute paths to the generated PNG files (one per page).
                   Files live in a system temp directory; caller should delete
                   them after use (see cleanup_images()).
    """
    tmp_dir = tempfile.mkdtemp(prefix="attendx_pdf_")

    try:
        paths = _convert_with_pdf2image(pdf_path, dpi, tmp_dir)
    except Exception:
        try:
            paths = _convert_with_pdftoppm(pdf_path, dpi, tmp_dir)
        except Exception as e:
            raise RuntimeError(
                f"Could not convert PDF to images. "
                f"Make sure poppler-utils is installed.\nError: {e}"
            )

    if not paths:
        raise RuntimeError(f"No pages extracted from {pdf_path}")

    return paths


def cleanup_images(image_paths: list):
    """Delete temp PNG files created by pdf_to_images()."""
    dirs_seen = set()
    for p in image_paths:
        try:
            os.remove(p)
            dirs_seen.add(os.path.dirname(p))
        except OSError:
            pass
    for d in dirs_seen:
        try:
            os.rmdir(d)   # only removes if empty
        except OSError:
            pass


def is_pdf(filename: str) -> bool:
    return filename.lower().endswith(".pdf")
