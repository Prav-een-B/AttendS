"""
pipeline/ocr_engine.py
-----------------------
Tesseract-based OCR with cell-aware configurations.

For each cell we:
  1. Pre-clean the cell image (morphological clean, padding)
  2. Choose the right Tesseract PSM based on expected content
  3. Run OCR
  4. Return raw text per cell
"""

import re
import cv2
import numpy as np
import pytesseract
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Tesseract config strings
# ──────────────────────────────────────────────────────────────────────────────

# PSM 7  = treat image as a single text line
# PSM 6  = assume a uniform block of text
# PSM 8  = treat image as a single word
# PSM 13 = raw line, no OSD

CFG_LINE  = "--psm 7 --oem 3"
CFG_BLOCK = "--psm 6 --oem 3"
CFG_WORD  = "--psm 8 --oem 3"
CFG_DIGITS = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789-/"


# ──────────────────────────────────────────────────────────────────────────────
# Cell image preparation
# ──────────────────────────────────────────────────────────────────────────────

def _prep_cell(cell_img: np.ndarray, padding: int = 8) -> Image.Image:
    """
    Clean and pad a cell binary image before passing to Tesseract.
    Returns a PIL Image.
    """
    if cell_img is None or cell_img.size == 0:
        return Image.new("L", (64, 32), 255)

    # Make sure it's grayscale
    if cell_img.ndim == 3:
        cell_img = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

    # Upscale tiny cells
    h, w = cell_img.shape
    if h < 30 or w < 30:
        scale = max(30 / h, 30 / w, 1.0)
        cell_img = cv2.resize(cell_img, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_CUBIC)

    # Threshold (in case we received grayscale)
    _, cell_img = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Slight morphological close to fill gaps in handwriting
    kernel = np.ones((2, 2), np.uint8)
    cell_img = cv2.morphologyEx(cell_img, cv2.MORPH_CLOSE, kernel)

    # Add white padding
    padded = cv2.copyMakeBorder(cell_img, padding, padding, padding, padding,
                                cv2.BORDER_CONSTANT, value=255)

    return Image.fromarray(padded)


# ──────────────────────────────────────────────────────────────────────────────
# Single cell OCR
# ──────────────────────────────────────────────────────────────────────────────

def ocr_cell(cell_img: np.ndarray, config: str = CFG_LINE) -> str:
    """Run Tesseract on a single cell image, return stripped text."""
    pil = _prep_cell(cell_img)
    try:
        text = pytesseract.image_to_string(pil, config=config)
    except Exception:
        text = ""
    return text.strip()


def ocr_cell_digits(cell_img: np.ndarray) -> str:
    """OCR optimised for numeric roll numbers."""
    return ocr_cell(cell_img, config=CFG_DIGITS)


# ──────────────────────────────────────────────────────────────────────────────
# Full-row OCR (bands mode)
# ──────────────────────────────────────────────────────────────────────────────

def ocr_row(row_img: np.ndarray) -> str:
    """OCR a full row image as a block of text."""
    return ocr_cell(row_img, config=CFG_BLOCK)


# ──────────────────────────────────────────────────────────────────────────────
# Table OCR dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def ocr_table(table: dict) -> list:
    """
    Run OCR on the full table returned by detect_table().

    Returns a list of rows, each row is a list of cell text strings.
    e.g. [['1', 'John Smith', 'P'], ['2', 'Jane Doe', 'A'], ...]
    """
    cells  = table["cells"]
    mode   = table["mode"]
    n_cols = table["n_cols"]

    results = []

    for row_cells in cells:
        row_texts = []
        for cell in row_cells:
            img = cell.get("gray") if cell.get("gray") is not None else cell.get("binary")
            if cell.get("full_row"):
                text = ocr_row(img)
            else:
                text = ocr_cell(img)
            row_texts.append(text)
        results.append(row_texts)

    return results
