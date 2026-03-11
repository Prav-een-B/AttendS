"""
pipeline/table_detector.py
---------------------------
Detects table structure (rows × columns) from a binarized attendance sheet.

Strategy:
  1. Detect horizontal and vertical lines via morphological ops
  2. Find cell intersections → grid
  3. If no explicit lines found, fall back to horizontal-band segmentation
     (works for hand-ruled sheets and digitally printed sheets alike)
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Line detection via morphology
# ──────────────────────────────────────────────────────────────────────────────

def _detect_lines(binary: np.ndarray):
    """
    Extract horizontal and vertical line masks using morphological operations.
    Returns (h_mask, v_mask).
    """
    inv = cv2.bitwise_not(binary)
    h, w = inv.shape

    # Horizontal lines: erode horizontally, dilate back
    h_kernel_len = max(30, w // 15)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_mask = cv2.erode(inv, h_kernel, iterations=2)
    h_mask = cv2.dilate(h_mask, h_kernel, iterations=2)

    # Vertical lines
    v_kernel_len = max(20, h // 20)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_mask = cv2.erode(inv, v_kernel, iterations=2)
    v_mask = cv2.dilate(v_mask, v_kernel, iterations=2)

    return h_mask, v_mask


def _lines_to_positions(mask: np.ndarray, axis: int, min_gap: int = 8) -> list:
    """
    Project a line mask onto one axis and find line centre positions.
    axis=0 → horizontal lines (y positions)
    axis=1 → vertical lines  (x positions)
    """
    projection = np.sum(mask, axis=axis).astype(np.float32)
    # Normalise
    if projection.max() == 0:
        return []
    projection /= projection.max()

    # Threshold
    positions = np.where(projection > 0.3)[0]
    if len(positions) == 0:
        return []

    # Cluster close positions
    clusters = []
    group = [positions[0]]
    for p in positions[1:]:
        if p - group[-1] <= min_gap:
            group.append(p)
        else:
            clusters.append(int(np.mean(group)))
            group = [p]
    clusters.append(int(np.mean(group)))
    return clusters


# ──────────────────────────────────────────────────────────────────────────────
# Fallback: horizontal-band segmentation
# ──────────────────────────────────────────────────────────────────────────────

def _horizontal_bands(binary: np.ndarray, min_row_h: int = 15) -> list:
    """
    Find row boundaries by looking at horizontal projection of dark pixels.
    Returns list of (y_start, y_end) tuples.
    """
    inv = cv2.bitwise_not(binary)
    projection = np.sum(inv, axis=1)  # sum across columns for each row

    # Normalise
    if projection.max() == 0:
        return []
    norm = projection / projection.max()

    in_row = False
    bands = []
    start = 0
    for y, val in enumerate(norm):
        if val > 0.05 and not in_row:
            in_row = True
            start = y
        elif val <= 0.05 and in_row:
            in_row = False
            if (y - start) >= min_row_h:
                bands.append((start, y))
    if in_row and (len(norm) - start) >= min_row_h:
        bands.append((start, len(norm)))

    return bands


def _merge_small_bands(bands: list, min_h: int = 18) -> list:
    """Merge bands that are too thin (noise rows)."""
    merged = []
    i = 0
    while i < len(bands):
        s, e = bands[i]
        if (e - s) < min_h and merged:
            ps, pe = merged[-1]
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
        i += 1
    return merged


# ──────────────────────────────────────────────────────────────────────────────
# Cell extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_cells_from_grid(binary: np.ndarray, gray: np.ndarray,
                              row_ys: list, col_xs: list) -> list:
    """
    Given sorted row Y positions and column X positions,
    return a 2-D list of cell images.
    cells[row_idx][col_idx] = {'binary': ..., 'gray': ..., 'bbox': (x1,y1,x2,y2)}
    """
    cells = []
    h, w = binary.shape

    row_bounds = list(zip(row_ys[:-1], row_ys[1:]))
    col_bounds = list(zip(col_xs[:-1], col_xs[1:]))

    for (y1, y2) in row_bounds:
        row_cells = []
        for (x1, x2) in col_bounds:
            pad = 3
            cy1, cy2 = max(0, y1+pad), min(h, y2-pad)
            cx1, cx2 = max(0, x1+pad), min(w, x2-pad)
            cell_bin  = binary[cy1:cy2, cx1:cx2]
            cell_gray = gray[cy1:cy2, cx1:cx2]
            row_cells.append({
                "binary": cell_bin,
                "gray":   cell_gray,
                "bbox":   (cx1, cy1, cx2, cy2),
            })
        cells.append(row_cells)
    return cells


def _extract_cells_from_bands(binary: np.ndarray, gray: np.ndarray,
                               bands: list) -> list:
    """
    When only row bands are available (no column lines detected),
    return each full row as a single-column cell.
    The OCR module will later split within the row.
    """
    h, w = binary.shape
    cells = []
    for (y1, y2) in bands:
        pad = 2
        cy1, cy2 = max(0, y1+pad), min(h, y2-pad)
        row_bin  = binary[cy1:cy2, :]
        row_gray = gray[cy1:cy2, :]
        cells.append([{
            "binary": row_bin,
            "gray":   row_gray,
            "bbox":   (0, cy1, w, cy2),
            "full_row": True,
        }])
    return cells


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def detect_table(preprocessed: dict) -> dict:
    """
    Takes the output of preprocess.preprocess() and returns:
    {
      'cells':      2-D list of cell dicts
      'row_ys':     list of row Y boundaries (or None)
      'col_xs':     list of col X boundaries (or None)
      'mode':       'grid' | 'bands'
      'n_rows':     int
      'n_cols':     int
    }
    """
    binary = preprocessed["binary"]
    gray   = preprocessed["gray"]

    h_mask, v_mask = _detect_lines(binary)

    row_ys = _lines_to_positions(h_mask, axis=1)  # project along x → row lines
    col_xs = _lines_to_positions(v_mask, axis=0)  # project along y → col lines

    # Need at least 2 lines in each direction for a proper grid
    has_grid = len(row_ys) >= 2 and len(col_xs) >= 2

    if has_grid:
        # Add image edges if not already there
        h_img, w_img = binary.shape
        if row_ys[0]  > 10:         row_ys.insert(0, 0)
        if row_ys[-1] < h_img - 10: row_ys.append(h_img)
        if col_xs[0]  > 10:         col_xs.insert(0, 0)
        if col_xs[-1] < w_img - 10: col_xs.append(w_img)

        cells = _extract_cells_from_grid(binary, gray, row_ys, col_xs)
        mode  = "grid"
        n_rows = len(cells)
        n_cols = len(cells[0]) if cells else 0

    else:
        # Fallback: row band segmentation
        bands = _horizontal_bands(binary)
        bands = _merge_small_bands(bands)

        if not bands:
            # Last resort: treat whole image as one row
            bands = [(0, binary.shape[0])]

        cells = _extract_cells_from_bands(binary, gray, bands)
        mode  = "bands"
        n_rows = len(cells)
        n_cols = 1

    return {
        "cells":  cells,
        "row_ys": row_ys if has_grid else None,
        "col_xs": col_xs if has_grid else None,
        "mode":   mode,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "binary": binary,
        "gray":   gray,
    }
