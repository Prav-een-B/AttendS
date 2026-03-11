"""
pipeline/cnn_digit_recognizer.py
----------------------------------
Integrates Praveen's Keras digit recognition model (MYmod.keras) into
the attendance sheet pipeline.

Model facts (from github.com/Prav-een-B/Recognising-Handwritten-Digits):
  Architecture : Flatten → Dense(128,ReLU) → Dense(128,ReLU) → Dense(10,softmax)
  Trained on   : MNIST dataset  (~98% accuracy)
  Input        : (28, 28) grayscale, WHITE digit on BLACK background, values /255
  Model file   : MYmod.keras  →  place in  models/MYmod.keras
"""

import os
import cv2
import numpy as np


_model = None   # module-level singleton — loaded once at first use


# ══════════════════════════════════════════════════════════════════════════════
# Model loader
# ══════════════════════════════════════════════════════════════════════════════

def load_model(model_path: str):
    """Load MYmod.keras (or any .keras/.h5) once and cache globally."""
    global _model
    if _model is not None:
        return _model

    from tensorflow import keras
    _model = keras.models.load_model(model_path)
    print(f"[CNN] Loaded  ▸  {model_path}")
    print(f"[CNN] Input   ▸  {_model.input_shape}")    # (None,28,28,1)
    print(f"[CNN] Output  ▸  {_model.output_shape}")   # (None,10)
    return _model


def get_model():
    """Return cached model, or None if not loaded yet."""
    return _model


# ══════════════════════════════════════════════════════════════════════════════
# Image preprocessing  — mirrors model.py from the repo exactly
# ══════════════════════════════════════════════════════════════════════════════

def _remove_grid_lines(cell_gray: np.ndarray) -> np.ndarray:
    """
    Erase the horizontal and vertical box borders drawn around each digit cell,
    leaving only the digit ink on a white background.
    """
    h, w = cell_gray.shape
    _, bw = cv2.threshold(cell_gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(bw)

    # --- horizontal lines (wide, 1-pixel tall streaks) ---
    hk  = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w // 5), 1))
    inv = cv2.subtract(inv, cv2.morphologyEx(inv, cv2.MORPH_OPEN, hk))

    # --- vertical lines (tall, 1-pixel wide streaks) ---
    vk  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, h // 4)))
    inv = cv2.subtract(inv, cv2.morphologyEx(inv, cv2.MORPH_OPEN, vk))

    return cv2.bitwise_not(inv)   # back to: white background, black ink


def _tight_crop(clean_bw: np.ndarray, margin: int = 3) -> np.ndarray:
    """
    Crop tightly around the digit ink blob so resizing to 28×28 fills
    the frame (same behaviour as the MNIST training images).
    """
    inv = cv2.bitwise_not(clean_bw)
    pts = np.where(inv > 0)
    if len(pts[0]) == 0:
        return clean_bw
    h, w = clean_bw.shape
    y0 = max(0,  pts[0].min() - margin)
    y1 = min(h,  pts[0].max() + margin)
    x0 = max(0,  pts[1].min() - margin)
    x1 = min(w,  pts[1].max() + margin)
    return clean_bw[y0:y1, x0:x1]


def preprocess_cell(cell_gray: np.ndarray) -> np.ndarray:
    """
    Raw digit-cell crop  →  (1, 28, 28, 1) float32 tensor for model.predict().

    Pipeline (mirrors model.py preprocessing):
      1. Remove grid-line borders
      2. Tight-crop to digit bounding box
      3. Resize  → 28 × 28
      4. Invert  → WHITE digit on BLACK background  (MNIST convention)
      5. Normalise pixel values  → [0.0, 1.0]
      6. Add batch + channel dims  → shape (1, 28, 28, 1)
    """
    clean   = _remove_grid_lines(cell_gray)   # step 1
    cropped = _tight_crop(clean)              # step 2
    if cropped.size == 0:
        cropped = clean

    # step 3
    resized = cv2.resize(cropped, (28, 28), interpolation=cv2.INTER_AREA)

    # step 4 — attendance sheet: black ink on white paper → must invert for MNIST
    if resized.mean() > 127:          # background is still white
        resized = cv2.bitwise_not(resized)

    # step 5
    arr = resized.astype(np.float32) / 255.0

    # step 6  (Flatten layer in the model handles (28,28,1) → 784 internally)
    return arr.reshape(1, 28, 28, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Single-digit prediction
# ══════════════════════════════════════════════════════════════════════════════

def predict_digit(cell_gray: np.ndarray,
                  confidence_threshold: float = 0.50) -> tuple:
    """
    Predict one digit from a raw grayscale cell image.

    Returns
    -------
    (digit_str, confidence)
        digit_str  :  '0' – '9'   or   '?'  if below confidence_threshold
        confidence :  float  0.0 – 1.0
    """
    model = get_model()
    if model is None:
        raise RuntimeError("CNN model not loaded — call load_model() first.")

    if cell_gray is None or cell_gray.size == 0:
        return '?', 0.0

    tensor = preprocess_cell(cell_gray)
    probs  = model.predict(tensor, verbose=0)[0]   # (10,)
    idx    = int(np.argmax(probs))
    conf   = float(probs[idx])

    return (str(idx) if conf >= confidence_threshold else '?'), conf


# ══════════════════════════════════════════════════════════════════════════════
# Full roll-number prediction for one sheet row
# ══════════════════════════════════════════════════════════════════════════════

def predict_roll_number(gray_image: np.ndarray,
                        row_y1: int,
                        row_y2: int,
                        digit_col_bounds: list,
                        confidence_threshold: float = 0.50) -> tuple:
    """
    Predict all digits in one row and return the assembled roll number.

    Parameters
    ----------
    gray_image         : full preprocessed grayscale sheet (H×W np.ndarray)
    row_y1, row_y2     : y-crop boundaries for this row (padding already applied)
    digit_col_bounds   : list of (x1, x2) per digit column, e.g.
                         [(14,108),(108,201),(201,295),(295,388),(388,482)]
    confidence_threshold : minimum softmax score to accept a digit (default 0.5)

    Returns
    -------
    (roll_str, digits_list, confidences_list)
        roll_str    : '22035'                    (unknown digits replaced with '?')
        digits_list : ['2','2','0','3','5']
        conf_list   : [0.99, 0.97, 0.95, 0.98, 0.96]
    """
    h_img, w_img = gray_image.shape
    digits, confs = [], []

    for (dx1, dx2) in digit_col_bounds:
        cell = gray_image[row_y1:row_y2,
                          max(0, dx1 + 3):min(w_img, dx2 - 3)]
        d, c = predict_digit(cell, confidence_threshold)
        digits.append(d)
        confs.append(round(c, 3))

    roll = ''.join(d for d in digits if d != '?')
    return roll, digits, confs


# ══════════════════════════════════════════════════════════════════════════════
# Batch — every data row in the sheet
# ══════════════════════════════════════════════════════════════════════════════

def predict_all_rolls(gray_image: np.ndarray,
                      row_ys: list,
                      digit_col_bounds: list,
                      skip_header_rows: int = 2,
                      skip_footer_rows: int = 1,
                      confidence_threshold: float = 0.50) -> list:
    """
    Run the CNN over every data row in the sheet.

    Parameters
    ----------
    gray_image          : preprocessed grayscale image  (H×W)
    row_ys              : Y-boundary list from table_detector
    digit_col_bounds    : (x1,x2) list for each digit column
    skip_header_rows    : rows to skip at top    (default 2)
    skip_footer_rows    : rows to skip at bottom (default 1)
    confidence_threshold: per-digit softmax cutoff

    Returns
    -------
    List of dicts, one per data row:
    [
      {
        'roll'        : '22035',
        'digits'      : ['2','2','0','3','5'],
        'confidences' : [0.99, 0.97, 0.95, 0.98, 0.96],
        'avg_conf'    : 0.97,
        'complete'    : True      # True when all digits were confident
      }, ...
    ]
    """
    h_img  = gray_image.shape[0]
    bounds = list(zip(row_ys[:-1], row_ys[1:]))
    data   = bounds[skip_header_rows: len(bounds) - skip_footer_rows]

    results = []
    for (y1, y2) in data:
        pad = 3
        ry1 = max(0,     y1 + pad)
        ry2 = min(h_img, y2 - pad)
        if ry2 - ry1 < 8:
            continue

        roll, digits, confs = predict_roll_number(
            gray_image, ry1, ry2, digit_col_bounds, confidence_threshold
        )

        avg_conf = round(sum(confs) / len(confs), 3) if confs else 0.0
        results.append({
            'roll':        roll,
            'digits':      digits,
            'confidences': confs,
            'avg_conf':    avg_conf,
            'complete':    len(roll) == len(digit_col_bounds),
        })

    return results
