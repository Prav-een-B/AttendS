import os
import cv2
import numpy as np

_model = None


def load_model(model_path: str):
    global _model
    if _model is not None:
        return _model
    from tensorflow import keras
    _model = keras.models.load_model(model_path)
    print(f"[CNN] Loaded  ▸  {model_path}")
    print(f"[CNN] Input   ▸  {_model.input_shape}")
    print(f"[CNN] Output  ▸  {_model.output_shape}")
    return _model


def get_model():
    return _model


def _remove_grid_lines(cell_gray: np.ndarray) -> np.ndarray:
    """Strip box borders so only digit ink remains."""
    h, w = cell_gray.shape
    _, bw = cv2.threshold(cell_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(bw)
    hk  = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w // 5), 1))
    inv = cv2.subtract(inv, cv2.morphologyEx(inv, cv2.MORPH_OPEN, hk))
    vk  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, h // 4)))
    inv = cv2.subtract(inv, cv2.morphologyEx(inv, cv2.MORPH_OPEN, vk))
    return cv2.bitwise_not(inv)  # white bg, black ink


def preprocess_cell(cell_gray: np.ndarray) -> np.ndarray:
    """
    Raw digit-cell crop → (1, 28, 28, 1) float32 tensor for model.predict()

    Pipeline:
      1. OTSU threshold  → binary (255=bg, 0=ink)
      2. Remove grid-line borders on binary image
      3. Invert          → (255=ink, 0=bg)
      4. Tight-crop to digit bounding box
      5. Resize → 28×28
      6. Ensure white digit on black bg (MNIST format)
      7. Normalise → [0.0, 1.0]
      8. Reshape → (1, 28, 28, 1)
    """
    if cell_gray is None or cell_gray.size == 0:
        return np.zeros((1, 28, 28, 1), dtype=np.float32)

    # Step 1 — binarize
    _, bw = cv2.threshold(cell_gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 2 — remove grid lines on inverted image (ink=255)
    inv = cv2.bitwise_not(bw)
    h, w = inv.shape
    hk  = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8, w // 5), 1))
    inv = cv2.subtract(inv, cv2.morphologyEx(inv, cv2.MORPH_OPEN, hk))
    vk  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(5, h // 4)))
    inv = cv2.subtract(inv, cv2.morphologyEx(inv, cv2.MORPH_OPEN, vk))
    # inv: 255=digit ink, 0=clean background

    # Step 3 — tight crop around digit ink
    pts = np.where(inv > 0)
    if len(pts[0]) > 10:
        margin = 4
        y0 = max(0, pts[0].min() - margin);  y1 = min(h, pts[0].max() + margin)
        x0 = max(0, pts[1].min() - margin);  x1 = min(w, pts[1].max() + margin)
        inv = inv[y0:y1, x0:x1]

    if inv.size == 0:
        inv = cv2.bitwise_not(bw)  # fallback: full cell

    # Step 4 — resize to 28×28
    resized = cv2.resize(inv, (28, 28), interpolation=cv2.INTER_AREA)

    # Step 5 — at this point resized has white digit on black background (MNIST format)
    # The inv pipeline already produces this correctly. No re-inversion needed.

    # Step 6 — normalise
    arr = resized.astype(np.float32) / 255.0

    # Step 7 — shape (1, 28, 28, 1)
    return arr.reshape(1, 28, 28, 1)


def predict_digit(cell_gray: np.ndarray,
                  confidence_threshold: float = 0.40) -> tuple:
    """
    Predict one digit from a raw grayscale cell crop.
    Returns (digit_str, confidence).
    """
    model = get_model()
    if model is None:
        raise RuntimeError("CNN model not loaded — call load_model() first.")
    if cell_gray is None or cell_gray.size == 0:
        return '?', 0.0

    tensor = preprocess_cell(cell_gray)
    probs  = model.predict(tensor, verbose=0)[0]
    idx    = int(np.argmax(probs))
    conf   = float(probs[idx])

    return (str(idx) if conf >= confidence_threshold else '?'), conf


def predict_roll_number(gray_image: np.ndarray,
                        row_y1: int, row_y2: int,
                        digit_col_bounds: list,
                        confidence_threshold: float = 0.40) -> tuple:
    """
    Predict all digits in one row → assembled roll number string.
    Returns (roll_str, digits_list, confidences_list).
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


def predict_all_rolls(gray_image: np.ndarray,
                      row_ys: list,
                      digit_col_bounds: list,
                      skip_header_rows: int = 2,
                      skip_footer_rows: int = 1,
                      confidence_threshold: float = 0.40) -> list:
    """
    Run CNN over every data row. Returns list of dicts with roll + confidence.
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
