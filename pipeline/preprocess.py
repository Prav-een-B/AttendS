"""
pipeline/preprocess.py
-----------------------
Image preprocessing for attendance sheet OCR.

Steps:
1. Load & convert to grayscale
2. Deskew (correct rotation)
3. Denoise
4. Adaptive threshold → clean binary image
5. Detect and return both the processed image and page crop
"""

import cv2
import numpy as np
from scipy.ndimage import interpolation as inter


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_gray(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    return gray, img


def _resize_if_needed(img: np.ndarray, max_dim: int = 3000) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Estimate and correct skew using projection profile method.
    Works well for ±15 degree tilts.
    """
    # Threshold to get foreground
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find all non-zero points
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 100:
        return gray  # not enough content to estimate skew

    # Minimum area rectangle of text blobs
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90

    # Only correct if skew is meaningful but not extreme
    if abs(angle) < 0.3 or abs(angle) > 20:
        return gray

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def _denoise(gray: np.ndarray) -> np.ndarray:
    """Apply fast non-local means denoising."""
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive thresholding handles uneven illumination (camera photos).
    Falls back to Otsu for very clean scans.
    """
    # Check if image is already clean (std dev of pixel values)
    std = np.std(gray)
    if std < 30:
        # Very low contrast – just Otsu
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Adaptive works best for photos
        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=31,
            C=11,
        )
    return binary


def _auto_crop(binary: np.ndarray, gray: np.ndarray) -> tuple:
    """
    Detect the document / sheet boundary and crop to it.
    Returns (cropped_binary, cropped_gray, crop_rect)
    """
    # Find external contours on inverted binary
    inv = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return binary, gray, None

    h, w = binary.shape
    # Pick the largest contour that is a reasonable fraction of the image
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 0.05 * h * w and area > best_area:
            best_area = area
            best = c

    if best is None:
        return binary, gray, None

    x, y, cw, ch = cv2.boundingRect(best)
    # Add small padding
    pad = 10
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + cw + pad), min(h, y + ch + pad)

    return binary[y1:y2, x1:x2], gray[y1:y2, x1:x2], (x1, y1, x2, y2)


def _upscale_for_ocr(binary: np.ndarray, gray: np.ndarray, target_dpi_equiv: int = 300) -> tuple:
    """
    If the image is low-res, upscale it so Tesseract performs better.
    Heuristic: aim for at least 2000px on the longer edge.
    """
    h, w = binary.shape
    if max(h, w) < 1800:
        scale = 1800 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        binary = cv2.resize(binary, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        gray   = cv2.resize(gray,   (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return binary, gray


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(image_path: str) -> dict:
    """
    Full preprocessing pipeline.

    Returns dict with:
      - 'binary'      : clean binarized image (np.ndarray uint8)
      - 'gray'        : denoised grayscale     (np.ndarray uint8)
      - 'original_bgr': original color image   (np.ndarray uint8)
      - 'crop_rect'   : (x1,y1,x2,y2) or None
    """
    gray_orig, bgr_orig = _load_gray(image_path)

    # Resize large images first
    gray_orig = _resize_if_needed(gray_orig, 3000)
    bgr_orig  = _resize_if_needed(bgr_orig,  3000)

    # Deskew
    gray_desk = _deskew(gray_orig)

    # Denoise
    gray_clean = _denoise(gray_desk)

    # Binarize
    binary = _binarize(gray_clean)

    # Auto-crop to sheet
    binary, gray_clean, crop_rect = _auto_crop(binary, gray_clean)

    # Upscale for OCR if needed
    binary, gray_clean = _upscale_for_ocr(binary, gray_clean)

    return {
        "binary":       binary,
        "gray":         gray_clean,
        "original_bgr": bgr_orig,
        "crop_rect":    crop_rect,
    }
