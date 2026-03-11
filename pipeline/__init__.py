"""
Attendance extraction pipeline — fully local, no API calls.

Components:
  preprocess     → OpenCV image cleaning & deskewing
  table_detector → Morphology-based table/row detection
  ocr_engine     → Tesseract OCR
  parser         → Structured data extraction from raw OCR
  excel_writer   → Formatted .xlsx output
"""

from .preprocess     import preprocess
from .table_detector import detect_table
from .ocr_engine     import ocr_table
from .parser         import parse_ocr_output
from .excel_writer   import write_excel


def process_image(image_path: str) -> dict:
    """
    Full pipeline: image_path → structured dict with records + meta.
    """
    preprocessed = preprocess(image_path)
    table        = detect_table(preprocessed)
    ocr_rows     = ocr_table(table)
    parsed       = parse_ocr_output(ocr_rows, mode=table["mode"])
    return parsed
