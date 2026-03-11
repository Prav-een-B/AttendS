"""
pipeline/parser.py
-------------------
Transforms raw OCR text (list of rows × cols) into structured attendance records.

Strategy:
  1. Detect which columns correspond to: serial/roll, name, status, date(s)
  2. Parse each row into a structured dict
  3. Handle both grid mode (multi-column) and band mode (full-row text)
"""

import re
from typing import List, Dict, Optional


# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ──────────────────────────────────────────────────────────────────────────────

RE_ROLL      = re.compile(r'\b([A-Z]{0,4}\d{2,6}[A-Z0-9]{0,4})\b', re.IGNORECASE)
RE_SERIAL    = re.compile(r'^\s*(\d{1,4})\s*$')
RE_STATUS_P  = re.compile(r'\b(P|present|✓|✔|/|1|yes|Y)\b', re.IGNORECASE)
RE_STATUS_A  = re.compile(r'\b(A|absent|✗|✘|x|X|0|no|N)\b', re.IGNORECASE)
RE_STATUS_L  = re.compile(r'\b(L|late|lb|leave)\b', re.IGNORECASE)
RE_NAME      = re.compile(r"^[A-Za-z][A-Za-z .' -]{3,}$")

HEADER_WORDS = {'name', 'student', 'roll', 'no', 'sr', 'sno', 's.no', 'serial',
                'roll no', 'enrollment', 'date', 'status', 'present', 'absent',
                '#', 'sl', 'sl.no', 'reg', 'registration', 'attendance', 'marks'}


# ──────────────────────────────────────────────────────────────────────────────
# Column role detection
# ──────────────────────────────────────────────────────────────────────────────

def _is_header_row(row: List[str]) -> bool:
    row_lower = ' '.join(row).lower()
    hits = sum(1 for w in HEADER_WORDS if w in row_lower)
    return hits >= 1


def _detect_column_roles(rows: List[List[str]]) -> Dict[str, Optional[int]]:
    """
    Heuristically map column indices to roles: serial, roll, name, status.
    Returns dict like {'serial': 0, 'roll': 1, 'name': 2, 'status': 3}
    """
    if not rows:
        return {}

    n_cols = max(len(r) for r in rows)
    if n_cols == 0:
        return {}

    # Score columns
    scores = {role: [0] * n_cols for role in ('serial', 'roll', 'name', 'status')}

    data_rows = [r for r in rows if not _is_header_row(r)][:30]

    for row in data_rows:
        for c, cell in enumerate(row):
            cell = cell.strip()
            if not cell:
                continue
            if RE_SERIAL.match(cell):
                scores['serial'][c] += 1
            if RE_ROLL.search(cell):
                scores['roll'][c] += 2
            if RE_NAME.match(cell):
                scores['name'][c] += 2
            if RE_STATUS_P.search(cell) or RE_STATUS_A.search(cell):
                scores['status'][c] += 2

    def best_col(role):
        col_scores = scores[role]
        best = max(range(n_cols), key=lambda i: col_scores[i])
        return best if col_scores[best] > 0 else None

    roles = {
        'serial': best_col('serial'),
        'roll':   best_col('roll'),
        'name':   best_col('name'),
        'status': best_col('status'),
    }

    # Disambiguate: serial and roll shouldn't share a column
    if roles['serial'] == roles['roll'] and roles['serial'] is not None:
        # Prefer roll
        roles['serial'] = None

    return roles


# ──────────────────────────────────────────────────────────────────────────────
# Row parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_status(text: str) -> str:
    t = text.strip()
    if RE_STATUS_P.search(t): return "Present"
    if RE_STATUS_A.search(t): return "Absent"
    if RE_STATUS_L.search(t): return "Late"
    return "Unknown"


def _clean_name(text: str) -> str:
    name = re.sub(r'[^A-Za-z .\'\-]', '', text).strip()
    # Title-case if all uppercase
    if name == name.upper():
        name = name.title()
    return name


def _clean_roll(text: str) -> str:
    # Remove stray spaces and common OCR noise
    roll = text.strip().replace(' ', '')
    roll = re.sub(r'[oO]', '0', roll)  # O→0 in numeric contexts
    return roll


def _parse_row_grid(row: List[str], roles: Dict) -> Optional[Dict]:
    """Parse a single multi-column row using detected column roles."""
    roll   = row[roles['roll']].strip()   if roles.get('roll')   is not None and roles['roll']   < len(row) else ""
    name   = row[roles['name']].strip()   if roles.get('name')   is not None and roles['name']   < len(row) else ""
    status = row[roles['status']].strip() if roles.get('status') is not None and roles['status'] < len(row) else ""
    serial = row[roles['serial']].strip() if roles.get('serial') is not None and roles['serial'] < len(row) else ""

    roll_clean = _clean_roll(roll)
    name_clean = _clean_name(name)

    # Skip empty or header-like rows
    if not roll_clean and not name_clean:
        return None
    if _is_header_row([roll, name, status, serial]):
        return None

    return {
        "serial":      serial,
        "roll_number": roll_clean,
        "name":        name_clean,
        "status":      _parse_status(status) if status else "Unknown",
        "raw":         " | ".join(row),
    }


def _parse_row_band(row_text: str) -> Optional[Dict]:
    """
    Parse a full-row OCR string when no column structure was detected.
    Tries to extract roll, name, status from free text.
    """
    text = row_text.strip()
    if not text or len(text) < 2:
        return None
    if _is_header_row([text]):
        return None

    # Find roll number
    roll_match = RE_ROLL.search(text)
    roll = roll_match.group(1) if roll_match else ""

    # Find name: longest sequence of title-case words
    words = text.split()
    name_words = [w for w in words if re.match(r'^[A-Z][a-z]+$', w)]
    name = ' '.join(name_words) if name_words else ""
    if not name:
        # Try any alpha sequence 
        alpha = re.findall(r'[A-Za-z]{2,}', text)
        name = ' '.join(alpha[:3])

    # Find serial number
    serial_match = RE_SERIAL.match(text.split()[0]) if text else None
    serial = serial_match.group(1) if serial_match else ""

    # Status
    status = "Unknown"
    if RE_STATUS_P.search(text): status = "Present"
    elif RE_STATUS_A.search(text): status = "Absent"
    elif RE_STATUS_L.search(text): status = "Late"

    if not roll and not name:
        return None

    return {
        "serial":      serial,
        "roll_number": _clean_roll(roll),
        "name":        _clean_name(name),
        "status":      status,
        "raw":         text,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Meta extraction
# ──────────────────────────────────────────────────────────────────────────────

def _extract_meta(rows: List[List[str]]) -> Dict:
    """Try to extract sheet title, date, subject from header rows."""
    meta = {"title": None, "date": None, "subject": None}

    RE_DATE = re.compile(
        r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
    )

    for row in rows[:5]:
        full = ' '.join(row)
        dm = RE_DATE.search(full)
        if dm and not meta['date']:
            meta['date'] = dm.group()

        lower = full.lower()
        if any(w in lower for w in ('attendance', 'register', 'roll call', 'class')):
            if not meta['title']:
                meta['title'] = full.strip()[:80]
        if any(w in lower for w in ('subject', 'course', 'paper', 'dept', 'department')):
            if not meta['subject']:
                meta['subject'] = full.strip()[:80]

    return meta


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def parse_ocr_output(ocr_rows: List[List[str]], mode: str = "grid") -> Dict:
    """
    Takes the raw OCR rows from ocr_engine.ocr_table() and returns:
    {
      'records': [{'serial', 'roll_number', 'name', 'status', 'raw'}, ...],
      'meta':    {'title', 'date', 'subject'},
      'stats':   {'total', 'present', 'absent', 'late', 'unknown'},
      'confidence': 'High' | 'Medium' | 'Low'
    }
    """
    if not ocr_rows:
        return {"records": [], "meta": {}, "stats": {}, "confidence": "Low"}

    records = []
    meta    = _extract_meta(ocr_rows)

    if mode == "grid":
        roles = _detect_column_roles(ocr_rows)
        # Skip header row(s)
        data_rows = [r for r in ocr_rows if not _is_header_row(r)]
        for row in data_rows:
            rec = _parse_row_grid(row, roles)
            if rec:
                records.append(rec)
    else:
        # Band mode: each row is one cell containing the full row text
        for row in ocr_rows:
            text = row[0] if row else ""
            rec  = _parse_row_band(text)
            if rec:
                records.append(rec)

    # Fill in serial numbers if missing
    for i, rec in enumerate(records, start=1):
        if not rec.get('serial'):
            rec['serial'] = str(i)

    # Stats
    statuses = [r['status'] for r in records]
    stats = {
        "total":   len(records),
        "present": statuses.count("Present"),
        "absent":  statuses.count("Absent"),
        "late":    statuses.count("Late"),
        "unknown": statuses.count("Unknown"),
    }

    # Confidence heuristic
    good = sum(1 for r in records if r['roll_number'] or r['name'])
    if len(records) == 0:
        confidence = "Low"
    elif good / len(records) > 0.8:
        confidence = "High"
    elif good / len(records) > 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "records":    records,
        "meta":       meta,
        "stats":      stats,
        "confidence": confidence,
    }
