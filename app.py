"""
app.py  —  AttendX Flask Application
--------------------------------------
Supported input formats: JPG, PNG, BMP, TIFF, WEBP, PDF

Pipeline per image/page:
  PDF → pdf_converter (poppler) → PNG pages
  PNG/JPG → OpenCV preprocess → Table detection
           → CNN digit recognizer (MYmod.keras)  [roll numbers]
           → Tesseract OCR                        [names]
           → Parser → openpyxl Excel
"""

import os, re, uuid, cv2, numpy as np, pytesseract
from PIL      import Image
from datetime import datetime
from flask    import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename

from pipeline.preprocess      import preprocess
from pipeline.table_detector  import detect_table
from pipeline.excel_writer    import write_excel
from pipeline.pdf_converter   import pdf_to_images, cleanup_images, is_pdf
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024   # 64 MB (PDFs can be large)
app.config["UPLOAD_FOLDER"]  = "uploads"
app.config["OUTPUT_FOLDER"]  = "outputs"
MODEL_PATH = os.path.join("models", "MYmod.keras")

ALLOWED_EXT = {"jpg","jpeg","png","bmp","tiff","tif","webp","pdf"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── Load CNN once at startup ───────────────────────────────────────────────────
def _init_cnn():
    if not os.path.exists(MODEL_PATH):
        print(f"[app] No model at {MODEL_PATH} — Tesseract fallback active")
        return False
    try:
        from pipeline.cnn_digit_recognizer import load_model
        load_model(MODEL_PATH)
        return True
    except Exception as e:
        print(f"[app] CNN load failed: {e}")
        return False

CNN_READY = _init_cnn()

# ══════════════════════════════════════════════════════════════════════════════
# OCR helpers
# ══════════════════════════════════════════════════════════════════════════════

def _clean_lines(sg):
    h, w = sg.shape
    _, bw = cv2.threshold(sg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(bw)
    hk  = cv2.getStructuringElement(cv2.MORPH_RECT, (max(8,w//5),1))
    inv = cv2.subtract(inv, cv2.morphologyEx(inv,cv2.MORPH_OPEN,hk))
    vk  = cv2.getStructuringElement(cv2.MORPH_RECT, (1,max(5,h//4)))
    inv = cv2.subtract(inv, cv2.morphologyEx(inv,cv2.MORPH_OPEN,vk))
    return cv2.bitwise_not(inv)

def _tess_digit(cell_gray):
    clean = _clean_lines(cell_gray)
    big   = cv2.resize(clean,(120,120),interpolation=cv2.INTER_CUBIC)
    _,big = cv2.threshold(big,127,255,cv2.THRESH_BINARY)
    pad   = cv2.copyMakeBorder(big,25,25,25,25,cv2.BORDER_CONSTANT,value=255)
    pil   = Image.fromarray(pad)
    for cfg in ['--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789',
                '--psm 8  --oem 3 -c tessedit_char_whitelist=0123456789']:
        try:
            d = re.findall(r'\d', pytesseract.image_to_string(pil,config=cfg))
            if d: return d[0]
        except: pass
    return '?'

def _ocr_name(cell_gray):
    h, w  = cell_gray.shape
    scale = max(56/h, 1.0)
    big   = cv2.resize(cell_gray,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_CUBIC)
    _,bw  = cv2.threshold(big,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    pad   = cv2.copyMakeBorder(bw,14,14,14,14,cv2.BORDER_CONSTANT,value=255)
    text  = pytesseract.image_to_string(Image.fromarray(pad),config='--psm 7 --oem 3').strip()
    text  = re.sub(r'^[|/\\\[{<\s]+','',text)
    text  = re.sub(r'[|_\[\]{}<>]+',' ',text)
    text  = re.sub(r'[^A-Za-z .\'-]',' ',text)
    text  = re.sub(r'\s+',' ',text).strip()
    if len(text)>2 and text.upper()==text: text=text.title()
    return text

# ══════════════════════════════════════════════════════════════════════════════
# Core extraction — one image/page at a time
# ══════════════════════════════════════════════════════════════════════════════

def extract_from_image(image_path: str, source_label: str = "") -> dict:
    """Run full pipeline on a single image file. Returns parsed dict."""

    pre    = preprocess(image_path)
    gray   = pre["gray"]
    h_img, w_img = gray.shape

    table  = detect_table(pre)
    row_ys = table["row_ys"]
    col_xs = table["col_xs"]
    mode   = table["mode"]

    has_grid = (mode == "grid" and col_xs and len(col_xs) >= 10)

    if has_grid:
        digit_col_bounds = [(col_xs[i], col_xs[i+1]) for i in range(1, 6)]
        name_x1, name_x2 = col_xs[8], col_xs[9]
    else:
        digit_col_bounds = []
        name_x1 = name_x2 = 0

    row_bounds = list(zip(row_ys[:-1], row_ys[1:])) if row_ys else []
    data_rows  = row_bounds[2:-1] if len(row_bounds) > 3 else row_bounds

    engine      = "Tesseract"
    cnn_results = None

    if CNN_READY and has_grid:
        try:
            from pipeline.cnn_digit_recognizer import predict_all_rolls
            cnn_results = predict_all_rolls(
                gray_image           = gray,
                row_ys               = row_ys,
                digit_col_bounds     = digit_col_bounds,
                skip_header_rows     = 2,
                skip_footer_rows     = 1,
                confidence_threshold = 0.50,
            )
            engine = "CNN (MYmod.keras)"
        except Exception as e:
            print(f"[CNN] Prediction failed: {e} — falling back to Tesseract")

    records = []
    for i, (y1, y2) in enumerate(data_rows):
        pad = 3
        ry1 = max(0, y1+pad); ry2 = min(h_img, y2-pad)
        if ry2-ry1 < 8: continue

        # Roll number
        if cnn_results and i < len(cnn_results):
            r        = cnn_results[i]
            roll     = r['roll']
            raw_note = f"CNN conf={r['avg_conf']:.2f}"
        elif has_grid and digit_col_bounds:
            digs     = [_tess_digit(gray[ry1:ry2, max(0,dx1+2):min(w_img,dx2-2)])
                        for dx1,dx2 in digit_col_bounds]
            roll     = ''.join(d for d in digs if d!='?')
            raw_note = f"Tess digits={digs}"
        else:
            roll     = ''
            raw_note = ''

        # Name
        if has_grid and name_x2 > name_x1:
            name = _ocr_name(gray[ry1:ry2, max(0,name_x1):min(w_img,name_x2)])
        else:
            name = _ocr_name(gray[ry1:ry2, :])

        if not roll and not name:
            continue

        records.append({
            "serial":      str(i+1),
            "roll_number": roll,
            "name":        name,
            "status":      "Present",
            "raw":         raw_note,
        })

    good = sum(1 for r in records if len(r["roll_number"])==5 or r["name"])
    conf = ("High"   if records and good/len(records)>0.8 else
            "Medium" if records and good/len(records)>0.4 else "Low")

    return {
        "_source_file": source_label or os.path.basename(image_path),
        "meta": {"title": "Attendance Sheet", "date": None, "subject": None},
        "records":     records,
        "stats":       {"total":len(records),"present":len(records),
                        "absent":0,"late":0,"unknown":0},
        "confidence":  conf,
        "engine_used": engine,
    }


def process_file(file_path: str, original_name: str) -> list:
    """
    Process one uploaded file (image or PDF).
    Returns a list of parsed dicts — one per page for PDFs, one for images.
    """
    if is_pdf(original_name):
        # ── PDF: convert each page to PNG, then process ─────────────────────
        page_images = pdf_to_images(file_path, dpi=300)
        results = []
        try:
            for page_num, img_path in enumerate(page_images, start=1):
                label  = (f"{original_name} — page {page_num}"
                          if len(page_images) > 1 else original_name)
                parsed = extract_from_image(img_path, source_label=label)
                results.append(parsed)
        finally:
            cleanup_images(page_images)
        return results
    else:
        # ── Image: process directly ──────────────────────────────────────────
        parsed = extract_from_image(file_path, source_label=original_name)
        return [parsed]

# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html", cnn_ready=CNN_READY, model_path=MODEL_PATH)

@app.route("/model-status")
def model_status():
    return jsonify({"cnn_ready": CNN_READY, "model_path": MODEL_PATH,
                    "model_exists": os.path.exists(MODEL_PATH)})

@app.route("/upload-model", methods=["POST"])
def upload_model():
    global CNN_READY
    if "model" not in request.files:
        return jsonify({"error": "No model file"}), 400
    f = request.files["model"]
    if not f.filename.endswith((".keras", ".h5")):
        return jsonify({"error": "Must be .keras or .h5"}), 400
    f.save(MODEL_PATH)
    CNN_READY = _init_cnn()
    return jsonify({"success": True, "cnn_ready": CNN_READY})

@app.route("/process", methods=["POST"])
def process():
    if "files" not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    files = request.files.getlist("files")
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files selected"}), 400

    all_parsed, errors = [], []

    for f in files:
        if not f or f.filename == "": continue
        ext = f.filename.rsplit(".", 1)[-1].lower()
        if ext not in ALLOWED_EXT:
            errors.append(f"{f.filename}: unsupported format"); continue

        tmp_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            f"{uuid.uuid4().hex[:8]}_{secure_filename(f.filename)}"
        )
        f.save(tmp_path)

        try:
            results = process_file(tmp_path, f.filename)
            all_parsed.extend(results)
        except Exception as e:
            errors.append(f"{f.filename}: {str(e)}")
        finally:
            try: os.remove(tmp_path)
            except: pass

    if not all_parsed:
        return jsonify({"error": "No sheets processed", "details": errors}), 400

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    outname = f"attendance_{ts}.xlsx"
    outpath = os.path.join(app.config["OUTPUT_FOLDER"], outname)
    write_excel(all_parsed, outpath)

    preview = [{
        "source":     p.get("_source_file", ""),
        "title":      p.get("meta", {}).get("title"),
        "date":       p.get("meta", {}).get("date"),
        "subject":    p.get("meta", {}).get("subject"),
        "confidence": p.get("confidence"),
        "engine":     p.get("engine_used", ""),
        "records":    p.get("records", []),
        "stats":      p.get("stats", {}),
    } for p in all_parsed]

    return jsonify({"success": True, "filename": outname,
                    "sheets": preview, "errors": errors})

@app.route("/download/<filename>")
def download(filename):
    path = os.path.join(app.config["OUTPUT_FOLDER"], secure_filename(filename))
    if not os.path.exists(path):
        return jsonify({"error": "File not found"}), 404
    return send_file(path, as_attachment=True, download_name=filename,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
