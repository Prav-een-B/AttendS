# AttendS — Attendance Sheet OCR + CNN Digit Recognizer


Extracts roll numbers and student names from attendance sheet photos using:
- **OpenCV** — image preprocessing & table detection
- **CNN** — handwritten digit recognition (~98% MNIST accuracy)
- **Tesseract** — student name OCR
- **openpyxl** — formatted Excel output

---

## Quick Start

### 1. Install Tesseract
```bash
sudo apt-get install tesseract-ocr   # Ubuntu
brew install tesseract               # macOS
```

### 2. Install Python deps
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
python app.py
# Open http://localhost:5000
```

---

## CNN Integration

The model from [Prav-een-B/Recognising-Handwritten-Digits](https://github.com/Prav-een-B/Recognising-Handwritten-Digits) is used as a drop-in digit recognizer.

**Architecture:** `Flatten → Dense(128,ReLU) → Dense(128,ReLU) → Dense(10,Softmax)`

Each digit cell is preprocessed to match MNIST training format:
```
Raw cell → remove grid lines → tight crop → resize 28×28
         → invert (white-on-black) → normalize /255 → predict
```

Falls back to Tesseract automatically if `MYmod.keras` is not present.

---

## Pipeline

```
Image → OpenCV preprocess → Table detection
      → CNN per digit cell → Tesseract for names
      → Parser → Excel
```

## Project Structure

```
attendx/
├── app.py
├── requirements.txt
├── models/
│   └── MYmod.keras            
├── pipeline/
│   ├── preprocess.py
│   ├── table_detector.py
│   ├── cnn_digit_recognizer.py  ← CNN integration
│   ├── ocr_engine.py
│   ├── parser.py
│   └── excel_writer.py
└── templates/
    └── index.html
```

## License
MIT
