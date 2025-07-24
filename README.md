# Catalog Spread OCR Extractor - Lite

This project is a lightweight demo web application for extracting text from catalog spread images.

## Features
- Upload single JPEG/PNG or multi-page PDF files.
- Converts PDF pages into images and extracts text using OCR.
- Returns structured JSON with coordinates and labels.
- Displays bounding boxes overlayed on the original image.

## Requirements
- Python 3.8+
- Tesseract OCR installed and available in the system path.

## Setup
```bash
pip install -r requirements.txt
```

## Running
```bash
python app.py
## 1. Navigate to frontend folder
cd react_frontend
npx create-react-app . --template cra-template
npm install
npm start
This should open http://localhost:5173
```

Then open `http://localhost:5000` in your browser.

## Notes
If a Claude API key is available (`CLAUDE_API_KEY` environment variable), the app will attempt to
call the Claude API using the `anthropic` library. Otherwise, it will fall back to local
`tesseract` OCR for text extraction.
