# === app.py ===
import io
import os
import uuid
import base64
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import requests
import fitz  # PyMuPDF for reading PDF pages as images
import anthropic

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def call_claude_vision(image_b64, input_prompt):
    api_key = "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"
    if api_key and anthropic:
        client = anthropic.Anthropic(api_key=api_key)

        stream = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=32000,
            temperature=1,
            system="""You are an AI catalog parser. This is a product catalog page showing furniture items, each labeled with a letter (A, B, C...).
            Your Tasks:
            1. Extract structured product information from the image.
            For each product figure, return a JSON object with:
            - "figure": the product label (e.g., "A")
            - "product_name": e.g., "Burl Plank Floor Lamp"
            - "product_id": the SKU (e.g., "138-111562")
            - "price": e.g., 1750.00
            - "bounding_box": object with x, y, width, height
            - "label_type": e.g., "Collection Name", "Price", "Product ID"
            - "collection_name": e.g., "Timeless Elegance"
            2. Ignore irrelevant text like phone numbers, page numbers, slashes (//), and links.
            3. Do not miss the bounding boxes in the JSON response.
            4. Map all labeled figures from a primary index image (A, B, C...) to collection product images across pages.
            5. Always include collection name.
            6. Ignore images or regions without figure labels.
            7. Return a clean JSON list. No markdown, no commentary.""",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_prompt or "Extract structured product data from this catalog image."},
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64
                        }}
                    ]
                }
            ],
            stream=True
        )

        collected = ""
        for chunk in stream:
            if getattr(chunk, "type", "") == "content_block_delta":
                if hasattr(chunk.delta, "text"):
                    collected += chunk.delta.text

        # === DEBUG LOG OUTPUT ===
        print("\n=== Claude Raw Output ===\n", collected, "\n=========================\n")

        # Attempt 1: Look for ```json ... ``` block
        #start = collected.find("```json")
        #end = collected.find("```", start + 7)
        #if start != -1 and end != -1:
        #    json_str = collected[start + 7:end].strip()
        #else:
            # Fallback: try parsing entire output as JSON
        #json_str = collected.strip()

        #try:
        #    return json.loads(json_str)
        #except Exception as e:
        #    print("❌ Claude JSON parse error:", e)
        #    return [{"error": "Failed to parse model output", "raw": collected}]
    return collected
    return ""
    

def process_pdf(file_path, input_prompt):
    doc = fitz.open(file_path)
    all_text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=200)
        image_data = pix.tobytes()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img.thumbnail((2000, 2000))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        compressed_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(compressed_bytes).decode("utf-8")
        result = call_claude_vision(image_b64, input_prompt)
        all_text += f"\n--- Page {page_num + 1} ---\n{result}"
    return all_text
def process_image(image_path, input_prompt):
    with open(image_path, 'rb') as f:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((2000, 2000))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        compressed_bytes = buffer.getvalue()
        image_b64 = base64.b64encode(compressed_bytes).decode("utf-8")
        return call_claude_vision(image_b64, input_prompt)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    input_prompt = request.form.get('prompt', '')

    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    uid = str(uuid.uuid4())
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}.{ext}")
    file.save(save_path)

    if ext == 'pdf':
        ocr_result = process_pdf(save_path, input_prompt)
    else:
        ocr_result = process_image(save_path, input_prompt)

    return jsonify({"results": ocr_result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
