import io
import os
import uuid
import base64
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
from pdf2image import convert_from_path
import anthropic

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def call_claude_vision(image_path, input_prompt):
    api_key = "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"
    if api_key and anthropic:
        client = anthropic.Anthropic(api_key=api_key)
        with open(image_path, 'rb') as f:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((2000, 2000))
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            compressed_bytes = buffer.getvalue()
            image_b64 = base64.b64encode(compressed_bytes).decode("utf-8")

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

        try:
            start = collected.find("```json")
            end = collected.find("```", start + 1)
            if start != -1 and end != -1:
                json_str = collected[start + 7:end].strip()
                return json.loads(json_str)
        except Exception as e:
            print("Claude JSON parse error:", e)

    return []

def process_image(image_path, input_prompt):
    results = call_claude_vision(image_path, input_prompt)
    if isinstance(results, str):
        try:
            results = json.loads(results)
        except:
            results = []
    return results

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    input_prompt = request.form.get('prompt', '')

    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    uid = str(uuid.uuid4())

    if ext == 'pdf':
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], uid + '.pdf')
        file.save(pdf_path)
        pages = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)
        if not pages:
            return jsonify({"error": "PDF has no pages"}), 400
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], uid + '.png')
        pages[0].save(img_path, 'PNG')
    else:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], uid + '.' + ext)
        file.save(img_path)

    ocr_result = process_image(img_path, input_prompt)
    return jsonify({"results": ocr_result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
