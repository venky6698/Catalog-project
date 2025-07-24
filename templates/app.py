import io
import os
import uuid
import base64
import json
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import requests
from pdf2image import convert_from_path
import anthropic


UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def call_claude_vision(image_path):
    api_key = "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"
    if api_key and anthropic:
        client = anthropic.Anthropic(api_key=api_key)
        with open(image_path, 'rb') as f:
            img = Image.open(image_path).convert("RGB")
            img.thumbnail((2000, 2000))  # Change as needed
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)  # Lower quality = smaller size
            compressed_bytes = buffer.getvalue()

            image_b64 = base64.b64encode(compressed_bytes).decode("utf-8")

            
            message = client.messages.create(
            model="claude-opus-4-20250514",
            max_tokens=32000,
            temperature=1,
            system=""" You are an AI catalog parser. This is a product catalog page showing furniture items, each labeled with a letter (A, B, C...).
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
            2.Ignore irrelevant text like phone numbers, page numbers, slashes (//), and links.
            3.Do not miss the bounding boxes in the json response.
            4.When a image is fed as input at first, try to map the figures A,B.. etc from next stream of collections until you finished mapping all the figures because it is not possible to have 20-30 different products in single image. Also The name of collections would be like "Eg: Timeless Elegance", under that we have all the products from that collection. 
            5.Do not forget to include collection name.
            6. Ignore products in images or whole images which are not having figure numbers like A, B... Z.
            7.Return only a clean JSON list. No markdown, no commentary.""",
            messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": input_prompt
                    },
                    {
                        "type": "image",
                        "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64
                        }
                    }
                ]
            }
        ],
        thinking={
            "type": "enabled",
            "budget_tokens": 31999
        }
    )
    print(message.content)
    return message.content

def process_image(image_path):
    results = call_claude_vision(image_path)
    if isinstance(results, str):
        try:
            results = json.loads(results)
        except:
            results = []
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return 'Invalid file', 400

    ext = file.filename.rsplit('.', 1)[1].lower()
    uid = str(uuid.uuid4())

    if ext == 'pdf':
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], uid + '.pdf')
        file.save(pdf_path)
        pages = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)
        if not pages:
            return 'No pages in PDF', 400
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], uid + '.png')
        pages[0].save(img_path, 'PNG')
    else:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], uid + '.' + ext)
        file.save(img_path)

    ocr_result = process_image(img_path)
    return render_template('results.html', results=ocr_result)

if __name__ == '__main__':
    app.run(debug=True)
