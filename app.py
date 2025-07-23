import base64
import os
import uuid
from flask import Flask, render_template, request 
from werkzeug.utils import secure_filename
from PIL import Image
from pdf2image import convert_from_path
import pytesseract

try:
    import anthropic
except ImportError:
    anthropic = None

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def call_claude_vision(image_path):
    api_key = os.getenv('CLAUDE_API_KEY')
    if api_key and anthropic:
        client = anthropic.Anthropic(api_key=api_key)
        with open(image_path, 'rb') as f:
            img_data = f.read()
        message = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an AI model analyzing a catalog spread image. Extract the key pieces of text (e.g., product names, prices, short descriptions) and return them in JSON format including approximate coordinates."
                        },
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(img_data).decode('ascii')
                            }
                        }
                    ]
                }
            ]
        )
        # For simplicity assume Claude returns the JSON directly in the content
        return message.content[0].text
    else:
        # fallback to pytesseract
        image = Image.open(image_path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        results = []
        n_boxes = len(data['text'])
        for i in range(n_boxes):
            text = data['text'][i].strip()
            if text:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                results.append({
                    'text': text,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'label': ''
                })
        return results


def process_image(image_path):
    results = call_claude_vision(image_path)
    if isinstance(results, str):
        # try to parse JSON
        import json
        try:
            results = json.loads(results)
        except Exception:
            results = []
    return results


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file or not allowed_file(file.filename):
        return 'Invalid file', 400
    filename = secure_filename(file.filename)
    ext = filename.rsplit('.', 1)[1].lower()
    uid = str(uuid.uuid4())
    if ext == 'pdf':
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], uid + '.pdf')
        file.save(pdf_path)
        pages = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=2)
        if not pages:
            return 'No pages found', 400
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], uid + '.png')
        pages[0].save(img_path, 'PNG')
    else:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], uid + '.' + ext)
        file.save(img_path)

    boxes = process_image(img_path)
    image_url = '/' + img_path
    return render_template('index.html', image_url=image_url, boxes=boxes)


if __name__ == '__main__':
    app.run(debug=True)
