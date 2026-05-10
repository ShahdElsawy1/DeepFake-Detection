import os, json, base64, io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
CORS(app)

print("Loading model...")
with open('C:/Users/hp/OneDrive/Desktop/python/deepfake_app/model_files/config.json') as f:
    config = json.load(f)
model = tf.keras.models.model_from_json(json.dumps(config))
model.load_weights('C:/Users/hp/OneDrive/Desktop/python/deepfake_app/model_files/model.weights.h5')
print("Model ready.")

def preprocess_image(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        img_bytes = base64.b64decode(data['image'])
        arr = preprocess_image(img_bytes)
        prob = float(model.predict(arr, verbose=0)[0][0])

        if prob > 0.5:
            verdict = 'REAL'
            confidence = round(prob * 100, 1)
        else:
            verdict = 'FAKE'
            confidence = round((1 - prob) * 100, 1)

        return jsonify({
            'verdict': verdict,
            'confidence': confidence,
            'raw_probability': round(prob, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
