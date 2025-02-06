from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # CORS qo'shish
import cv2
import numpy as np
from deepface import DeepFace

app = Flask(__name__)
CORS(app)  # CORS ni yoqish

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/detect_emotion', methods=['POST'])
def detect_emotion():
    if 'image' not in request.files:
        return jsonify({"error": "Rasm yuklanmadi"}), 400

    # Yuklangan rasmni olish
    file = request.files['image']
    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    try:
        # DeepFace yordamida yuz ifodasini aniqlash
        analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        # Natijani JSON formatda qaytarish
        emotions = {key: float(value) for key, value in analysis[0]["emotion"].items()}
        return jsonify({"emotions": emotions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)