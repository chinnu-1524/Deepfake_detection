from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from utils.preprocess import preprocess_image
from utils.video_utils import analyze_video

# ---------------- APP CONFIG ----------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- LOAD MODELS ----------------
image_model = load_model("model/image_model.h5")
video_model = load_model("model/video_model.h5")

# ---------------- HOME (FRONTEND) ----------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------- IMAGE DETECTION ----------------
@app.route("/detect-image", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty file name"}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = cv2.imread(path)
    if img is None:
        return jsonify({"error": "Invalid image file"}), 400

    processed_img = preprocess_image(img)
    prediction = image_model.predict(processed_img)[0][0]

    if prediction > 0.5:
        result = "DEEPFAKE ❌"
        confidence = prediction * 100
    else:
        result = "REAL ✅"
        confidence = (1 - prediction) * 100

    return jsonify({
        "result": result,
        "confidence": f"{confidence:.2f}%"
    })

# ---------------- VIDEO DETECTION ----------------
@app.route("/detect-video", methods=["POST"])
def detect_video():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty file name"}), 400

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    result, confidence = analyze_video(path, video_model)

    return jsonify({
        "result": result,
        "confidence": f"{confidence:.2f}%"
    })

# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True)
