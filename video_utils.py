import cv2
import numpy as np

def analyze_video(video_path, model):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 10 != 0:
            continue

        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)

        pred = model.predict(frame)[0][0]
        predictions.append(pred)

    cap.release()

    avg_pred = sum(predictions) / len(predictions)

    if avg_pred > 0.5:
        return "DEEPFAKE ❌", avg_pred * 100
    else:
        return "REAL ✅", (1 - avg_pred) * 100
