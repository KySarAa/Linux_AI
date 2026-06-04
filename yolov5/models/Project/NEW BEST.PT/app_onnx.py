from ultralytics import YOLO
import cv2
import time
from flask import Flask, Response

# -----------------------------
# CHARGEMENT DU MODELE
# -----------------------------
MODEL_PATH = "best.onnx"
# Ultralytics prend en charge nativement les modeles ONNX avec la meme syntaxe
model = YOLO(MODEL_PATH, task='detect')
print(">>> Modele YOLO (ONNX) charge avec succes :", MODEL_PATH)

# -----------------------------
# FLASK
# -----------------------------
app = Flask(__name__)
cap = cv2.VideoCapture(0)

# -----------------------------
# GENERATEUR DE FRAMES MJPEG
# -----------------------------
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        t0 = time.time()

        # Inference YOLO
        results = model(frame, imgsz=640, conf=0.5, verbose=False)

        # Recuperation des boxes
        annotated = results[0].plot()  # YOLO dessine automatiquement

        # FPS
        fps = 1 / (time.time() - t0)
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Encodage JPEG
        ret2, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret2:
            continue

        frame_bytes = buffer.tobytes()

        # Flux MJPEG
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

# -----------------------------
# ROUTE /video
# -----------------------------
@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------
# MAIN
# -----------------------------
if __name__ == '__main__':
    print(">>> Serveur YOLO en ligne sur /video")
    app.run(host='0.0.0.0', port=5000, debug=False)
