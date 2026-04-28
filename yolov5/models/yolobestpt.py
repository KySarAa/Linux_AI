# -*- coding: utf-8 -*-
print(">>> Script lance, initialisation")

from flask import Flask, Response
from ultralytics import YOLO
import cv2
import numpy as np
import time

from stitcher import stitch_homography
from autocam import detect_cameras
from reader import CameraReader

app = Flask(__name__)
model = YOLO("best.pt")

# ---------------------------------------------------------
# DETECTION DES CAMERAS
# ---------------------------------------------------------
cam_indexes = detect_cameras()
print(">>> Cameras detectees:", cam_indexes)

if len(cam_indexes) == 0:
    print(">>> Aucune camera detectee")
    exit()

readers = [CameraReader(i) for i in cam_indexes]
print(f">>> {len(readers)} CameraReader initialisees")

# ---------------------------------------------------------
# PARAMETRES YOLO
# ---------------------------------------------------------
YOLO_TARGET_SIZE = (640, 360)
YOLO_FPS_LIMIT = 15
last_yolo_time = 0
last_results = None

# ---------------------------------------------------------
# GENERATE FRAMES
# ---------------------------------------------------------
def generate_frames():
    global last_yolo_time, last_results

    while True:
        frames = [r.get() for r in readers]

        # Filtrer les frames valides
        valid_frames = []
        valid_indices = []
        for idx, f in enumerate(frames):
            if f is not None:
                valid_frames.append(f)
                valid_indices.append(idx)

        if len(valid_frames) == 0:
            continue

        # -------------------------------------------------
        # YOLO RATE LIMITER
        # -------------------------------------------------
        now = time.time()
        run_yolo = (now - last_yolo_time) >= (1.0 / YOLO_FPS_LIMIT)

        if run_yolo:
            small_batch = [
                cv2.resize(f, YOLO_TARGET_SIZE)
                for f in valid_frames
            ]
            last_results = model(small_batch, verbose=False)
            last_yolo_time = now

        # -------------------------------------------------
        # Appliquer YOLO
        # -------------------------------------------------
        processed = [None] * len(frames)

        if last_results is not None:
            for res_idx, cam_idx in enumerate(valid_indices):
                frame = frames[cam_idx].copy()
                r = last_results[res_idx]

                sx = frames[cam_idx].shape[1] / YOLO_TARGET_SIZE[0]
                sy = frames[cam_idx].shape[0] / YOLO_TARGET_SIZE[1]

                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, x2 = int(x1 * sx), int(x2 * sx)
                    y1, y2 = int(y1 * sy), int(y2 * sy)

                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"{model.names[cls]} {conf:.2f}"

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.putText(frame, time.strftime("%H:%M:%S"), (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                processed[cam_idx] = frame

        # -------------------------------------------------
        # Construction du flux final
        # -------------------------------------------------

        # Cas 0 : YOLO n'a pas encore tourné ? afficher brut
        if last_results is None:
            if len(valid_frames) == 1:
                final = valid_frames[0]
            else:
                final = cv2.hconcat(valid_frames)

        # Cas 1 : YOLO a tourné mais une seule cam valide
        else:
            valid_processed = [f for f in processed if f is not None]

            if len(valid_processed) == 1:
                final = valid_processed[0]

            elif len(valid_processed) >= 2:
                stitched = stitch_homography(valid_processed[:2])

                if stitched is None or stitched.size == 0:
                    print(">>> Stitching KO ? fallback concat")
                    final = cv2.hconcat(valid_processed[:2])
                else:
                    final = stitched

            else:
                final = cv2.hconcat(valid_frames)

        # Encodage MJPEG
        ret, buffer = cv2.imencode('.jpg', final, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print(">>> Flask demarre")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    finally:
        for r in readers:
            r.stop()
