# -*- coding: utf-8 -*-
print(">>> Script YOLO lance, initialisation...")

from flask import Flask, Response
from ultralytics import YOLO
import cv2
import numpy as np
import time
import serial
import requests

from autocam import detect_cameras
from reader import CameraReader
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


# ---------------------------------------------------------
# ESP32 SERIAL
# ---------------------------------------------------------
try:
    ser = serial.Serial('/dev/ttyACM0', 115200, timeout=0.1)
    print(">>> ESP32 connecte sur /dev/ttyACM0")
except Exception as e:
    print(">>> ERREUR : Impossible d'ouvrir /dev/ttyACM0 :", e)
    ser = None

last_sent = None  # Anti-spam

# ---------------------------------------------------------
# GATEWAY
# ---------------------------------------------------------
GATEWAY_URL = "http://127.0.0.1:8088"
WEED_CLASS  = "carre"  # classe test = mauvaise herbe ? is_weed: 1

def send_to_api(is_weed):
    payload = {"is_weed": is_weed}
    try:
        r = requests.post(f"{GATEWAY_URL}/api/detection", json=payload, timeout=2)
        print(f">>> Gateway envoi : is_weed={is_weed} | status={r.status_code}")
        if r.status_code != 200:
            print(f">>> Gateway reponse : {r.text}")
    except Exception as e:
        print(">>> Gateway ERREUR :", e)

# ---------------------------------------------------------
# FLASK + YOLO
# ---------------------------------------------------------
app = Flask(__name__)
model = YOLO("best.pt")

# ---------------------------------------------------------
# AUTO-DETECTION + LECTURE MULTI-THREAD
# ---------------------------------------------------------
cam_indexes = detect_cameras(max_test=10)

if len(cam_indexes) == 0:
    print(">>> Aucune camera detectee")
    exit()

readers = [CameraReader(i) for i in cam_indexes]
print(f">>> {len(readers)} CameraReader initialises :", cam_indexes)

# ---------------------------------------------------------
# PARAMÈTRES OPTIMISATION FPS
# ---------------------------------------------------------
YOLO_TARGET_SIZE = (640, 360)
YOLO_FPS_LIMIT   = 20
last_yolo_time   = 0
last_results     = None

# ---------------------------------------------------------
# GENERATE FRAMES
# ---------------------------------------------------------
def generate_frames():
    global last_yolo_time, last_results, last_sent

    while True:
        frames = [r.get() for r in readers]

        valid_frames  = []
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
        now      = time.time()
        run_yolo = (now - last_yolo_time) >= (1.0 / YOLO_FPS_LIMIT)

        if run_yolo:
            small_batch  = [cv2.resize(f, YOLO_TARGET_SIZE) for f in valid_frames]
            last_results = model(small_batch, verbose=False)
            last_yolo_time = now

        # -------------------------------------------------
        # Appliquer YOLO + construire les zones
        # -------------------------------------------------
        processed = [None] * len(frames)
        zones     = [0, 0, 0]
        is_weed   = 0  # par défaut : pas de mauvaise herbe

        if last_results is not None:
            for res_idx, cam_idx in enumerate(valid_indices):
                frame = frames[cam_idx].copy()
                r     = last_results[res_idx]

                sx = frames[cam_idx].shape[1] / YOLO_TARGET_SIZE[0]
                sy = frames[cam_idx].shape[0] / YOLO_TARGET_SIZE[1]

                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, x2 = int(x1 * sx), int(x2 * sx)
                    y1, y2 = int(y1 * sy), int(y2 * sy)

                    cls      = int(box.cls[0])
                    conf     = float(box.conf[0])
                    cls_name = model.names[cls].lower()
                    label    = f"{model.names[cls]} {conf:.2f}"

                    # Zones ESP32
                    if cls == 0:
                        zones[0] = 1
                    elif cls == 1:
                        zones[1] = 1
                    elif cls == 2:
                        zones[2] = 1

                    # Carré détecté ? is_weed = 1
                    if cls_name == WEED_CLASS:
                        is_weed = 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                processed[cam_idx] = frame

        # -------------------------------------------------
        # ENVOI ESP32 + GATEWAY (anti-spam)
        # -------------------------------------------------
        if zones != last_sent:
            if ser is not None:
                msg = f"{zones[0]},{zones[1]},{zones[2]}\n"
                ser.write(msg.encode())
                print(">>> Envoi ESP32 :", msg.strip())

            send_to_api(is_weed)
            last_sent = zones

        # Lecture ACK ESP32
        if ser is not None and ser.in_waiting:
            ack = ser.readline().decode().strip()
            if ack:
                print(">>> Reponse ESP32 :", ack)

        # -------------------------------------------------
        # Construction du flux final
        # -------------------------------------------------
        valid_processed = [f for f in processed if f is not None]

        if len(valid_processed) == 1:
            final = valid_processed[0]
        else:
            target_h = 480
            resized  = []
            for f in valid_processed:
                h, w  = f.shape[:2]
                scale = target_h / float(h)
                new_w = int(w * scale)
                resized.append(cv2.resize(f, (new_w, target_h)))
            final = cv2.hconcat(resized)

        ret, buffer = cv2.imencode('.jpg', final, [cv2.IMWRITE_JPEG_QUALITY, 60])
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )


@app.route('/video')
def video():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    print(">>> Serveur YOLO en ligne sur /video")
    app.run(host="0.0.0.0", port=5000, debug=False)