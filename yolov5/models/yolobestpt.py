# -*- coding: utf-8 -*-
print(">>> Script YOLO lance, initialisation...")

from flask import Flask, Response
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
import time
import serial
import requests
import threading
import socket

from autocam import detect_cameras
from reader import CameraReader
import os
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


# ---------------------------------------------------------
# ESP32 SERIAL (AUTO-DETECTION)
# ---------------------------------------------------------
import serial
import time

def open_serial():
    ports = ["/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyUSB0"]
    print(">>> Recherche du port serie ESP32...")

    for p in ports:
        for i in range(10):  # 10 tentatives = ~5 secondes
            try:
                ser = serial.Serial(p, 115200, timeout=0.1)
                print(f">>> ESP32 connecte sur {p}")
                return ser
            except:
                time.sleep(0.3)

    print(">>> ERREUR : Aucun port serie ESP32 disponible")
    return None

ser = open_serial()

# ---------------------------------------------------------
# LANCER start.py SUR L'ESP32
# ---------------------------------------------------------
if ser is not None:
    try:
        ser.write(b"import start\n")
        ser.flush()
        print(">>> start.py lance sur l'ESP32")
        time.sleep(0.2)
    except Exception as e:
        print(">>> ERREUR lancement start.py :", e)

last_sent = None

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
CORS(app)
@app.after_request
def add_private_network_header(response):
    response.headers['Access-Control-Allow-Private-Network'] = 'true'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response
    
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
# PARAMETRES OPTIMISATION FPS
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
        is_weed   = 0  # par defaut : pas de mauvaise herbe

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

                    # Carre detecte ? is_weed = 1
                    if cls_name == WEED_CLASS:
                        is_weed = 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                processed[cam_idx] = frame

        # -------------------------------------------------
        # ENVOI ESP32 + GATEWAY (1 seule electrovanne)
        # -------------------------------------------------

        # 3 zones par défaut
        zones = [0, 0, 0]

        # Pour chaque caméra valide
        for res_idx, cam_idx in enumerate(valid_indices):
            r = last_results[res_idx]
            weed_detected = False

            for box in r.boxes:
                cls_name = model.names[int(box.cls[0])].lower()
                if cls_name == WEED_CLASS:
                    weed_detected = True
                    break

            # Si mauvaise herbe détectée ? zone = 1
            if weed_detected:
                zones[cam_idx] = 1

        # Anti-spam : n'envoyer que si changement
        if zones != last_sent:
            if ser is not None:
                msg = '{"cmd":"ai_zones","value":' + str(zones) + '}\n'
                ser.write(msg.encode("utf-8"))
                ser.flush()
                time.sleep(0.01)
                print(">>> Envoi ESP32 :", msg.strip())

            # API : 1 si au moins une zone active
            is_weed = 1 if any(zones) else 0
            send_to_api(is_weed)

            last_sent = zones.copy()

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

# ---------------------------------------------------------
# RAPPORT IP AUTOMATIQUE
# ---------------------------------------------------------
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return '127.0.0.1'

def ip_reporter_thread():
    url = 'http://127.0.0.1:8088/api/robot/ip'
    print('[NETWORK] Demarrage du rapport dIP pour la video...')
    while True:
        try:
            ip = get_local_ip()
            requests.post(url, json={'ip': ip}, timeout=3)
        except:
            pass
        time.sleep(30)

if __name__ == '__main__':
    threading.Thread(target=ip_reporter_thread, daemon=True).start()
    print('>>> Serveur YOLO en ligne sur /video')
    app.run(host='0.0.0.0', port=5000, debug=False)