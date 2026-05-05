# -*- coding: utf-8 -*-

import cv2
import torch
import requests
import time
import sys
import os

# ---------------------------------------------------------------
# CONFIG — a adapter si besoin
# ---------------------------------------------------------------

GATEWAY_URL   = "http://127.0.0.1:8088"   # gateway.py (proxy Flask)
API_ENDPOINT  = "/api/detections"          # route Laravel a creer (voir bas du fichier)

MODEL_PATH    = "/home/ajpi/yolov5/models/best.pt"   # ton modele entraine
CAMERA_INDEX  = 0                          # 0 = premiere camera USB
CONFIDENCE    = 0.5                        # seuil de confiance YOLOv5
SEND_INTERVAL = 0.5                        # secondes entre chaque envoi API

# Noms des classes dans ton dataset (ordre = meme que data.yaml)
# 0 = culture, 1 = adventice (mauvaise herbe)
WEED_CLASS_NAMES = ["carre", "cercle"]
WEED_CLASS_ID    = 1   # index de la classe "mauvaise herbe"

# ---------------------------------------------------------------
# CHARGEMENT DU MODELE
# ---------------------------------------------------------------

print("[AI] Chargement du modele YOLOv5...")
try:
    model = torch.hub.load(
        "/home/ajpi/yolov5",   # dossier yolov5 local (pas internet)
        "custom",
        path=MODEL_PATH,
        source="local",
        force_reload=False,
        verbose=False,
    )
    model.conf = CONFIDENCE
    print(f"[AI] Modele charge : {MODEL_PATH}")
except Exception as e:
    print(f"[AI] ERREUR chargement modele : {e}")
    sys.exit(1)

# ---------------------------------------------------------------
# OUVERTURE CAMERA
# ---------------------------------------------------------------

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"[AI] ERREUR : impossible d'ouvrir la camera (index {CAMERA_INDEX})")
    sys.exit(1)

print(f"[AI] Camera ouverte (index {CAMERA_INDEX})")

# ---------------------------------------------------------------
# FONCTION ENVOI API via GATEWAY
# ---------------------------------------------------------------

def send_detection(weed_detected: int, confidence: float, bbox: dict):
    payload = {
        "weed_detected": weed_detected,
        "confidence":    round(confidence, 4),
        "bbox":          bbox,
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        response = requests.post(
            f"{GATEWAY_URL}{API_ENDPOINT}",
            json=payload,
            timeout=5,
        )
        status = response.status_code
        print(f"[API] Envoi ? weed={weed_detected} conf={confidence:.2f} | HTTP {status}")
        if status not in (200, 201):
            print(f"[API] Reponse : {response.text[:200]}")
    except requests.exceptions.ConnectionError:
        print("[API] ERREUR : gateway inaccessible (gateway.py est-il lance ?)")
    except requests.exceptions.Timeout:
        print("[API] ERREUR : timeout gateway")
    except Exception as e:
        print(f"[API] ERREUR inattendue : {e}")

# ---------------------------------------------------------------
# BOUCLE PRINCIPALE
# ---------------------------------------------------------------

print("[AI] Demarrage de la detection... (Ctrl+C pour arreter)")
last_send = 0.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[AI] ERREUR : impossible de lire le flux camera")
            time.sleep(1)
            continue

        # --- Inférence YOLOv5 ---
        results = model(frame)
        detections = results.xyxy[0]  # tensor [x1, y1, x2, y2, conf, class]

        # --- Analyse des resultats ---
        weed_detected = 0
        best_conf     = 0.0
        best_bbox     = {"x1": 0, "y1": 0, "x2": 0, "y2": 0}

        for *box, conf, cls in detections:
            cls_id = int(cls.item())
            conf_val = float(conf.item())

            if cls_id == WEED_CLASS_ID and conf_val > best_conf:
                weed_detected = 1
                best_conf     = conf_val
                best_bbox     = {
                    "x1": int(box[0].item()),
                    "y1": int(box[1].item()),
                    "x2": int(box[2].item()),
                    "y2": int(box[3].item()),
                }

        # --- Affichage console rapide ---
        label = "ADVENTICE DETECTEE" if weed_detected else "culture OK"
        print(f"[IA] {label} | conf={best_conf:.2f}", end="\r")

        # --- Envoi à l'API (limite dans le temps) ---
        now = time.time()
        if now - last_send >= SEND_INTERVAL:
            send_detection(weed_detected, best_conf, best_bbox)
            last_send = now

        # Optionnel : affichage video avec bounding boxes (desactiver si pas d'ecran)
        # annotated = results.render()[0]
        # cv2.imshow("AgriASpray - Detection", annotated)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

except KeyboardInterrupt:
    print("\n[AI] Arret par l'utilisateur.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("[AI] Camera liberee. Fin du script.")

# ---------------------------------------------------------------
# ROUTE LARAVEL A CREER dans routes/api.php
# ---------------------------------------------------------------
#
# Route::post('/detections', [DetectionController::class, 'store']);
#
# Et dans app/Http/Controllers/DetectionController.php :
#
# public function store(Request $request) {
#     $data = $request->validate([
#         'weed_detected' => 'required|integer|in:0,1',
#         'confidence'    => 'required|numeric',
#         'bbox'          => 'required|array',
#         'timestamp'     => 'required|string',
#     ]);
#     Detection::create($data);   // migration : weed_detected, confidence, bbox (json), timestamp
#     return response()->json(['status' => 'ok'], 201);
# }
#
# ---------------------------------------------------------------