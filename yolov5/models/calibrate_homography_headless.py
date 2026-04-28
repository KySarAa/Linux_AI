from flask import Flask, Response
import cv2
import numpy as np
import time

app = Flask(__name__)

# Parametres du damier
CHECKERBOARD = (8, 11)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Cams couleur
cams = [0, 2]
caps = [cv2.VideoCapture(i) for i in cams]

print(">>> Calibration web : ouvre /calib pour voir les cams")
print(">>> Place un damier visible par les deux cams")
print(">>> Le script calcule H automatiquement")

pts_cam0 = None
pts_cam2 = None
H = None

def detect_and_draw(frame, checkerboard):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

    if ret:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        cv2.drawChessboardCorners(frame, checkerboard, corners2, ret)
        return frame, corners2.reshape(-1, 2), True

    return frame, None, False

def generate_frames():
    global pts_cam0, pts_cam2, H

    while True:
        frames = []
        for cap in caps:
            ret, frame = cap.read()
            frames.append(frame if ret else None)

        if any(f is None for f in frames):
            continue

        # Detection damier + overlay
        f0, pts0, ok0 = detect_and_draw(frames[0].copy(), CHECKERBOARD)
        f2, pts2, ok2 = detect_and_draw(frames[1].copy(), CHECKERBOARD)

        # Si damier detecte sur les deux cams ? calcul H
        if ok0 and ok2 and H is None:
            print(">>> Damier detecte sur les deux cams, calcul de H...")
            H, _ = cv2.findHomography(pts2, pts0, cv2.RANSAC)
            np.save("homography.npy", H)
            print(">>> Homographie sauvegardee dans homography.npy")

        # Fusion affichage (concat)
        final = cv2.hconcat([f0, f2])

        # Si H calculee ? afficher un message
        if H is not None:
            cv2.putText(final, "HOMOGRAPHIE OK", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

        ret, buffer = cv2.imencode('.jpg', final)
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )

@app.route('/calib')
def calib():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print(">>> Serveur de calibration lance")
    app.run(host="0.0.0.0", port=5000, debug=False)
