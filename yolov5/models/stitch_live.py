from flask import Flask, Response
from reader import CameraReader
import cv2
import numpy as np

app = Flask(__name__)

# Charger la matrice H
H = np.load("homography.npy")
print(">>> Homographie chargée :")
print(H)

# Cams couleur
cam_indexes = [0, 2]
readers = [CameraReader(i) for i in cam_indexes]

def stitch(frames):
    left, right = frames
    h, w = left.shape[:2]

    warped = cv2.warpPerspective(right, H, (w*2, h))
    stitched = warped.copy()
    stitched[0:h, 0:w] = left

    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w2, h2 = cv2.boundingRect(mask)

    return stitched[y:y+h2, x:x+w2]

def generate_frames():
    while True:
        frames = [r.get() for r in readers]
        if any(f is None for f in frames):
            continue

        pano = stitch(frames)
        ret, buffer = cv2.imencode('.jpg', pano)
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )

@app.route('/stitch')
def stitch_route():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print(">>> Stitching en live lancé")
    app.run(host="0.0.0.0", port=5000, debug=False)
