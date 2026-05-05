from flask import Flask, Response
from reader import CameraReader
import cv2
import numpy as np

app = Flask(__name__)

# Charger la matrice H
H = np.load("homography.npy")
print(">>> Homographie chargee :")
print(H)

# Cams couleur
cam_indexes = [0, 2]
readers = [CameraReader(i) for i in cam_indexes]


# ---------------------------------------------------------
# BLENDING LOCAL (transition douce uniquement dans l'overlap)
# ---------------------------------------------------------
def blend_local(base, warped):
    mask_base = (base.sum(axis=2) > 0).astype(np.uint8)
    mask_warp = (warped.sum(axis=2) > 0).astype(np.uint8)

    overlap = (mask_base & mask_warp).astype(np.uint8)
    only_base = (mask_base & (1 - mask_warp)).astype(np.uint8)
    only_warp = (mask_warp & (1 - mask_base)).astype(np.uint8)

    blended = np.zeros_like(base)

    # Zones exclusives ? copie directe
    blended[only_base == 1] = base[only_base == 1]
    blended[only_warp == 1] = warped[only_warp == 1]

    # Zone de recouvrement ? blending local
    ys, xs = np.where(overlap == 1)
    if len(xs) > 0:
        x_min, x_max = xs.min(), xs.max()
        width = max(1, x_max - x_min)

        # Reduire la zone de blending a 40 px max
        blend_width = min(width, 40)

        for y, x in zip(ys, xs):
            alpha = min(1.0, (x - x_min) / blend_width)
            blended[y, x] = (1 - alpha) * base[y, x] + alpha * warped[y, x]

    return blended.astype(np.uint8)


# ---------------------------------------------------------
# STITCHING PANORAMA LARGE (warp propre + blending local)
# ---------------------------------------------------------
def stitch(frames):
    left, right = frames
    h0, w0 = left.shape[:2]
    h2, w2 = right.shape[:2]

    # Projeter les coins de la cam2 pour dimensionner le panorama
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)

    all_pts = np.concatenate(
        (warped_corners2,
         np.float32([[0, 0], [w0, 0], [w0, h0], [0, h0]]).reshape(-1, 1, 2)),
        axis=0
    )

    [xmin, ymin] = np.int32(all_pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_pts.max(axis=0).ravel() + 0.5)

    tx, ty = -xmin, -ymin
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=np.float32)

    pano_w = xmax - xmin
    pano_h = ymax - ymin

    # Warp de la cam2
    warped = cv2.warpPerspective(right, T @ H, (pano_w, pano_h))

    # Placement de la cam0
    base = warped.copy()
    base[ty:ty + h0, tx:tx + w0] = left

    # Blending local (pas de flou global)
    stitched = blend_local(base, warped)

    # Crop utile
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(mask)

    return stitched[y:y+h, x:x+w]


# ---------------------------------------------------------
# STREAM FLASK
# ---------------------------------------------------------
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
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    print(">>> Stitching en live lance")
    app.run(host="0.0.0.0", port=5000, debug=False)
