# -*- coding: utf-8 -*-
import cv2
import numpy as np

PTS1 = np.float32([
    [226, 166],
    [377, 131],
    [237, 333],
    [397, 398]
])

PTS2 = np.float32([
    [872, 166],
    [1037, 131],
    [884, 333],
    [1063, 397]
])

def stitch_homography(frames):
    left, right = frames[0], frames[1]

    h, w = left.shape[:2]

    # Homographie
    H, _ = cv2.findHomography(PTS2, PTS1, cv2.RANSAC)
    if H is None:
        print(">>> Homographie impossible ? fallback concat")
        return cv2.hconcat(frames)

    warped = cv2.warpPerspective(right, H, (w*2, h))
    if warped is None or warped.size == 0:
        print(">>> warpPerspective vide ? fallback concat")
        return cv2.hconcat(frames)

    stitched = warped.copy()
    stitched[0:h, 0:w] = left

    # Recadrage
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w2, h2 = cv2.boundingRect(mask)

    if w2 == 0 or h2 == 0:
        print(">>> boundingRect vide ? fallback concat")
        return cv2.hconcat(frames)

    return stitched[y:y+h2, x:x+w2]
