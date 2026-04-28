import cv2
import numpy as np

CHECKERBOARD = (10, 7)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(2)

print(">>> Recherche du checkerboard...")

pts0 = None
pts1 = None

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()
    if not ret0 or not ret1:
        print("Erreur : une caméra ne renvoie rien.")
        break

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    found0, corners0 = cv2.findChessboardCorners(gray0, CHECKERBOARD, None)
    found1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, None)

    if found0:
        pts0 = cv2.cornerSubPix(gray0, corners0, (11,11), (-1,-1), criteria)
        print("Checkerboard trouvé sur cam0")

    if found1:
        pts1 = cv2.cornerSubPix(gray1, corners1, (11,11), (-1,-1), criteria)
        print("Checkerboard trouvé sur cam1")

    if pts0 is not None and pts1 is not None:
        break

# Sélection de 4 points
idxs = [
    0,
    CHECKERBOARD[0] - 1,
    (CHECKERBOARD[1] - 1) * CHECKERBOARD[0],
    CHECKERBOARD[1] * CHECKERBOARD[0] - 1
]

src = np.float32([pts1[i] for i in idxs])
dst = np.float32([pts0[i] for i in idxs])

H, _ = cv2.findHomography(src, dst)
print(">>> Homographie calculée :")
print(H)

np.save("homography.npy", H)
print(">>> Homographie sauvegardée dans homography.npy")

