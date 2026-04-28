# autocam.py
import cv2

def detect_cameras():
    cams = []
    for i in [0, 2]:
        cap = cv2.VideoCapture(i)
        ret, frame = cap.read()
        cap.release()
        if ret:
            print("Camera couleur valide :", i)
            cams.append(i)
    return cams
