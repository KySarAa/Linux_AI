# autocam.py
import cv2
import os

def detect_cameras(max_test=10):
    cams = []

    # Lister les devices réellement presents
    video_devices = [f"/dev/video{i}" for i in range(max_test)]
    video_devices = [d for d in video_devices if os.path.exists(d)]

    print(">>> Devices detectes :", video_devices)

    for dev in video_devices:
        idx = int(dev.replace("/dev/video", ""))

        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            continue

        ret, frame = cap.read()
        cap.release()

        if ret:
            print(">>> Camera valide :", idx)
            cams.append(idx)

    return cams
