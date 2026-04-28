import cv2

print("=== TEST DES CAMERAS ===")

for i in [0, 1, 2, 3]:
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()
    if ret and frame is not None:
        print(f"Camera {i} OK -> resolution = {frame.shape}")
    else:
        print(f"Camera {i} KO -> aucune frame")
    cap.release()
