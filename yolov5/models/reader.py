# reader.py
import cv2
import threading
import time

class CameraReader:
    def __init__(self, index):
        self.index = index
        self.cap = cv2.VideoCapture(index)
        self.frame = None
        self.running = True

        # OBSBOT a besoin de temps pour initialiser
        time.sleep(1.0)

        t = threading.Thread(target=self.update, daemon=True)
        t.start()

    def update(self):
        while self.running:
            if not self.cap.isOpened():
                self.cap.open(self.index)
                time.sleep(0.2)
                continue

            ret, frame = self.cap.read()

            if ret and frame is not None:
                self.frame = frame
            else:
                # On attend la prochaine frame
                time.sleep(0.01)

    def get(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()
