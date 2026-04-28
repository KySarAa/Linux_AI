import cv2
from flask import Flask, Response
from reader import CameraReader

app = Flask(__name__)

# On force les deux cams couleur
cam_indexes = [0, 2]
readers = [CameraReader(i) for i in cam_indexes]

def generate_frames():
    while True:
        frames = [r.get() for r in readers]

        # Si une cam n'a pas encore de frame → on attend
        if any(f is None for f in frames):
            continue

        # Concat horizontale simple
        final = cv2.hconcat(frames)

        ret, buffer = cv2.imencode('.jpg', final)
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print(">>> MultiCam Test")
    app.run(host="0.0.0.0", port=5000, debug=False)
