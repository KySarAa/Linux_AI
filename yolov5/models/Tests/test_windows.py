from flask import Flask, Response
from reader import CameraReader
import cv2

app = Flask(__name__)

# On force les deux cams couleur
cam_indexes = [0, 2]
readers = [CameraReader(i) for i in cam_indexes]

def stream_single(reader):
    while True:
        frame = reader.get()
        if frame is None:
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        )

@app.route('/cam0')
def cam0():
    return Response(stream_single(readers[0]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/cam2')
def cam2():
    return Response(stream_single(readers[1]),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/multi')
def multi():
    def generate():
        while True:
            f0 = readers[0].get()
            f2 = readers[1].get()

            if f0 is None or f2 is None:
                continue

            final = cv2.hconcat([f0, f2])
            ret, buffer = cv2.imencode('.jpg', final)
            if not ret:
                continue

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
            )

    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print(">>> MultiCam Flask lancé")
    app.run(host="0.0.0.0", port=5000, debug=False)
