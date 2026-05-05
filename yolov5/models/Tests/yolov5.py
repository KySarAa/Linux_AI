from flask import Flask, Response
import cv2
import numpy as np
import tensorflow as tf
import time
import threading
import os

app = Flask(__name__)

# --------- CLASSES ----------
CLASSES = [
    "Trapeze",
    "Triangle",
    "carre",
    "cercle",
    "etoile",
    "losange",
    "rectangle"
]

# --------- MODELE TFLITE ----------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best-int8.tflite")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
print("INPUT SHAPE =", input_details[0]['shape'])
print("INPUT DTYPE =", input_details[0]['dtype'])
print("INPUT QUANT =", input_details[0]['quantization'])

print("OUTPUT SHAPE =", output_details[0]['shape'])
print("OUTPUT DTYPE =", output_details[0]['dtype'])
print("OUTPUT QUANT =", output_details[0]['quantization'])

import sys
sys.exit(0)


input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# INPUT_SIZE = (h, w)
INPUT_SIZE = tuple(input_details[0]['shape'][1:3])

# --------- CAMERA ----------
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

TARGET_FPS = 10
FRAME_INTERVAL = 1 / TARGET_FPS

# --------- PARTAGE ENTRE THREADS ----------
latest_frame = None
latest_annotated = None
lock = threading.Lock()

# --------- UTILS YOLO ----------
def letterbox(img, new_shape, color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_h, new_w = new_shape

    r = min(new_h / h, new_w / w)
    nh, nw = int(h * r), int(w * r)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    top = (new_h - nh) // 2
    left = (new_w - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized

    return canvas, r, (left, top)

def nms(boxes, scores, iou_threshold=0.5):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return keep

def compute_iou(box1, boxes):
    x1, y1, x2, y2 = box1
    xx1 = np.maximum(x1, boxes[:, 0])
    yy1 = np.maximum(y1, boxes[:, 1])
    xx2 = np.minimum(x2, boxes[:, 2])
    yy2 = np.minimum(y2, boxes[:, 3])
    inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area1 + area2 - inter + 1e-6)

def postprocess(output, frame_w, frame_h, r, pad):
    # Tout en float32 pour éviter les erreurs de cast
    preds = np.squeeze(output).astype(np.float32)

    boxes = preds[:4, :].T.astype(np.float32)
    scores = preds[4, :].astype(np.float32)
    class_scores = preds[5:, :].astype(np.float32)

    mask = scores > 0.25
    boxes = boxes[mask]
    scores = scores[mask]
    class_scores = class_scores[:, mask]

    if boxes.shape[0] == 0:
        return [], [], []

    xywh = boxes
    xyxy = np.zeros_like(xywh, dtype=np.float32)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    left, top = pad
    xyxy[:, [0, 2]] -= left
    xyxy[:, [1, 3]] -= top
    xyxy[:, [0, 2]] /= r
    xyxy[:, [1, 3]] /= r

    xyxy[:, 0] = np.clip(xyxy[:, 0], 0, frame_w - 1)
    xyxy[:, 2] = np.clip(xyxy[:, 2], 0, frame_w - 1)
    xyxy[:, 1] = np.clip(xyxy[:, 1], 0, frame_h - 1)
    xyxy[:, 3] = np.clip(xyxy[:, 3], 0, frame_h - 1)

    keep = nms(xyxy, scores)
    xyxy = xyxy[keep]
    scores = scores[keep]
    class_ids = np.argmax(class_scores[:, keep], axis=0)

    return xyxy, scores, class_ids


# --------- THREAD CAPTURE ----------
def capture_loop():
    global latest_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with lock:
            latest_frame = frame.copy()

# --------- THREAD INFERENCE ----------
def inference_loop():
    global latest_frame, latest_annotated
    last_time = time.time()
    while True:
        now = time.time()
        if now - last_time < FRAME_INTERVAL:
            time.sleep(0.001)
            continue
        last_time = now

        with lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        h, w = frame.shape[:2]

        lb_img, r, pad = letterbox(frame, INPUT_SIZE)

        # ---- IMPORTANT : UINT8 ----
        img = lb_img[:, :, ::-1].astype(np.uint8)
        img = np.expand_dims(img, axis=0)

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        print("OUTPUT SHAPE =", output.shape)
        print("INPUT QUANT =", input_details[0]['quantization'])
        print("OUTPUT QUANT =", output_details[0]['quantization'])


        boxes, scores, class_ids = postprocess(output, w, h, r, pad)

        annotated = frame.copy()
        for (x1, y1, x2, y2), score, cls in zip(boxes, scores, class_ids):
            color = (0, 255, 0)
            label = CLASSES[int(cls)] if int(cls) < len(CLASSES) else str(int(cls))
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated, f"{label} {score:.2f}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        with lock:
            latest_annotated = annotated

# --------- FLUX MJPEG ----------
def generate():
    global latest_annotated
    while True:
        with lock:
            frame = None if latest_annotated is None else latest_annotated.copy()
        if frame is None:
            time.sleep(0.01)
            continue

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    t1 = threading.Thread(target=capture_loop, daemon=True)
    t2 = threading.Thread(target=inference_loop, daemon=True)
    t1.start()
    t2.start()
    app.run(host="0.0.0.0", port=8080)

