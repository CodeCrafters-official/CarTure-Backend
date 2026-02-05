from ultralytics import YOLO
import cv2
from flask import Flask, jsonify

app = Flask(__name__)

model = YOLO("model/best.pt")
cap = cv2.VideoCapture("queue.mp4")

QUEUE_LINE_Y = 350
WAIT_PER_PERSON = 30

latest_data = {
    "queue_count": 0,
    "wait_time": 0
}

def process_video():
    global latest_data

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        results = model.track(
            frame,
            persist=True,
            classes=[0],
            tracker="bytetrack.yaml"
        )

        queue_count = 0
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                cy = int((y1 + y2) / 2)
                if cy > QUEUE_LINE_Y:
                    queue_count += 1

        wait_time = queue_count * WAIT_PER_PERSON

        latest_data = {
            "queue_count": queue_count,
            "wait_time": wait_time
        }

@app.route("/queue")
def get_queue():
    return jsonify(latest_data)

if __name__ == "__main__":
    import threading
    t = threading.Thread(target=process_video)
    t.daemon = True
    t.start()

    app.run(host="0.0.0.0", port=5000)
