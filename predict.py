from ultralytics import YOLO
import cv2
import os

print("Current directory:", os.getcwd())

model = YOLO("model/best.pt")

cap = cv2.VideoCapture("queue.mp4")

if not cap.isOpened():
    print("❌ ERROR: queue.mp4 could not be opened")
    exit()

print("✅ Video opened successfully")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No frame read (end of video or error)")
        break

    results = model(frame)
    annotated = results[0].plot()

    cv2.imshow("Queue Detection", annotated)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
