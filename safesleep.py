import cv2
import torch
import asyncio
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

# Load YOLOv8 model
model = YOLO("/Users/brianmason/Desktop/SafeSleep/runs/detect/train3/weights/best.pt")  

# Initialize FastAPI
app = FastAPI()

# Open webcam
cap = cv2.VideoCapture(0)

async def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLOv8 detection
        results = model(frame)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]}: {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               frame_bytes + b"\r\n")

        await asyncio.sleep(0.01)  # Prevent CPU overuse

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")




