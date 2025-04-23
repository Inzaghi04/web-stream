from ultralytics import YOLO
import cv2
import time
import threading
import torch
from flask import Flask, Response

# Kiểm tra và sử dụng GPU nếu có
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLO model
model = YOLO(r"C:\Users\thinh\Downloads\yolo12n.pt")

# RTSP stream URL
rtsp_url = "rtsp://192.168.0.17:1578/1"
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 20)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

frame = None
lock = threading.Lock()

def read_frames():
    global frame
    while cap.isOpened():
        cap.grab()
        success, new_frame = cap.read()
        if success:
            with lock:
                frame = new_frame

# Start background thread
threading.Thread(target=read_frames, daemon=True).start()

conf_threshold = 0.5
time.sleep(1)

# Flask App
app = Flask(__name__)

def generate_frames():
    while True:
        with lock:
            if frame is None:
                continue
            current_frame = frame.copy()

        start_time = time.time()
        results = model.predict(current_frame, imgsz=320, conf=conf_threshold, classes=[0], device=device)
        annotated_frame = results[0].plot()

        fps = 1 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
        <head><title>YOLOv8 RTSP Stream</title></head>
        <body>
            <h1>RTSP Detection Stream</h1>
            <img src="/video_feed">
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
