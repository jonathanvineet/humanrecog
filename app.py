import os
import time
from threading import Thread
from dotenv import load_dotenv
import cv2

from inference import get_model

# =============================
# Load env and model
# =============================
load_dotenv()
MODEL_ID = os.getenv("MODEL_ID")
API_KEY = os.getenv("ROBOFLOW_API_KEY")

model = get_model(model_id=MODEL_ID, api_key=API_KEY)

# =============================
# Camera capture thread
# =============================
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.frame = None
        self.stopped = False
        self.thread = Thread(target=self.update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

camera = CameraStream(0).start()

# =============================
# Flask streaming
# =============================
from flask import Flask, Response

app = Flask(__name__)

def generate_frames():
    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.01)
            continue

        # run inference on frame
        results = model.infer(frame)[0]

        # draw bounding boxes
        for pred in results.predictions:
            if pred.class_name.lower() in ["person", "human"]:
                x, y = int(pred.x), int(pred.y)
                w, h = int(pred.width), int(pred.height)
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(
                    frame,
                    f"{pred.class_name} {pred.confidence:.2f}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2,
                )

        # encode for browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """<html><body><h1>Human Detection Stream</h1>
              <img src="/video_feed" width="640" height="480"></body></html>"""

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
