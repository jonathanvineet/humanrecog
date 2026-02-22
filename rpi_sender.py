#!/usr/bin/env python3
"""
RPi sender script to send frames from local detection to Vercel web app
Place this in your humanrecog folder and run it on the RPi
"""

import os
import time
import cv2
import base64
import requests
from threading import Thread
from dotenv import load_dotenv
from inference import get_model

# =============================
# Configuration
# =============================
load_dotenv()
MODEL_ID = os.getenv("MODEL_ID")
API_KEY = os.getenv("ROBOFLOW_API_KEY")
VERCEL_APP_URL = os.getenv("VERCEL_APP_URL", "http://localhost:3000")  # Update this or add to .env

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


def send_frame_to_vercel(frame_base64, detections_count):
    """Send frame to Vercel API"""
    try:
        payload = {
            "frameData": frame_base64,
            "detections": detections_count,
            "timestamp": int(time.time() * 1000),
        }
        
        response = requests.post(
            f"{VERCEL_APP_URL}/api/upload-frame",
            json=payload,
            timeout=5
        )
        
        if response.status_code == 200:
            print(f"✓ Frame sent successfully ({detections_count} detections)")
        else:
            print(f"✗ Failed to send frame: {response.status_code}")
    except Exception as e:
        print(f"✗ Error sending frame: {e}")


def main():
    print("[INFO] Starting RPi camera stream sender...")
    
    # Setup camera
    camera = CameraStream(0).start()
    time.sleep(2)  # Wait for camera to warm up
    
    print("[INFO] Loading model...")
    model = get_model(model_id=MODEL_ID, api_key=API_KEY)
    
    print(f"[INFO] Sending frames to: {VERCEL_APP_URL}")
    print("[INFO] Press Ctrl+C to stop")
    
    frame_count = 0
    
    try:
        while True:
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame_count += 1
            
            # Run inference every 5 frames
            if frame_count % 5 == 0:
                try:
                    # Draw on frame
                    display_frame = frame.copy()
                    results = model.infer(frame)[0]
                    detections_count = 0
                    
                    if hasattr(results, "predictions"):
                        for prediction in results.predictions:
                            if prediction.class_name.lower() in ["person", "human"]:
                                detections_count += 1
                                
                                x = int(prediction.x)
                                y = int(prediction.y)
                                w = int(prediction.width)
                                h = int(prediction.height)
                                x1 = int(x - w/2)
                                y1 = int(y - h/2)
                                x2 = int(x + w/2)
                                y2 = int(y + h/2)
                                
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(
                                    display_frame,
                                    f"{prediction.class_name} {prediction.confidence:.2f}",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6,
                                    (0, 255, 0),
                                    2,
                                )
                    
                    # Encode frame to base64
                    ret, buffer = cv2.imencode('.jpg', display_frame)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send to Vercel in a separate thread (non-blocking)
                    Thread(
                        target=send_frame_to_vercel,
                        args=(f"data:image/jpeg;base64,{frame_base64}", detections_count),
                        daemon=True
                    ).start()
                    
                except Exception as e:
                    print(f"[ERROR] Inference error: {e}")
            
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
        camera.stop()
        print("[INFO] Done!")


if __name__ == "__main__":
    main()
