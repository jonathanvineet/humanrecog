#!/usr/bin/env python3
"""
RPi sender script to send frames from local detection to Vercel web app
Improved version with local storage, location data, and faster streaming.
"""

import os
import time
import cv2
import base64
import requests
import json
from datetime import datetime
import random
from threading import Thread
from dotenv import load_dotenv
from inference import get_model

# =============================
# Configuration
# =============================
load_dotenv()
MODEL_ID = os.getenv("MODEL_ID")
API_KEY = os.getenv("ROBOFLOW_API_KEY")
VERCEL_APP_URL = os.getenv("VERCEL_APP_URL", "http://localhost:3000")

# Local Storage Configuration
SAVE_DIRECTORY = "detections"
SAVE_DELAY = 2  # Seconds between local saves to prevent waste of space
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)

# =============================
# State Management
# =============================
last_save_time = 0

# =============================
# Location Helper
# =============================
# Mock variables for coordinate drift
current_lat = 12.971598765432
current_lng = 77.594567890123

def get_location():
    """
    Mock function for high-precision location data with movement.
    """
    global current_lat, current_lng
    # Small random walk to simulate movement (approx 1-2 meters)
    current_lat += random.uniform(-0.00001, 0.00001)
    current_lng += random.uniform(-0.00001, 0.00001)
    return round(current_lat, 10), round(current_lng, 10)

# =============================
# Camera capture thread
# =============================
class CameraStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        # Set lower resolution for faster processing and transmission
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
            # No sleep here for maximum possible throughput

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


def send_frame_to_vercel(frame_base64, detections_count, lat, lng):
    """Send frame to Vercel API and broadcast to clients"""
    try:
        payload = {
            "frameData": frame_base64,
            "detections": detections_count,
            "timestamp": int(time.time() * 1000),
            "location": {
                "lat": lat,
                "lng": lng
            }
        }
        
        response = requests.post(
            f"{VERCEL_APP_URL}/api/upload-frame",
            json=payload,
            timeout=3
        )
        
        # We don't print for every frame to keep log clean at higher speeds
        if response.status_code != 200:
            print(f"✗ Failed to send frame: {response.status_code}")
    except Exception as e:
        # print(f"✗ Error sending frame: {e}")
        pass


def save_detection_locally(frame, detections_count, lat, lng):
    """Save the detection frame locally with metadata"""
    global last_save_time
    current_time = time.time()
    
    if current_time - last_save_time < SAVE_DELAY:
        return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{SAVE_DIRECTORY}/human_{timestamp}_{lat}_{lng}.jpg"
    
    # Save the image
    cv2.imwrite(filename, frame)
    
    # Save metadata as well
    metadata = {
        "timestamp": timestamp,
        "detections": detections_count,
        "location": {"lat": lat, "lng": lng}
    }
    with open(f"{filename}.json", "w") as f:
        json.dump(metadata, f)
        
    print(f"📁 Detection saved locally: {filename}")
    last_save_time = current_time
    return True


def main():
    print("[INFO] Starting Optimized RPi Human Detection Sender...")
    
    # Setup camera
    camera = CameraStream(0).start()
    time.sleep(2)  # Wait for camera to warm up
    
    print("[INFO] Loading model...")
    model = get_model(model_id=MODEL_ID, api_key=API_KEY)
    
    print(f"[INFO] Sending frames to: {VERCEL_APP_URL}")
    print(f"[INFO] Local save delay: {SAVE_DELAY}s")
    print("[INFO] Press Ctrl+C to stop")
    
    frame_count = 0
    
    try:
        while True:
            frame = camera.read()
            if frame is None:
                continue
            
            frame_count += 1
            
            # Run inference every 2 frames for a "smoother" faster-updating feed
            if frame_count % 2 == 0:
                try:
                    display_frame = frame.copy()
                    lat, lng = get_location()
                    
                    results = model.infer(frame)[0]
                    detections_count = 0
                    
                    if hasattr(results, "predictions"):
                        for prediction in results.predictions:
                            if prediction.class_name.lower() in ["person", "human"]:
                                detections_count += 1
                                
                                x, y, w, h = int(prediction.x), int(prediction.y), int(prediction.width), int(prediction.height)
                                x1, y1 = int(x - w/2), int(y - h/2)
                                x2, y2 = int(x + w/2), int(y + h/2)
                                
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(display_frame, f"Human {prediction.confidence:.2f}", 
                                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # If human detected, check local storage
                    if detections_count > 0:
                        save_detection_locally(display_frame, detections_count, lat, lng)
                    
                    # Encode frame to base64 (using lower quality for maximum speed)
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 45]
                    ret, buffer = cv2.imencode('.jpg', display_frame, encode_param)
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send to Vercel in a separate thread
                    Thread(
                        target=send_frame_to_vercel,
                        args=(f"data:image/jpeg;base64,{frame_base64}", detections_count, lat, lng),
                        daemon=True
                    ).start()
                    
                except Exception as e:
                    print(f"[ERROR] Loop error: {e}")
            
            # Tiny sleep to yield without blocking throughput
            time.sleep(0.001)
            
    except KeyboardInterrupt:
        print("\n[INFO] Stopping...")
        camera.stop()
        print("[INFO] Done!")


if __name__ == "__main__":
    main()

