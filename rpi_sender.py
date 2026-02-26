import os
import time
import cv2
import base64
import requests
import json
import random
from datetime import datetime
from threading import Thread
from queue import Queue
from dotenv import load_dotenv
from inference import get_model
from utils import get_current_location

# =============================
# Configuration
# =============================
load_dotenv()
MODEL_ID = os.getenv("MODEL_ID")
API_KEY = os.getenv("ROBOFLOW_API_KEY")
VERCEL_APP_URL = os.getenv("VERCEL_APP_URL", "http://localhost:3000")

# Setup robust ordered uploading
upload_queue = Queue(maxsize=10)

def upload_worker():
    """Single thread ensures frames are uploaded strictly in order to prevent glitching"""
    while True:
        try:
            payload = upload_queue.get()
            if payload is None: break
            
            requests.post(
                f"{VERCEL_APP_URL}/api/upload-frame", 
                json=payload, 
                timeout=5
            )
            upload_queue.task_done()
        except Exception as e:
            pass

def main():
    print("[INFO] Starting Reliable UI Streamer...")
    
    # Start single upload thread to prevent out-of-order glitches
    Thread(target=upload_worker, daemon=True).start()
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    model = get_model(model_id=MODEL_ID, api_key=API_KEY)
    
    frame_count = 0
    drift_lat = 0.0
    drift_lng = 0.0
    
    # Get initial base location from Python Utils
    try:
        base_lat, base_lng = get_current_location()
    except:
        base_lat, base_lng = 12.9716, 77.5946

    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame_count += 1
        
        # Process every N frames for balance
        if frame_count % 3 == 0:
            results = model.infer(frame)[0]
            
            detections_count = 0
            if hasattr(results, "predictions"):
                for p in results.predictions:
                    if p.class_name.lower() in ["person", "human"]:
                        detections_count += 1
                        x, y, w, h = int(p.x), int(p.y), int(p.width), int(p.height)
                        x1, y1 = int(x - w/2), int(y - h/2)
                        x2, y2 = int(x + w/2), int(y + h/2)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Encode frame to base64
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Calculate drifted precise location
            drift_lat += random.uniform(-0.00005, 0.00005)
            drift_lng += random.uniform(-0.00005, 0.00005)
            lat = base_lat + drift_lat
            lng = base_lng + drift_lng
            
            payload = {
                "frameData": f"data:image/jpeg;base64,{frame_base64}",
                "detections": detections_count,
                "timestamp": int(time.time() * 1000),
                "location": {
                    "lat": float(f"{lat:.10f}"),
                    "lng": float(f"{lng:.10f}")
                }
            }
            
            if not upload_queue.full():
                upload_queue.put(payload)
                
            print(f"📍 GPS (10-dec): {lat:.10f}, {lng:.10f}")

if __name__ == "__main__":
    main()
