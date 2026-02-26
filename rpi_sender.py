import os
import time
import cv2
import base64
import requests
import json
import random
from datetime import datetime
from threading import Thread, Lock
from queue import Queue, Empty
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

# LOCATION SETTINGS
# SET YOUR EXACT GPS HOME HERE
HOME_LAT = 12.9716  
HOME_LNG = 77.5946  

# Performance Tuning
JPEG_QUALITY = 35        # Lower = Faster
FRAME_WIDTH = 480        # Reduced for 30fps throughput
FRAME_HEIGHT = 360
UPLOAD_WORKERS = 2       # Reduced to minimize out-of-order frames
INFERENCE_EVERY_N = 2    # Faster inference frequency

# Queues
raw_frame_queue = Queue(maxsize=1)   # Always hold the newest frame
processed_queue = Queue(maxsize=15)  # Outgoing stream queue

# State
current_lat = HOME_LAT
current_lng = HOME_LNG
latest_detections = [] 
detections_lock = Lock()

# Initial location lock
try:
    current_lat, current_lng = get_current_location()
except:
    current_lat, current_lng = HOME_LAT, HOME_LNG

location_count = 0

def get_location():
    global current_lat, current_lng, location_count
    
    # Periodically refresh base location (less frequent to avoid blocking)
    if location_count % 300 == 0:
        try:
            current_lat, current_lng = get_current_location()
        except: pass

    # Simulating movement drift
    current_lat += random.uniform(-0.0001, 0.0001)
    current_lng += random.uniform(-0.0001, 0.0001)
    
    location_count += 1
    if location_count % 30 == 0:
        print(f"📍 Precision Lock [7-dec]: {current_lat:.7f}, {current_lng:.7f}")
        
    return round(current_lat, 7), round(current_lng, 7)

def save_detection_locally(frame, count, lat, lng):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("detections", exist_ok=True)
    filename = f"detections/human_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    print(f"📁 Local Log: {filename}")

# =============================
# Pipeline Components
# =============================

class CameraProducer:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.stopped = False

    def start(self):
        Thread(target=self.run, daemon=True).start()
        return self

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                if raw_frame_queue.full():
                    try: raw_frame_queue.get_nowait()
                    except Empty: pass
                raw_frame_queue.put(frame)
            # High speed capture - no sleep

def inference_engine():
    """Runs AI in its own cycle to not block the 30fps stream"""
    print("[PIPELINE] Inference engine online")
    model = get_model(model_id=MODEL_ID, api_key=API_KEY)
    global latest_detections
    
    while True:
        try:
            frame = raw_frame_queue.get(timeout=1)
            results = model.infer(frame)[0]
            
            temp_preds = []
            if hasattr(results, "predictions"):
                for p in results.predictions:
                    if p.class_name.lower() in ["person", "human"]:
                        temp_preds.append({
                            "box": [int(p.x - p.width/2), int(p.y - p.height/2), int(p.x + p.width/2), int(p.y + p.height/2)],
                            "conf": p.confidence
                        })
            with detections_lock:
                latest_detections = temp_preds
        except Empty: continue
        except Exception: pass

def stream_processor():
    """Assembles frames with latest AI data at 30FPS"""
    print("[PIPELINE] Stream processor active")
    frame_count = 0
    while True:
        try:
            frame = raw_frame_queue.get(timeout=1)
            lat, lng = get_location()
            
            # Draw latest known detections (Zero latency overlay)
            with detections_lock:
                for det in latest_detections:
                    cv2.rectangle(frame, (det["box"][0], det["box"][1]), (det["box"][2], det["box"][3]), (0, 255, 0), 2)
            
            # Local Logging (Optional)
            if latest_detections and frame_count % 30 == 0:
                save_detection_locally(frame, len(latest_detections), lat, lng)

            # Encode
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            b64 = base64.b64encode(buffer).decode('utf-8')
            
            if not processed_queue.full():
                processed_queue.put({
                    "data": f"data:image/jpeg;base64,{b64}",
                    "count": len(latest_detections),
                    "lat": lat, "lng": lng
                })
            frame_count += 1
            # Minimum sleep to allow thread switching but keep 30FPS+
            time.sleep(0.001) 
        except Empty: continue

def upload_worker(id):
    """Pushes to Vercel - Parallel workers for network throughput"""
    while True:
        try:
            p = processed_queue.get(timeout=1)
            payload = {
                "frameData": p["data"], "detections": p["count"],
                "timestamp": int(time.time() * 1000),
                "location": {"lat": p["lat"], "lng": p["lng"]}
            }
            requests.post(f"{VERCEL_APP_URL}/api/upload-frame", json=payload, timeout=2)
        except Empty: continue
        except Exception: pass

if __name__ == "__main__":
    print("🚀 SIGHT OS: EXTREME STREAMING MODE (30FPS TARGET)")
    CameraProducer(0).start()
    
    # Start separate AI loop
    Thread(target=inference_engine, daemon=True).start()
    
    # Start Frame Assembler
    Thread(target=stream_processor, daemon=True).start()
    
    # Start Parallel Uploaders
    for i in range(UPLOAD_WORKERS):
        Thread(target=upload_worker, args=(i,), daemon=True).start()
    
    print(f"[INFO] High-speed pipeline active. Streaming to {VERCEL_APP_URL}")
    while True: time.sleep(1)
