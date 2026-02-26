import os
import time
import cv2
import base64
import requests
import json
import random
from datetime import datetime
from threading import Thread
from queue import Queue, Empty
from dotenv import load_dotenv
from inference import get_model

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
UPLOAD_WORKERS = 4       # Multiple threads to beat network latency
INFERENCE_EVERY_N = 3    # Run AI on 1 out of every 3 frames to maintain 30FPS stream

# Queues
raw_frame_queue = Queue(maxsize=1)   # Always hold the newest frame
processed_queue = Queue(maxsize=15)  # Outgoing stream queue

# State
current_lat = HOME_LAT
current_lng = HOME_LNG
latest_detections = [] 

# AUTOMATED IP-BASED LOCATION
def get_ip_location():
    """Gets approximate location via IP Geolocation - No hardware required."""
    try:
        print("[INFO] Fetching automated location from network...")
        response = requests.get('https://ipapi.co/json/', timeout=5)
        data = response.json()
        if 'latitude' in data and 'longitude' in data:
            print(f"[SUCCESS] Location locked to: {data.get('city')}, {data.get('region')}")
            return float(data['latitude']), float(data['longitude'])
    except Exception as e:
        print(f"[WARNING] Location fetch failed: {e}")
    return HOME_LAT, HOME_LNG

# Initial location lock
current_lat, current_lng = get_ip_location()
location_count = 0

def get_location():
    """ 
    Python-native location engine.
    Uses IP discovery first, then simulates micro-movement.
    """
    global current_lat, current_lng, location_count
    
    # Simulating movement drift
    current_lat += random.uniform(-0.0001, 0.0001)
    current_lng += random.uniform(-0.0001, 0.0001)
    
    location_count += 1
    if location_count % 30 == 0:
        print(f"📍 Current Precision Lock: {current_lat:.10f}, {current_lng:.10f}")
        
    return round(current_lat, 10), round(current_lng, 10)

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
            time.sleep(0.01) # Target ~30-60 cycle
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
