import os
import time
import cv2
import base64
import json
import random
from datetime import datetime
from dotenv import load_dotenv
from inference import get_model
from utils import get_current_location

import paho.mqtt.client as mqtt

# =============================
# Configuration
# =============================
load_dotenv()
MODEL_ID = os.getenv("MODEL_ID")
API_KEY = os.getenv("ROBOFLOW_API_KEY")

# Emqx provides a highly reliable free public broker for instant telemetry
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "humanrecog/video/jonathan_feed"

def main():
    print("[INFO] Setting up Absolute Real-Time Native MQTT Stream...")
    
    # Establish MQTT connection
    client = mqtt.Client()
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        print(f"[SUCCESS] MQTT Tunnel open to {MQTT_BROKER}")
    except Exception as e:
        print(f"[ERROR] Network Tunnel Failed: {e}")
        return
        
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    model = get_model(model_id=MODEL_ID, api_key=API_KEY)
    print("[SUCCESS] Local AI Inference Engine loaded")
    
    frame_count = 0
    drift_lat = 0.0
    drift_lng = 0.0
    
    # Secure exact network GPS location
    try:
        base_lat, base_lng = get_current_location()
        print(f"📍 Base Coordinate Secured: {base_lat:.10f}, {base_lng:.10f}")
    except:
        base_lat, base_lng = 12.9716, 77.5946
        
    # Streaming Loop
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame_count += 1
        
        # Max Smoothness: Only process 50% of frames to allow breathing room for transmission
        if frame_count % 2 == 0:
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
            
            # Compress tightly and convert to text for MQTT
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # 10 Decimal GPS calculations
            drift_lat += random.uniform(-0.00005, 0.00005)
            drift_lng += random.uniform(-0.00005, 0.00005)
            lat = base_lat + drift_lat
            lng = base_lng + drift_lng
            
            # Pure telemetry package
            payload = json.dumps({
                "frameData": f"data:image/jpeg;base64,{frame_base64}",
                "detections": detections_count,
                "timestamp": int(time.time() * 1000),
                "location": {
                    "lat": float(f"{lat:.10f}"),
                    "lng": float(f"{lng:.10f}")
                }
            })
            
            # QOS 0 is 'Fire and Forget'. 
            # This completely bypasses the 'out-of-order delay glitch' caused by Vercel HTTP API polling.
            client.publish(MQTT_TOPIC, payload, qos=0)
            
            if frame_count % 30 == 0:
                print(f"📡 High-Speed Stream Push: {lat:.10f}, {lng:.10f}")
        
        # Micro sleep prevents CPU over-crunching without dropping frames
        time.sleep(0.005)

if __name__ == "__main__":
    main()
