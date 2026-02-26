import os
import time
import cv2
import base64
import json
import random
import threading
import multiprocessing as mp
from dotenv import load_dotenv
from utils import get_current_location
import paho.mqtt.client as mqtt

load_dotenv()
MODEL_ID = os.getenv("MODEL_ID")
API_KEY  = os.getenv("ROBOFLOW_API_KEY")

MQTT_BROKER       = "broker.emqx.io"
MQTT_PORT         = 1883
MQTT_TOPIC        = "humanrecog/video/jonathan_feed"
INFERENCE_EVERY_N = 12   # submit frame for inference every N frames

# Ensure detection storage exists
os.makedirs("detections/metadata", exist_ok=True)
os.makedirs("detections/frames", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
# INFERENCE PROCESS  (separate OS process — fully GIL-free)
# ═══════════════════════════════════════════════════════════════════
def inference_process_fn(frame_q: mp.Queue, result_q: mp.Queue, model_id: str, api_key: str):
    from inference import get_model
    model = get_model(model_id=model_id, api_key=api_key)
    print("[INFERENCE PROC] Ready")

    while True:
        try:
            frame = frame_q.get(timeout=2)
        except Exception:
            continue

        try:
            results    = model.infer(frame)[0]
            boxes      = []
            detections = 0

            if hasattr(results, "predictions"):
                for p in results.predictions:
                    if p.class_name.lower() in ["person", "human"]:
                        detections += 1
                        x, y, w, h = int(p.x), int(p.y), int(p.width), int(p.height)
                        # Box format: (x1, y1, x2, y2)
                        boxes.append((int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)))

            # Drain stale results before writing new
            while not result_q.empty():
                try: result_q.get_nowait()
                except: pass

            result_q.put({"boxes": boxes, "detections": detections})

        except Exception as e:
            print(f"[INFERENCE PROC] Error: {e}")


# ═══════════════════════════════════════════════════════════════════
# CAPTURE THREAD  (continuously drains camera buffer)
# ═══════════════════════════════════════════════════════════════════
class CaptureThread(threading.Thread):
    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap     = cap
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    print("[INFO] Starting GIL-free high-speed stream...")

    # Queues: maxsize=1 means we only ever hold the freshest frame/result
    frame_q  = mp.Queue(maxsize=1)
    result_q = mp.Queue(maxsize=1)

    inf_proc = mp.Process(
        target=inference_process_fn,
        args=(frame_q, result_q, MODEL_ID, API_KEY),
        daemon=True
    )
    inf_proc.start()
    print("[SUCCESS] Inference process spawned (separate OS process, GIL-free)")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS,          30)
    # Buffersize 1 ensures we are always reading the latest "live" frame from the sensor
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    capture = CaptureThread(cap)
    capture.start()
    print("[SUCCESS] Capture thread running")

    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    print(f"[SUCCESS] MQTT tunnel open → {MQTT_BROKER}")

    try:
        base_lat, base_lng = get_current_location()
    except:
        base_lat, base_lng = 12.9716, 77.5946
    drift_lat, drift_lng = 0.0, 0.0

    cached_boxes      = []
    cached_detections = 0
    frame_count       = 0
    fps_count         = 0
    fps_timer         = time.time()
    last_save_ts      = 0

    while True:
        frame = capture.read()
        if frame is None:
            time.sleep(0.001)
            continue

        frame_count += 1

        # Non-blocking poll for new inference results
        if not result_q.empty():
            try:
                res               = result_q.get_nowait()
                cached_boxes      = res["boxes"]
                cached_detections = res["detections"]
            except:
                pass

        # Submit frame to inference (drops if process is still busy)
        if frame_count % INFERENCE_EVERY_N == 0:
            if not frame_q.full():
                try:
                    frame_q.put_nowait(frame.copy())
                except:
                    pass

        # Draw cached boxes — zero inference cost on main thread
        for (x1, y1, x2, y2) in cached_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "HUMAN", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # High-speed JPEG encoding
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # Telemetry drift simulation
        drift_lat += random.uniform(-0.00005, 0.00005)
        drift_lng += random.uniform(-0.00005, 0.00005)
        lat = base_lat + drift_lat
        lng = base_lng + drift_lng

        timestamp = int(time.time() * 1000)
        data_payload = {
            "frameData":  f"data:image/jpeg;base64,{frame_b64}",
            "detections": cached_detections,
            "timestamp":  timestamp,
            "location":   {
                "lat": float(f"{lat:.10f}"),
                "lng": float(f"{lng:.10f}"),
            },
        }

        # Save detection locally if humans found (with 2s cooldown)
        if cached_detections > 0 and (timestamp - last_save_ts > 2000):
            last_save_ts = timestamp
            # Save Frame
            cv2.imwrite(f"detections/frames/{timestamp}.jpg", frame)
            # Save Metadata
            with open(f"detections/metadata/{timestamp}.json", "w") as f:
                json.dump(data_payload, f)
            
            # Cleanup: keep only last 10 entries
            try:
                all_meta = sorted(os.listdir("detections/metadata"))
                if len(all_meta) > 10:
                    for old_file in all_meta[:-10]:
                        try:
                            os.remove(os.path.join("detections/metadata", old_file))
                            os.remove(os.path.join("detections/frames", old_file.replace('.json', '.jpg')))
                        except: pass
            except: pass

        payload = json.dumps(data_payload)
        client.publish(MQTT_TOPIC, payload, qos=0)

        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 3.0:
            print(f"📡 {fps_count / elapsed:.1f} fps  |  {cached_detections} detections  |  {lat:.6f}, {lng:.6f}")
            fps_count = 0
            fps_timer = time.time()

        time.sleep(0.001)


if __name__ == "__main__":
    # Ensure spawn method for macOS/Linux safety
    mp.set_start_method("spawn", force=True)
    main()
