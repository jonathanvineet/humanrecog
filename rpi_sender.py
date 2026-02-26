import os
import time
import cv2
import glob
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
INFERENCE_EVERY_N = 12


# ═══════════════════════════════════════════════════════════════════
# CAMERA DETECTION
# Uses Linux sysfs to pre-filter ONLY real capture-capable devices.
# Skips encoder nodes (video10, video11...) without even opening them.
# ═══════════════════════════════════════════════════════════════════
def get_real_capture_devices() -> list[str]:
    """
    Reads /sys/class/video4linux/videoX/device/capabilities (or uevent)
    to find only devices with capture capability (0x00000001 bit set).
    Falls back to all /dev/video* if sysfs isn't available.
    """
    real_devices = []

    for path in sorted(glob.glob("/dev/video*")):
        name = os.path.basename(path)  # e.g. "video0"

        # Check sysfs capabilities — bit 0x00000001 = V4L2_CAP_VIDEO_CAPTURE
        caps_file = f"/sys/class/video4linux/{name}/device/capabilities"
        uevent_file = f"/sys/class/video4linux/{name}/uevent"

        is_capture = False

        if os.path.exists(caps_file):
            try:
                caps = int(open(caps_file).read().strip(), 16)
                if caps & 0x00000001:  # VIDEO_CAPTURE bit
                    is_capture = True
            except:
                pass

        # Fallback: check uevent for "video4linux" type
        if not is_capture and os.path.exists(uevent_file):
            try:
                content = open(uevent_file).read()
                # If no caps file, just include video0-9 (low-index = real cam)
                idx = int(name.replace("video", ""))
                if idx < 10:
                    is_capture = True
            except:
                pass

        if is_capture:
            real_devices.append(path)

    return real_devices


def find_camera(retries: int = 15, delay: float = 3.0) -> cv2.VideoCapture:
    """
    Scans only real capture devices (sysfs-filtered), tests each for a
    live frame. Retries every `delay` seconds to handle slow USB init.
    """
    for attempt in range(retries):
        devices = get_real_capture_devices()
        print(f"[CAM] Attempt {attempt + 1}/{retries} — capture devices: {devices or 'none'}")

        for device_path in devices:
            print(f"[CAM] Trying {device_path}...")

            # Open by full path (more reliable than index on Pi)
            cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)

            if not cap.isOpened():
                print(f"[CAM] {device_path} failed to open — skipping")
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            cap.set(cv2.CAP_PROP_FPS,          30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

            # Read-test: a device can open but still not deliver frames
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[CAM] ✅ {device_path} is live!")
                return cap

            print(f"[CAM] {device_path} opened but gave no frame — skipping")
            cap.release()

        print(f"[CAM] No working camera yet. Retrying in {delay}s...")
        time.sleep(delay)

    raise RuntimeError("[CAM] ❌ No working camera found after all retries.")


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
                        boxes.append((int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)))

            while not result_q.empty():
                try: result_q.get_nowait()
                except: pass

            result_q.put({"boxes": boxes, "detections": detections})

        except Exception as e:
            print(f"[INFERENCE PROC] Error: {e}")


# ═══════════════════════════════════════════════════════════════════
# CAPTURE THREAD
# ═══════════════════════════════════════════════════════════════════
class CaptureThread(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture):
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

    frame_q  = mp.Queue(maxsize=1)
    result_q = mp.Queue(maxsize=1)

    inf_proc = mp.Process(
        target=inference_process_fn,
        args=(frame_q, result_q, MODEL_ID, API_KEY),
        daemon=True
    )
    inf_proc.start()
    print("[SUCCESS] Inference process spawned (GIL-free)")

    cap     = find_camera(retries=15, delay=3.0)
    capture = CaptureThread(cap)
    capture.start()
    print("[SUCCESS] Capture thread running")

    client = mqtt.Client()
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    print(f"[SUCCESS] MQTT tunnel open → {MQTT_BROKER}")

    try:
        base_lat, base_lng = get_current_location()
        print(f"📍 Base coordinate: {base_lat:.10f}, {base_lng:.10f}")
    except:
        base_lat, base_lng = 12.9716, 77.5946
    drift_lat, drift_lng = 0.0, 0.0

    cached_boxes      = []
    cached_detections = 0
    frame_count       = 0
    fps_count         = 0
    fps_timer         = time.time()

    while True:
        frame = capture.read()
        if frame is None:
            time.sleep(0.001)
            continue

        frame_count += 1

        if not result_q.empty():
            try:
                res               = result_q.get_nowait()
                cached_boxes      = res["boxes"]
                cached_detections = res["detections"]
            except:
                pass

        if frame_count % INFERENCE_EVERY_N == 0:
            if not frame_q.full():
                try:
                    frame_q.put_nowait(frame.copy())
                except:
                    pass

        for (x1, y1, x2, y2) in cached_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "HUMAN", (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        drift_lat += random.uniform(-0.00005, 0.00005)
        drift_lng += random.uniform(-0.00005, 0.00005)
        lat = base_lat + drift_lat
        lng = base_lng + drift_lng

        payload = json.dumps({
            "frameData":  f"data:image/jpeg;base64,{frame_b64}",
            "detections": cached_detections,
            "timestamp":  int(time.time() * 1000),
            "location":   {
                "lat": float(f"{lat:.10f}"),
                "lng": float(f"{lng:.10f}"),
            },
        })
        client.publish(MQTT_TOPIC, payload, qos=0)

        fps_count += 1
        elapsed = time.time() - fps_timer
        if elapsed >= 3.0:
            print(f"📡 {fps_count / elapsed:.1f} fps  |  {cached_detections} detections  |  {lat:.6f}, {lng:.6f}")
            fps_count = 0
            fps_timer = time.time()

        time.sleep(0.001)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()