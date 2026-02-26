import os
import time
import cv2
import subprocess
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
# CAMERA DETECTION via v4l2-ctl
#
# v4l2-ctl --list-devices output looks like:
#
#   bcm2835-codec-decode (platform:bcm2835-codec):   ← ignore (ISP)
#           /dev/video10
#           /dev/video11
#
#   USB Camera (usb-xhci-hcd.0-1):                  ← want this
#           /dev/video0
#           /dev/video1
#
# We grab the first /dev/videoX listed under a "usb" entry.
# Then open it by passing the file descriptor directly — bypassing
# OpenCV's broken-on-Pi integer indexing.
# ═══════════════════════════════════════════════════════════════════
def get_usb_camera_nodes() -> list[int]:
    """
    Runs v4l2-ctl --list-devices and returns the /dev/videoX indices
    that belong to a USB camera device (not ISP/encoder nodes).
    """
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"],
            stderr=subprocess.DEVNULL,
            timeout=5
        ).decode()
    except Exception as e:
        print(f"[CAM] v4l2-ctl failed: {e}")
        return []

    nodes   = []
    in_usb  = False

    for line in out.splitlines():
        stripped = line.strip()
        if not stripped:
            in_usb = False
            continue

        # Device header line — check if it's USB
        if not stripped.startswith("/dev/"):
            in_usb = "usb" in stripped.lower()
            continue

        # Device node line
        if in_usb and stripped.startswith("/dev/video"):
            try:
                idx = int(stripped.replace("/dev/video", ""))
                nodes.append(idx)
            except ValueError:
                pass

    return nodes


def find_camera(retries: int = 30, delay: float = 3.0) -> cv2.VideoCapture:
    """
    Waits for a USB camera to appear via v4l2-ctl, then opens the
    correct /dev/videoX node by passing its file descriptor to OpenCV.
    This bypasses OpenCV's integer re-enumeration which doesn't match
    /dev/videoN on Pi.
    """
    print("[CAM] Waiting for USB camera...")

    for attempt in range(retries):
        nodes = get_usb_camera_nodes()

        if not nodes:
            print(f"[CAM] No USB camera found yet (attempt {attempt+1}/{retries}), retrying in {delay}s...")
            time.sleep(delay)
            continue

        print(f"[CAM] USB camera nodes found: {[f'/dev/video{n}' for n in nodes]}")

        for node_idx in nodes:
            device_path = f"/dev/video{node_idx}"
            print(f"[CAM] Trying {device_path} via file descriptor...")

            try:
                # Open the device file directly and pass the fd to OpenCV.
                # This is the only 100% reliable method on Pi — it skips
                # OpenCV's internal V4L2 enumeration entirely.
                fd  = open(device_path, "rb")
                cap = cv2.VideoCapture(fd.fileno())

                if not cap.isOpened():
                    fd.close()
                    cap.release()
                    print(f"[CAM] {device_path} fd open failed — skipping")
                    continue

                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  480)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                cap.set(cv2.CAP_PROP_FPS,          30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[CAM] ✅ {device_path} is live!")
                    # Keep fd open for the lifetime of cap
                    cap._fd = fd
                    return cap

                fd.close()
                cap.release()
                print(f"[CAM] {device_path} opened but no frame — skipping")

            except Exception as e:
                print(f"[CAM] {device_path} error: {e} — skipping")

        print(f"[CAM] No working node yet. Retrying in {delay}s...")
        time.sleep(delay)

    raise RuntimeError("[CAM] ❌ No working USB camera found after all retries.")


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

    cap     = find_camera(retries=30, delay=3.0)
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