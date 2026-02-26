import os
import time
import cv2
import subprocess
import numpy as np
import base64
import json
import random
import threading
import multiprocessing as mp
from dotenv import load_dotenv
from utils import get_current_location
import paho.mqtt.client as mqtt

# ── Disable obsensor BEFORE importing cv2 does anything ──────────────────────
os.environ["OPENCV_VIDEOIO_PRIORITY_OBSENSOR"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"]     = "0"

load_dotenv()
MODEL_ID = os.getenv("MODEL_ID")
API_KEY  = os.getenv("ROBOFLOW_API_KEY")

MQTT_BROKER       = "broker.emqx.io"
MQTT_PORT         = 1883
MQTT_TOPIC        = "humanrecog/video/jonathan_feed"
INFERENCE_EVERY_N = 12
WIDTH, HEIGHT     = 480, 360


# ═══════════════════════════════════════════════════════════════════
# CAMERA DETECTION
# ═══════════════════════════════════════════════════════════════════
def get_usb_camera_path() -> str | None:
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "--list-devices"],
            stderr=subprocess.DEVNULL, timeout=5
        ).decode()
    except Exception as e:
        print(f"[CAM] v4l2-ctl failed: {e}")
        return None

    in_usb = False
    for line in out.splitlines():
        stripped = line.strip()
        if not stripped:
            in_usb = False
            continue
        if not stripped.startswith("/dev/"):
            in_usb = "usb" in stripped.lower()
            continue
        if in_usb and stripped.startswith("/dev/video"):
            return stripped
    return None


# ═══════════════════════════════════════════════════════════════════
# FFMPEG CAPTURE THREAD
# Completely bypasses OpenCV's broken plugin system.
# ffmpeg talks directly to V4L2, pipes raw BGR frames to us.
# ═══════════════════════════════════════════════════════════════════
class FFmpegCaptureThread(threading.Thread):
    def __init__(self, device: str, width: int, height: int, fps: int = 30):
        super().__init__(daemon=True)
        self.device  = device
        self.width   = width
        self.height  = height
        self.fps     = fps
        self.frame   = None
        self.lock    = threading.Lock()
        self.running = True
        self.process = None

    def _build_cmd(self):
        return [
            "ffmpeg",
            "-loglevel", "error",          # suppress noise
            "-f", "v4l2",                  # force V4L2 — no plugin roulette
            "-input_format", "mjpeg",      # USB cams speak MJPEG natively (fast)
            "-video_size", f"{self.width}x{self.height}",
            "-framerate", str(self.fps),
            "-i", self.device,
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",           # OpenCV-native pixel format
            "pipe:1",                      # stream to stdout
        ]

    def run(self):
        frame_bytes = self.width * self.height * 3
        cmd = self._build_cmd()
        print(f"[CAM] ffmpeg capturing from {self.device}")

        while self.running:
            try:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=frame_bytes * 2
                )

                while self.running:
                    raw = self.process.stdout.read(frame_bytes)
                    if len(raw) != frame_bytes:
                        print("[CAM] ffmpeg pipe ended — restarting...")
                        break

                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (self.height, self.width, 3)
                    )
                    with self.lock:
                        self.frame = frame

            except Exception as e:
                print(f"[CAM] ffmpeg error: {e}")

            if self.process:
                self.process.kill()
                self.process = None

            if self.running:
                time.sleep(2)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        if self.process:
            self.process.kill()


def wait_for_camera(retries: int = 30, delay: float = 3.0) -> str:
    print("[CAM] Waiting for USB camera...")
    for attempt in range(retries):
        path = get_usb_camera_path()
        if path:
            print(f"[CAM] Found: {path}")
            return path
        print(f"[CAM] Not found yet (attempt {attempt+1}/{retries}), retrying in {delay}s...")
        time.sleep(delay)
    raise RuntimeError("[CAM] ❌ No USB camera found.")


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
                        boxes.append((int(x-w/2), int(y-h/2), int(x+w/2), int(y+h/2)))
            while not result_q.empty():
                try: result_q.get_nowait()
                except: pass
            result_q.put({"boxes": boxes, "detections": detections})
        except Exception as e:
            print(f"[INFERENCE PROC] Error: {e}")


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

    device  = wait_for_camera(retries=30, delay=3.0)
    capture = FFmpegCaptureThread(device, WIDTH, HEIGHT, fps=30)
    capture.start()

    # Wait until first frame arrives
    print("[CAM] Waiting for first frame...")
    for _ in range(60):
        if capture.read() is not None:
            break
        time.sleep(0.5)
    print("[SUCCESS] Camera stream live")

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
                try: frame_q.put_nowait(frame.copy())
                except: pass

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
            print(f"📡 {fps_count/elapsed:.1f} fps  |  {cached_detections} det  |  {lat:.6f}, {lng:.6f}")
            fps_count = 0
            fps_timer = time.time()

        time.sleep(0.001)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()