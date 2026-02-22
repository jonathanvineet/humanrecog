# ==========================================
# FORCE X11 FOR RASPBERRY PI
# ==========================================
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Disable unused Roboflow backends
os.environ["CORE_MODEL_SAM_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM2_ENABLED"] = "False"
os.environ["CORE_MODEL_SAM3_ENABLED"] = "False"
os.environ["CORE_MODEL_GAZE_ENABLED"] = "False"
os.environ["CORE_MODEL_YOLO_WORLD_ENABLED"] = "False"

import warnings
warnings.filterwarnings("ignore")

# ==========================================
# IMPORTS
# ==========================================
import time
import cv2
from dotenv import load_dotenv
from inference import get_model

# ==========================================
# SIMPLE FPS CLASS
# ==========================================
class FPS:
    def __init__(self):
        self.start_time = time.time()
        self.frames = 0

    def update(self):
        self.frames += 1

    def fps(self):
        elapsed = time.time() - self.start_time
        return self.frames / elapsed if elapsed > 0 else 0


# ==========================================
# MAIN FUNCTION
# ==========================================
def main():
    load_dotenv()

    print("[INFO] Starting camera...")
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Camera not opened")
        return

    print("[INFO] Loading model...")
    model = get_model(
        model_id=os.getenv("MODEL_ID"),
        api_key=os.getenv("ROBOFLOW_API_KEY")
    )

    print("[INFO] Running... Press Q to quit.")

    fps = FPS()
    frame_count = 0
    last_detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        display_frame = frame.copy()
        frame_count += 1

        # ======================================
        # RUN INFERENCE ONLY EVERY 5 FRAMES
        # ======================================
        if frame_count % 5 == 0:
            try:
                results = model.infer(frame)[0]
                last_detections = []

                if hasattr(results, "predictions"):
                    for prediction in results.predictions:
                        if prediction.class_name.lower() in ["person", "human"]:
                            last_detections.append(prediction)

            except Exception as e:
                print("Inference error:", e)

        # ======================================
        # DRAW LAST DETECTIONS (NON-BLOCKING)
        # ======================================
        for prediction in last_detections:
            x = int(prediction.x)
            y = int(prediction.y)
            w = int(prediction.width)
            h = int(prediction.height)

            x1 = int(x - w / 2)
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)
            y2 = int(y + h / 2)

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display_frame,
                f"Human {prediction.confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

        # ======================================
        # SHOW FPS
        # ======================================
        fps.update()
        cv2.putText(
            display_frame,
            f"FPS: {fps.fps():.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # ======================================
        # DISPLAY WINDOW (ALWAYS RUNS)
        # ======================================
        cv2.imshow("Human Recognition System", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("[INFO] Closing...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
