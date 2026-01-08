import cv2
import time
import os
from dotenv import load_dotenv
from inference import get_model
from utils import get_current_location

# Load environment variables
load_dotenv()

class RecognitionPipeline:
    def __init__(self):
        # Get credentials from environment variables
        api_key = os.getenv('ROBOFLOW_API_KEY')
        model_id = os.getenv('MODEL_ID', 'drone-3x7oz/2')
        
        if not api_key:
            raise ValueError("ROBOFLOW_API_KEY not found in .env file")
        
        # Initialize Roboflow model (downloads and runs locally)
        print(f"Loading model '{model_id}'... (first time may take a while)")
        self.model = get_model(
            model_id=model_id,
            api_key=api_key
        )
        
        self.last_geo_time = 0
        self.geo_interval = 5.0  # Seconds between geolocation checks
        self.current_location = "Unknown"
        
        # Detection confidence threshold
        self.confidence_threshold = 0.5
        
        print(f"Initialization Complete: Local model ready for fast inference!")

    def process_frame(self, frame):
        """
        Process a single frame using local Roboflow inference (FAST!).
        Input: frame (numpy array from cv2)
        Output: frame (annotated)
        """
        
        # Store original dimensions for scaling predictions back
        original_height, original_width = frame.shape[:2]
        
        # Resize frame to 320x320 for faster inference
        small_frame = cv2.resize(frame, (320, 320))
        
        # Run inference locally on smaller frame (FAST - no network delay!)
        results = self.model.infer(small_frame)
        
        # Extract predictions
        predictions = results[0].predictions if hasattr(results[0], 'predictions') else []
        
        found_detection = len(predictions) > 0
        
        if found_detection:
            current_time = time.time()
            # Check geolocation periodically
            if current_time - self.last_geo_time > self.geo_interval:
                print(f"[INFO] Detected {len(predictions)} object(s)! Fetching location...")
                self.current_location = get_current_location()
                print(f"[RESULT] Detection at: {self.current_location}")
                self.last_geo_time = current_time
        
        # Draw bounding boxes and labels
        for prediction in predictions:
            # Get bounding box coordinates (scale back to original frame size)
            scale_x = original_width / 320
            scale_y = original_height / 320
            
            x = int((prediction.x - prediction.width / 2) * scale_x)
            y = int((prediction.y - prediction.height / 2) * scale_y)
            w = int(prediction.width * scale_x)
            h = int(prediction.height * scale_y)
            
            confidence = prediction.confidence
            class_name = prediction.class_name
            
            # Only draw if confidence is above threshold
            if confidence >= self.confidence_threshold:
                # Draw rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw label with confidence
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display location and detection count
        if found_detection:
            cv2.putText(frame, f"Detections: {len(predictions)} | Loc: {self.current_location}", 
                       (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame
