import cv2
import time
from utils import get_current_location

class RecognitionPipeline:
    def __init__(self):
        # Initialize Haar Cascade for full body detection
        # Ensure the xml file is available or use the built-in one from cv2.data.haarcascades
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
        
        # Also init face detector as backup/alternative since webcam usually sees faces better
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        
        self.last_geo_time = 0
        self.geo_interval = 5.0 # Seconds between geolocation checks to avoid spamming
        self.current_location = "Unknown"
        
        print("Initialization Complete: Recognition Pipeline ready.")

    def process_frame(self, frame):
        """
        Process a single frame.
        Input: frame (numpy array from cv2)
        Output: frame (annotated)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect bodies
        bodies = self.detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=3, 
            minSize=(30, 30)
        )
        
        # Detect faces (often works better for webcam testing)
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detections = list(bodies) + list(faces)
        
        found_human = len(detections) > 0
        
        if found_human:
            current_time = time.time()
            # Check geolocation periodically
            if current_time - self.last_geo_time > self.geo_interval:
                print("[INFO] Human detected! Fetching location...")
                self.current_location = get_current_location()
                print(f"[RESULT] Human identified at: {self.current_location}")
                self.last_geo_time = current_time
        
        # Draw bounding boxes
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Human", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        for (x, y, w, h) in faces:
             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
             cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if found_human:
             cv2.putText(frame, f"Loc: {self.current_location}", (10, frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame
