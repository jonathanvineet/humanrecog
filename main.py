import cv2
import time
from video_stream import WebcamStream
from utils import FPS
from processor import RecognitionPipeline

def main():
    print("[INFO] sampling THREADED frames from webcam...")
    
    # Initialize the video stream and allow the camera sensor to warm up
    # src=0 is usually the default webcam. Change to 1 or a file path if needed.
    vs = WebcamStream(src=0).start()
    time.sleep(2.0)
    
    # Initialize the processing pipeline
    pipeline = RecognitionPipeline()
    
    # Initialize the FPS throughput estimator
    fps = FPS().start()

    # Loop over some frames...this time using the threaded stream
    while True:
        # Grab the frame from the threaded video stream
        frame = vs.read()
        
        # If no frame is received (end of stream or error), break
        if frame is None:
            break
            
        # --- PROCESSING START ---
        
        # Resize frame if needed for performance (optional, e.g. width=640)
        # frame = cv2.resize(frame, (640, 480))
        
        # Run the detection/recognition pipeline
        frame = pipeline.process_frame(frame)
        
        # --- PROCESSING END ---

        # Show the output frame
        # Draw the FPS on the frame
        cv2.putText(frame, "FPS: {:.2f}".format(fps.fps()), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
        cv2.imshow("Human Recognition System", frame)
        
        # Update the FPS counter
        fps.update()
        
        # Check for the 'q' key to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # Do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()
