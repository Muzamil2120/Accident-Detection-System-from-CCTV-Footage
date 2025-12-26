

import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Store previous positions to calculate movement
previous_positions = {}

def detect_accident(track_id, cx, cy):
    """Simple accident logic:
       If vehicle suddenly stops → possible accident"""
    if track_id not in previous_positions:
        previous_positions[track_id] = (cx, cy)
        return False

    px, py = previous_positions[track_id]
    movement = np.sqrt((cx - px)**2 + (cy - py)**2)

    previous_positions[track_id] = (cx, cy)

    # Very small movement = possible crash
    if movement < 2:
        return True
    return False


# --------------------------- MAIN PROTOTYPE LOGIC ---------------------------
def run_prototype(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Error: Unable to load video!")
        return

    print("▶ Prototype Running... Press ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # AI Detection
        results = model(frame, conf=0.4)

        for r in results:
            for obj in r.boxes:
                cls = int(obj.cls[0])
                conf = float(obj.conf[0])
                
                # Only detect vehicles (COCO classes)
                if cls not in [2, 3, 5, 7]:  
                    continue

                x1, y1, x2, y2 = obj.xyxy[0]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                track_id = int(obj.id[0]) if obj.id is not None else cx  # temporary track ID

                # Basic accident detection
                accident = detect_accident(track_id, cx, cy)

                # Draw detection
                color = (0, 255, 0)
                if accident:
                    color = (0, 0, 255)
                    cv2.putText(frame, "ACCIDENT!", (cx - 40, cy - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        cv2.imshow("Prototype - Traffic Accident Detection", frame)

        if cv2.waitKey(1) == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✔ Prototype Finished.")


# --------------------------------------------------------------------------
# SIMPLE USER INTERACTION
# --------------------------------------------------------------------------
video_path = "sample_video.mp4"  # Replace with your video or keep sample
run_prototype(video_path)
