import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# Store vehicle data
vehicle_data = {}
accident_log = []
current_accidents = {}
accident_display_frames = {}

def get_distance(pos1, pos2):
    """Calculate distance between two positions"""
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def classify_accident_severity(speed_diff):
    """Classify accident as serious or safe based on speed difference"""
    if speed_diff > 15:
        return "SERIOUS", (0, 0, 255)
    elif speed_diff > 5:
        return "MODERATE", (0, 165, 255)
    else:
        return "SAFE", (0, 255, 255)

def detect_vehicle_collision(track_id, cx, cy, current_frame):
    """Detect if vehicle collides with other vehicles"""
    collision_detected = False
    
    if track_id not in vehicle_data:
        vehicle_data[track_id] = {
            'positions': [(cx, cy)],
            'speed': 0
        }
    else:
        prev_pos = vehicle_data[track_id]['positions'][-1]
        speed = get_distance((cx, cy), prev_pos)
        vehicle_data[track_id]['speed'] = speed
        vehicle_data[track_id]['positions'].append((cx, cy))
        
        if len(vehicle_data[track_id]['positions']) > 5:
            vehicle_data[track_id]['positions'].pop(0)
    
    collision_threshold = 35
    for other_id, other_data in vehicle_data.items():
        if other_id == track_id or not other_data['positions']:
            continue
        
        other_pos = other_data['positions'][-1]
        distance = get_distance((cx, cy), other_pos)
        
        if distance < collision_threshold and vehicle_data[track_id]['speed'] > 1:
            collision_detected = True
            
            speed_diff = abs(vehicle_data[track_id]['speed'] - other_data['speed'])
            severity, _ = classify_accident_severity(speed_diff)
            
            collision_key = f"{min(track_id, other_id)}-{max(track_id, other_id)}"
            
            # Only log accident ONCE when first collision detected
            if collision_key not in current_accidents and collision_key not in accident_display_frames:
                current_accidents[collision_key] = {
                    'frame': current_frame,
                    'severity': severity,
                    'vehicle1': track_id,
                    'vehicle2': other_id,
                    'speed_diff': speed_diff
                }
                accident_log.append(current_accidents[collision_key])
                accident_display_frames[collision_key] = current_frame + 300  # Display for 300 frames (longer)
    
    return collision_detected

def run_prototype(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Error: Unable to load video!")
        return

    # Get video dimensions
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Scale if video is too large (max 1200 width)
    scale = 1.0
    if video_width > 1200:
        scale = 1200 / video_width
    
    display_width = int(video_width * scale)
    display_height = int(video_height * scale)
    
    print(f"▶ Video Size: {video_width}x{video_height}")
    print(f"▶ Display Size: {display_width}x{display_height}")
    print("▶ Prototype Running... Press ESC to exit")
    
    frame_count = 0
    
    # Set window with video dimensions
    cv2.namedWindow("Traffic Accident Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Traffic Accident Detection", display_width, display_height + 180)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Resize frame to display size
        frame = cv2.resize(frame, (display_width, display_height))
        
        results = model(frame, conf=0.4)

        detected_collisions = {}

        for r in results:
            for obj in r.boxes:
                cls = int(obj.cls[0])
                
                # Only detect vehicles
                if cls not in [2, 3, 5, 7]:  
                    continue

                x1, y1, x2, y2 = obj.xyxy[0]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                track_id = int(obj.id[0]) if obj.id is not None else int(cx)

                collision = detect_vehicle_collision(track_id, cx, cy, frame_count)
                
                if collision:
                    detected_collisions[track_id] = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (cx, cy)
                    }

        # Draw boxes with vehicle numbers ONLY for colliding vehicles
        for track_id, vehicle_info in detected_collisions.items():
            x1, y1, x2, y2 = vehicle_info['bbox']
            cx, cy = vehicle_info['center']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 255), 1)
            
            # Show vehicle number
            cv2.putText(frame, f"V{track_id}", (x1+5, y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Display accident impact details in panel
        h, w, _ = frame.shape
        panel_height = 180
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        # Show active accidents (only display once for 150 frames)
        active_accidents = []
        for acc_key, end_frame in list(accident_display_frames.items()):
            if frame_count <= end_frame:
                # Find accident in log
                for acc in accident_log:
                    key = f"{min(acc['vehicle1'], acc['vehicle2'])}-{max(acc['vehicle1'], acc['vehicle2'])}"
                    if key == acc_key:
                        active_accidents.append(acc)
            else:
                # Remove expired accident display
                del accident_display_frames[acc_key]
        
        if active_accidents:
            y_offset = 20
            cv2.putText(panel, "*** ACCIDENT IMPACT DETECTED ***", (15, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            y_offset += 30
            
            # Show ONLY the latest accident (not multiple)
            for accident in active_accidents[-1:]:
                severity_color = (0, 0, 255) if accident['severity'] == "SERIOUS" else (0, 165, 255) if accident['severity'] == "MODERATE" else (0, 255, 255)
                text = f"Vehicle {accident['vehicle1']} & Vehicle {accident['vehicle2']} | {accident['severity']} | Speed Diff: {accident['speed_diff']:.1f}px/frame"
                cv2.putText(panel, text, (15, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, severity_color, 1)
                y_offset += 30
        else:
            cv2.putText(panel, "Normal Traffic - No Accidents Detected", (15, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Combine frame with panel
        output = np.vstack([frame, panel])
        cv2.imshow("Traffic Accident Detection", output)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✔ Prototype Finished.")
    print(f"\n📊 Total Accidents Detected: {len(accident_log)}")

if __name__ == "__main__":
    video_path = "sample_video.mp4"  # Use your own video file
    run_prototype(video_path)
