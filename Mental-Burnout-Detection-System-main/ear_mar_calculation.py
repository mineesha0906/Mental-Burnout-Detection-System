import cv2
import mediapipe as mp
import math
import time
from collections import deque

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Landmark indices (MediaPipe 468-point model)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
INNER_MOUTH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,  # Upper inner lip
               95, 88, 178, 87, 14, 317, 402, 318, 324]     # Lower inner lip

# Moving average windows
EAR_WINDOW = deque(maxlen=5)
MAR_WINDOW = deque(maxlen=5)

def calculate_ear(eye_points, landmarks):
    """Enhanced EAR calculation with blink detection"""
    # Vertical distances (upper-lower eyelids)
    v1 = math.dist(landmarks[eye_points[1]], landmarks[eye_points[5]])
    v2 = math.dist(landmarks[eye_points[2]], landmarks[eye_points[4]])
    
    # Horizontal distance (eye corners)
    h = math.dist(landmarks[eye_points[0]], landmarks[eye_points[3]])
    
    # Protection against division by zero
    ear = (v1 + v2) / (2.0 * h) if h > 0 else 0.0
    
    # Dramatic EAR drop detection (blink/closed eyes)
    if ear < 0.18:
        ear = 0.0
    
    return ear

def calculate_inner_mar(landmarks):
    """Pure inner-mouth MAR calculation (lip-size independent)"""
    # Get all inner mouth points
    points = [landmarks[i] for i in INNER_MOUTH]
    
    # Calculate vertical range (mouth openness)
    y_coords = [p[1] for p in points]
    mouth_height = max(y_coords) - min(y_coords)
    
    # Calculate horizontal reference (between cheekbones)
    cheek_width = math.dist(landmarks[454], landmarks[234])
    
    # Normalized MAR (ratio to face width)
    mar = mouth_height / cheek_width
    
    # Amplify for wide open mouth
    if mar > 0.15:  # Lower threshold since we're using inner mouth
        mar = min(mar * 1.8, 1.0)  # Cap at 1.0
    
    return mar

# Main loop
cap = cv2.VideoCapture(0)
last_print_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process frame
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        landmarks = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0])) 
                    for l in results.multi_face_landmarks[0].landmark]
        print(landmarks)
        # Calculate metrics
        left_ear = calculate_ear(LEFT_EYE, landmarks)
        right_ear = calculate_ear(RIGHT_EYE, landmarks)
        mar = calculate_inner_mar(landmarks)  # Using inner-mouth only
        
        # Update moving averages
        EAR_WINDOW.append((left_ear + right_ear) / 2)
        MAR_WINDOW.append(mar)
        smoothed_ear = sum(EAR_WINDOW) / len(EAR_WINDOW)
        smoothed_mar = sum(MAR_WINDOW) / len(MAR_WINDOW)
        
        # Visual feedback
        cv2.putText(frame, f"EAR: {smoothed_ear:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 0) if smoothed_ear > 0.3 else (0, 0, 255), 2)
        cv2.putText(frame, f"MAR: {smoothed_mar:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                    (0, 255, 0) if smoothed_mar < 0.15 else (0, 0, 255), 2)
        
        # Print values every second
        if time.time() - last_print_time >= 1.0:
            status = []
            if smoothed_ear < 0.2:
                status.append("EYE CLOSURE")
            if smoothed_mar > 0.15:  # Lower threshold for inner mouth
                status.append("MOUTH OPEN")
            
            print(
                f"EAR: {smoothed_ear:.2f} | "
                f"MAR: {smoothed_mar:.2f} | "
                f"Status: {', '.join(status) if status else 'Normal'}"
            )
            last_print_time = time.time()
        
        # Draw landmarks
        for idx in LEFT_EYE + RIGHT_EYE:
            cv2.circle(frame, landmarks[idx], 1, (0, 255, 255), -1)  # Yellow for eyes
        for idx in INNER_MOUTH:
            cv2.circle(frame, landmarks[idx], 1, (0, 0, 255), -1)  # Red for inner mouth
    
    cv2.imshow('Inner-Mouth MAR Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
