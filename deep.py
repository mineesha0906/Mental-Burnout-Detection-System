import cv2
import numpy as np
from deepface import DeepFace
from collections import defaultdict

# Comprehensive emotion configuration with subtypes for multiple emotions
EMOTIONS = {
    'angry': {
        'subtypes': {
            'annoyance': {'brow_furrow': 1, 'eye_squint': 1},
            'frustration': {'brow_furrow': 2, 'jaw_clench': 1, 'lip_press': 1},
            'rage': {'mouth_open': 2, 'brow_furrow': 3, 'eye_squint': 2},
        },
        'color': (0, 0, 255),
        'description': "Eyebrows lowered, eyes glaring, mouth tense"
    },
    'happy': {
        'subtypes': {
            'amusement': {'mouth_smile': 1},
            'joy': {'mouth_smile': 2, 'eye_crinkle': 1},
            'elation': {'mouth_smile': 3, 'eye_crinkle': 2},
        },
        'color': (0, 255, 0),
        'description': "Cheeks raised, mouth corners up"
    },
    'sad': {
        'subtypes': {
            'disappointment': {'mouth_frown': 1},
            'sorrow': {'mouth_frown': 2, 'brow_raise': 1},
            'grief': {'mouth_frown': 3, 'brow_raise': 2},
        },
        'color': (255, 0, 0),
        'description': "Inner eyebrows raised, mouth corners down"
    },
    'fear': {
        'subtypes': {
            'anxiety': {'eye_widen': 1},
            'alarm': {'eye_widen': 2, 'mouth_open': 1},
            'terror': {'eye_widen': 3, 'mouth_open': 2},
        },
        'color': (255, 255, 0),
        'description': "Eyes wide, eyebrows raised, mouth open"
    },
    'disgust': {'color': (0, 153, 0), 'description': "Nose wrinkled, upper lip raised"},
    'surprise': {'color': (0, 255, 255), 'description': "Eyebrows raised, jaw drop"},
    'neutral': {'color': (255, 255, 255), 'description': "Relaxed facial muscles"}
}

class CompleteEmotionDetector:
    def __init__(self):
        self.emotion_counts = defaultdict(int)
        self.subtype_counts = defaultdict(lambda: defaultdict(int))
        self.total_frames = 0
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # --- Feature Detectors for Each Emotion ---

    def detect_anger_features(self, roi_gray, w, h):
        features = {'brow_furrow': 0, 'eye_squint': 0, 'jaw_clench': 0, 'lip_press': 0, 'mouth_open': 0}
        
        # Brow Furrow: Look for vertical edges between eyebrows
        brow_region = roi_gray[h//5:h//2, w//3:2*w//3]
        sobel_x = cv2.Sobel(brow_region, cv2.CV_64F, 1, 0, ksize=3)
        if np.mean(np.abs(sobel_x)) > 20: features['brow_furrow'] = 2

        # Eye Squint: Check for small eye aspect ratio
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) > 0:
            avg_eye_height = np.mean([eh for (ex, ey, ew, eh) in eyes])
            if (avg_eye_height / h) < 0.1: features['eye_squint'] = 2

        # Mouth State: Use contour area
        mouth_roi = roi_gray[2*h//3:h, w//4:3*w//4]
        thresh = cv2.adaptiveThreshold(mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            area = cv2.contourArea(max(contours, key=cv2.contourArea))
            normalized_area = area / (mouth_roi.shape[0] * mouth_roi.shape[1])
            if normalized_area > 0.15: features['mouth_open'] = 2
            elif normalized_area < 0.02: features['lip_press'] = 2
            
        return features

    def detect_happy_features(self, roi_gray, w, h):
        features = {'mouth_smile': 0, 'eye_crinkle': 0}
        
        # Mouth Smile: Check for wide aspect ratio of the mouth contour
        mouth_roi = roi_gray[2*h//3:h, w//5:4*w//5]
        thresh = cv2.adaptiveThreshold(mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x_c, y_c, w_c, h_c = cv2.boundingRect(largest_contour)
            if w_c > 0 and h_c > 0 and (w_c / h_c) > 2.0:
                features['mouth_smile'] = 2
                
        # Eye Crinkle (Crow's Feet): Use Laplacian variance to detect fine wrinkles
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) > 0:
            eye_crinkle_variance = 0
            for (ex, ey, ew, eh) in eyes:
                # Check region to the side of the eye
                corner_roi = roi_gray[ey:ey+eh, max(0, ex-ew//2):ex]
                if corner_roi.size > 0:
                    eye_crinkle_variance += cv2.Laplacian(corner_roi, cv2.CV_64F).var()
            if eye_crinkle_variance / len(eyes) > 100: # High texture indicates wrinkles
                features['eye_crinkle'] = 2
        return features
        
    def detect_sad_features(self, roi_gray, w, h):
        features = {'mouth_frown': 0, 'brow_raise': 0}
        
        # Mouth Frown: Check for downward curve
        mouth_roi = roi_gray[2*h//3:h, w//4:3*w//4]
        thresh = cv2.adaptiveThreshold(mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
            # For a frown, the lower part of the bounding box will have more contour points
            M = cv2.moments(cnt)
            if M['m00'] > 0:
                cy = int(M['m01'] / M['m00'])
                if (cy / h_c) > 0.6: # Centroid is in the lower 40% of the box
                    features['mouth_frown'] = 2

        # Inner Brow Raise: Look for increased vertical space and texture above nose
        brow_center_roi = roi_gray[h//5:h//2, w//3:2*w//3]
        if cv2.Laplacian(brow_center_roi, cv2.CV_64F).var() > 60:
            features['brow_raise'] = 1
        return features

    def detect_fear_features(self, roi_gray, w, h):
        features = {'eye_widen': 0, 'mouth_open': 0}

        # Wide Eyes: Check for a more circular shape (aspect ratio close to 1)
        eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5)
        if len(eyes) > 0:
            aspect_ratios = []
            for (ex, ey, ew, eh) in eyes:
                if ew > 0:
                    aspect_ratios.append(eh / ew)
            avg_ratio = np.mean(aspect_ratios)
            if avg_ratio > 0.8: # Eyes are very round
                features['eye_widen'] = 2

        # Mouth Open (Jaw Drop): Check for tall mouth shape
        mouth_roi = roi_gray[2*h//3:h, w//4:3*w//4]
        thresh = cv2.adaptiveThreshold(mouth_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            x_c, y_c, w_c, h_c = cv2.boundingRect(cnt)
            if h_c > 0 and (w_c / h_c) < 1.5: # Mouth is taller/more circular than wide
                features['mouth_open'] = 2
        return features

    def determine_subtype(self, emotion, features):
        """Generic subtype determination based on feature scores."""
        scores = defaultdict(int)
        if emotion not in EMOTIONS or 'subtypes' not in EMOTIONS[emotion]:
            return ""

        for subtype, feature_map in EMOTIONS[emotion]['subtypes'].items():
            score = 0
            for feature, weight in feature_map.items():
                if features.get(feature, 0) > 0:
                    score += features[feature] * weight
            scores[subtype] = score
        
        if not any(scores.values()):
            return list(EMOTIONS[emotion]['subtypes'].keys())[0] # Default to first subtype
        return max(scores.items(), key=lambda x: x[1])[0]

    def process_frame(self, frame):
        self.total_frames += 1
        
        try:
            results = DeepFace.analyze(
                frame, actions=['emotion'],
                detector_backend='opencv', enforce_detection=False, silent=True
            )
            
            result = results[0]
            emotions = result['emotion']
            region = result['region']
            x, y, w, h = region['x'], region['y'], region['w'], region['h']
            
            roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
            dominant_emotion = result['dominant_emotion']
            
            feature_detectors = {
                'angry': self.detect_anger_features,
                'happy': self.detect_happy_features,
                'sad': self.detect_sad_features,
                'fear': self.detect_fear_features
            }

            if dominant_emotion in feature_detectors:
                features = feature_detectors[dominant_emotion](roi_gray, w, h)
                subtype = self.determine_subtype(dominant_emotion, features)
                self.subtype_counts[dominant_emotion][subtype] += 1

            self.emotion_counts[dominant_emotion] += 1
            self.draw_analysis(frame, emotions, (x, y, w, h))
            
        except Exception as e:
            pass # Suppress errors in live feed
            
        return frame

    def draw_analysis(self, frame, emotions, face_region):
        x, y, w, h = face_region
        dominant = max(emotions.items(), key=lambda x: x[1])
        dominant_emotion = dominant[0]
        color = EMOTIONS.get(dominant_emotion, {}).get('color', (255, 255, 255))
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        
        dashboard_height, dashboard_width = 220, 300
        dashboard = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
        
        cv2.putText(dashboard, "Emotion Analysis", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(dashboard, f"Dominant: {dominant_emotion.capitalize()}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # Display subtype if available
        if dominant_emotion in self.subtype_counts and self.subtype_counts[dominant_emotion]:
             current_subtype = max(self.subtype_counts[dominant_emotion].items(), key=lambda x: x[1])[0]
             cv2.putText(dashboard, f"Type: {current_subtype.capitalize()}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Display emotion bars
        y_offset = 110
        for emotion, percent in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:5]:
            bar_width = int((percent / 100) * (dashboard_width - 110))
            emo_color = EMOTIONS.get(emotion, {}).get('color', (255, 255, 255))
            cv2.putText(dashboard, f"{emotion.capitalize()}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, emo_color, 1)
            cv2.rectangle(dashboard, (100, y_offset-10), (100+bar_width, y_offset), emo_color, -1)
            y_offset += 22
            
        frame[10:10+dashboard_height, 10:10+dashboard_width] = dashboard

def main():
    detector = CompleteEmotionDetector()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Press 'Q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        frame = cv2.flip(frame, 1)
        frame = detector.process_frame(frame)
        
        cv2.imshow('Complete Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final report
    print("\n=== Final Emotion Report ===")
    print(f"Total frames processed: {detector.total_frames}")
    
    print("\n--- Main Emotion Distribution ---")
    for emotion, count in detector.emotion_counts.items():
        percentage = (count / detector.total_frames) * 100 if detector.total_frames > 0 else 0
        print(f"{emotion.capitalize():<10}: {count:>4} frames ({percentage:.1f}%)")
    
    print("\n--- Subtype Distribution ---")
    for emotion, subtypes in detector.subtype_counts.items():
        if subtypes:
            print(f"\n{emotion.capitalize()} Subtypes:")
            total_subtype_frames = sum(subtypes.values())
            for subtype, count in subtypes.items():
                percentage = (count / total_subtype_frames) * 100 if total_subtype_frames > 0 else 0
                print(f"  - {subtype.capitalize():<12}: {count:>4} frames ({percentage:.1f}%)")

if __name__ == "__main__":
    main()
