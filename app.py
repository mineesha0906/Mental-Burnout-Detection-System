import cv2
import mediapipe as mp
import math
import gradio as gr
import numpy as np
import threading
import time
from deepface import DeepFace
import speech_recognition as sr
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter, defaultdict
import joblib
import re
import os

# Configuration
RUN_DURATION = 20

EMOTION_BUCKETS = {
    'positive': ['happy'],
    'neutral': ['neutral'],
    'negative': ['angry', 'sad', 'fear', 'disgust', 'surprise']
}

sentiment_analyzer = SentimentIntensityAnalyzer()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
INNER_MOUTH = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324]

questions = [
    "Do you often feel exhausted or stressed after classes and hostel work, even with enough sleep?",
    "Do you feel like you're just going through the motions, not really enjoying hostel or college life?",
    "Do you avoid people or group work because you just don’t feel like talking?",
    "Do you feel proud of the progress you're making in your academics or personal goals?",
    "Do you feel that your efforts are leading to meaningful outcomes or recognition?",
]
options = [
    ["Never", "Rarely", "Sometimes", "Often", "Always"],
    ["Never", "Rarely", "Sometimes", "Often", "Always"],
    ["Never", "Rarely", "Sometimes", "Often", "Always"],
    ["Never", "Rarely", "Sometimes", "Often", "Always"],
    ["Never", "Rarely", "Sometimes", "Often", "Always"],
]

# ---------------------------
# Emotion Detector (unchanged logic, but safe to call from a thread)
# ---------------------------
class CompleteEmotionDetector:
    def __init__(self):
        self.bucket_counts = defaultdict(int)
        self.lock = threading.Lock()
        
    def _get_emotion_bucket(self, emotion):
        for bucket, emotions_list in EMOTION_BUCKETS.items():
            if emotion in emotions_list:
                return bucket
        return "unknown"

    def analyze_frame(self, frame):
        try:
            results = DeepFace.analyze(
                frame, actions=['emotion'],
                detector_backend='mtcnn', enforce_detection=False, silent=True
            )
            with self.lock:
                if results:
                    dominant_emotion_deepface = results[0].get('dominant_emotion', 'neutral')
                    bucket = self._get_emotion_bucket(dominant_emotion_deepface)
                    self.bucket_counts[bucket] += 1
        except Exception:
            pass

    def get_stats(self):
        with self.lock:
            total = sum(self.bucket_counts.values())
            pos = (self.bucket_counts['positive'] / total) * 100 if total > 0 else 0.0
            neu = (self.bucket_counts['neutral'] / total) * 100 if total > 0 else 0.0
            neg = (self.bucket_counts['negative'] / total) * 100 if total > 0 else 0.0
            return pos, neu, neg
# ---------------------------
# Face Analyzer (restructured for per-frame processing)
# ---------------------------
# --- Face Analyzer Class ---
class FaceAnalyzer:
    def __init__(self, emotion_detector_instance):
        # Initialize lock FIRST
        self.face_lock = threading.Lock()
        self.emotion_detector = emotion_detector_instance
        self.cap = None
        self.running = False
        # Now call reset
        self.reset()

    def reset(self):
        with self.face_lock:
            self.ear_values = []
            self.mar_values = []
            self.start_time = None
            self.running = False

    def calculate_ear(self, eye_points, landmarks):
        """Calculate Eye Aspect Ratio"""
        try:
            # Vertical distances
            v1 = math.dist(landmarks[eye_points[1]], landmarks[eye_points[5]])
            v2 = math.dist(landmarks[eye_points[2]], landmarks[eye_points[4]])
            # Horizontal distance
            h = math.dist(landmarks[eye_points[0]], landmarks[eye_points[3]])
            return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
        except:
            return 0.0

    def calculate_mar(self, landmarks):
        """Calculate Mouth Aspect Ratio using inner mouth points"""
        try:
            points = [landmarks[i] for i in INNER_MOUTH]
            if len(points) < 2:
                return 0.0
            
            # Get mouth height (vertical distance)
            y_coords = [p[1] for p in points]
            mouth_height = max(y_coords) - min(y_coords)
            
            # Get mouth width using outer points (approximate)
            if len(landmarks) > 60:
                mouth_width = math.dist(landmarks[61], landmarks[291])  # Left to right mouth corner
            else:
                mouth_width = math.dist(points[0], points[6]) if len(points) > 6 else 1.0
            
            return mouth_height / mouth_width if mouth_width > 0 else 0.0
        except:
            return 0.0

    def start_run(self):
        with self.face_lock:
            self.ear_values = []
            self.mar_values = []
            self.running = True
            self.start_time = time.time()
        
        # Reset emotion detector counts for new session
        with self.emotion_detector.lock:
            self.emotion_detector.bucket_counts = defaultdict(int)
            self.emotion_detector.detailed_emotion_counts = defaultdict(int)
            self.emotion_detector.subtype_counts = defaultdict(lambda: defaultdict(int))

    def process_frame(self, frame):
        """Process a single frame for EAR, MAR, and emotion analysis"""
        if not self.running:
            return frame, "---", "---", "Not Running"

        elapsed = time.time() - self.start_time
        if elapsed >= RUN_DURATION:
            self.running = False
            return frame, "---", "---", "Stopped (Time reached)"

        # Convert to RGB for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        ear, mar = None, None
        
        if results.multi_face_landmarks:
            try:
                face_landmarks = results.multi_face_landmarks[0]
                landmarks = [(int(p.x * frame.shape[1]), int(p.y * frame.shape[0])) 
                             for p in face_landmarks.landmark]

                # Calculate EAR and MAR
                left_ear = self.calculate_ear(LEFT_EYE, landmarks)
                right_ear = self.calculate_ear(RIGHT_EYE, landmarks)
                ear = (left_ear + right_ear) / 2.0
                mar = self.calculate_mar(landmarks)

                with self.face_lock:
                    if ear is not None:
                        self.ear_values.append(ear)
                    if mar is not None:
                        self.mar_values.append(mar)

                # Draw landmarks for visualization
                for idx in LEFT_EYE + RIGHT_EYE + INNER_MOUTH:
                    if idx < len(landmarks):
                        frame = frame.copy()
                        cv2.circle(frame, tuple(map(int, landmarks[idx])), 1, (0, 255, 0), -1)

                # Display metrics on frame
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"MAR: {mar:.2f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"Landmark processing error: {e}")
                ear, mar = 0.3, 0.4  # Default values if processing fails

        # Run emotion detection in background
        try:
            if not hasattr(self, 'emotion_thread') or not self.emotion_thread.is_alive():
                frame_copy = frame.copy()
                self.emotion_thread = threading.Thread(
                    target=self.emotion_detector.analyze_frame, 
                    args=(frame_copy,),
                    daemon=True
                )
                self.emotion_thread.start()
        except Exception as e:
            print(f"Emotion thread error: {e}")

        # Add timer overlay
        remaining_time = max(0, RUN_DURATION - int(time.time() - self.start_time))
        timer_text = f"Time left: {remaining_time}s"
        text_size = cv2.getTextSize(timer_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = frame.shape[1] - text_size[0] - 20
        text_y = text_size[1] + 20
        
        # Create semi-transparent background for timer
        overlay = frame.copy()
        cv2.rectangle(overlay, (text_x - 10, text_y - text_size[1] - 10),
                     (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
        alpha = 0.4
        processed_frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        cv2.putText(processed_frame, timer_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return (cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
               f"{ear:.2f}" if ear is not None else "---",
               f"{mar:.2f}" if mar is not None else "---",
               f"Running ({elapsed:.1f}s)")

    def finalize_stats(self):
        """Compute final statistics after analysis completes"""
        with self.face_lock:
            if not self.ear_values or not self.mar_values:
                return (0, 0, 0, 0), (0, 0, 0)

            avg_ear = np.mean(self.ear_values)
            std_ear = np.std(self.ear_values)
            avg_mar = np.mean(self.mar_values)
            std_mar = np.std(self.mar_values)

        # Get emotion statistics from the detector
        emotion_stats = self.emotion_detector.get_stats()

        return (avg_ear, std_ear, avg_mar, std_mar), emotion_stats

    def stop(self):
        """Stop the analysis and clean up resources"""
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'emotion_thread') and self.emotion_thread.is_alive():
            self.emotion_thread.join(timeout=1.0)


# --- Initialize global objects ---
emotion_detector = CompleteEmotionDetector()
face_analyzer = FaceAnalyzer(emotion_detector)  # Pass the emotion detector
stored_data = {}

# Voice Analysis Functions
def analyze_voice(audio):

    max_secs = 10
    if isinstance(audio, tuple) and len(audio) == 2:
        sample_rate, audio_data = audio
        max_samples = int(sample_rate * max_secs)
        if len(audio_data.shape) == 2:  # stereo
            audio_data = audio_data[:max_samples, :]
        else:
            audio_data = audio_data[:max_samples]
    import tempfile
    import scipy.io.wavfile as wav
    import speech_recognition as sr
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import numpy as np

    sentiment_analyzer = SentimentIntensityAnalyzer()

    if audio is None:
        return "No audio input.", "N/A", "N/A"

    audio_path = None
    try:
        if isinstance(audio, tuple) and len(audio) == 2:
            sample_rate, audio_data = audio
            if np.issubdtype(audio_data.dtype, np.floating):
                if np.max(np.abs(audio_data)) <= 1.01:
                    audio_data = (audio_data * 32767).astype(np.int16)
                else:
                    audio_data = audio_data.astype(np.int16)
            temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            wav.write(temp.name, sample_rate, audio_data)
            audio_path = temp.name
        elif isinstance(audio, str):
            audio_path = audio
        else:
            return "Unrecognized audio input", "N/A", "N/A"

        r = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio_sr = r.record(source)
        try:
            text = r.recognize_google(audio_sr)
        except (sr.UnknownValueError, sr.RequestError):
            text = ""
        sentiment = sentiment_analyzer.polarity_scores(text) if text else {"pos": 0, "neu": 1, "neg": 0, "compound": 0}
        sentiment_str = (
            f"Pos: {sentiment['pos']:.2f}, Neu: {sentiment['neu']:.2f}, "
            f"Neg: {sentiment['neg']:.2f}, Comp: {sentiment['compound']:.2f}"
        )
        return text, sentiment_str, "N/A"  # Return only the 3 expected values
    except Exception as e:
        return f"Error: {str(e)}", "N/A", "N/A"

def aggregate_results(questionnaire_answers, face_stats, emotion_stats, voice_stats, subtype_counts):
    print("face_stats:", face_stats)
    print("emotion_stats:", emotion_stats)
    print("voice_stats:", voice_stats)
    avg_ear, std_ear, avg_mar, std_mar = face_stats
    pos_perc, neu_perc, neg_perc = emotion_stats
    text, sentiment_scores, pitch = voice_stats
    subtype_table = []
    for emotion, subtypes in subtype_counts.items():
        for subtype, count in subtypes.items():
            subtype_table.append([emotion, subtype, count])
    result_dict = {
        "Questionnaire Answers": questionnaire_answers,
        "EAR": f"{avg_ear:.2f} ± {std_ear:.2f}",
        "MAR": f"{avg_mar:.2f} ± {std_mar:.2f}",
        "Emotions": {
            "Positive %": f"{pos_perc:.1f}%",
            "Neutral %": f"{neu_perc:.1f}%",
            "Negative %": f"{neg_perc:.1f}%"
        },
        "Voice Transcript": text,
        "Sentiment": sentiment_scores,
        "Pitch": "N/A",
        "Subtypes": subtype_table
    }
    return result_dict

answer_mapping = {
    "Never": 0,
    "Rarely": 1,
    "Sometimes": 2,
    "Often": 3,
    "Always": 4,
    "I enjoy my work. I have no symptoms of burnout": 0,
    "Occasionally I am under stress, and I don’t always have as much energy as I once did, but I don’t feel burned out": 1,
    "I am definitely burning out and have one or more symptoms of burnout, such as physical and emotional exhaustion": 2,
    "The symptoms of burnout that I’m experiencing won’t go away. I think about frustration at work a lot": 3,
    "I feel completely burned out and often wonder if I can go on. I am at the point where I may need some changes or may need to seek some sort of help": 4,
}
try:
    scaler=joblib.load('scaler.joblib')
    model = joblib.load('model.joblib')
except FileNotFoundError:
    gr.Warning("Model files not found. The app will run, but prediction will not work.")
    scaler = None
    model = None

def preprocess_input(raw_dict, scaler):
    # Map questionnaire answers to ints
    q_encoded = [answer_mapping.get(ans, -1) for ans in raw_dict["Questionnaire Answers"]]
    if -1 in q_encoded:
        raise ValueError("Unknown answer detected in questionnaire answers.")

    # Parse EAR and MAR (only average, ignore std)
    EAR_avg = float(raw_dict["EAR"].split(" ± ")[0])
    MAR_avg = float(raw_dict["MAR"].split(" ± ")[0])

    # Parse emotions percentages (remove % and convert)
    emotions = raw_dict["Emotions"]
    pos_emotion = float(emotions["Positive %"].rstrip('%'))
    neu_emotion = float(emotions["Neutral %"].rstrip('%'))
    neg_emotion = float(emotions["Negative %"].rstrip('%'))

    # Parse sentiment scores (extract floats from string)
    sentiment_str = raw_dict["Sentiment"]
    sentiment_vals = re.findall(r'[-+]?\d*\.\d+|\d+', sentiment_str)
    # Expected order: Pos, Neu, Neg, Comp
    sentiment_vals = list(map(float, sentiment_vals))
    if len(sentiment_vals) != 4:
        raise ValueError("Sentiment string format unexpected.")
    sent_pos, sent_neu, sent_neg, sent_comp = sentiment_vals

    # Build input feature vector as per model's expected order:
    features = q_encoded + [
        EAR_avg, 0,  # Avg_EAR, placeholder for Std_EAR (not provided)
        MAR_avg, 0,  # Avg_MAR, placeholder for Std_MAR (not provided)
        pos_emotion, neu_emotion, neg_emotion,
        sent_pos, sent_neu, sent_neg, sent_comp
    ]

    # Convert to numpy array 2D for scaler and model
    X = np.array(features).reshape(1, -1)

    # Apply scaler
    X_scaled = scaler.transform(X)

    # Optionally convert to pandas DataFrame with feature names if needed
    # feature_names = [...] # from model.feature_names_
    # X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    return X_scaled

def predict_burnout(raw_dict):
    if not scaler or not model:
        raise RuntimeError("Model files not loaded. Cannot predict.")

    X_processed = preprocess_input(raw_dict, scaler)
    pred = model.predict(X_processed)
    # Assuming regression or float prediction; if classifier, adapt accordingly
    output_score = float(pred[0])
    return output_score

# --- PDF Download Function ---
def download_info_pdf():
    pdf_path = "info.pdf"
    if os.path.exists(pdf_path):
        return pdf_path
    else:
        print(f"Error: {pdf_path} not found.")
        return None
    
with gr.Blocks() as demo:
    # This is the JavaScript from your previous prompt to force the light theme.
    demo.load(
        None,
        None,
        js="""
        () => {
            const params = new URLSearchParams(window.location.search);
            if (!params.has('__theme')) {
                params.set('__theme', 'light');
                window.location.search = params.toString();
            }
        }
        """,
    )

    # Functionality Section
    with gr.Column(elem_classes=["functionality-section"], elem_id="functionality-section"):
        
        gr.Markdown('<h2 class="functionality-title">Try it out!</h2>')
        with gr.Tabs(elem_classes=["gradio-tabs"]) as tabs:
            # Questionnaire
            with gr.TabItem("Questionnaire", id=0):
                with gr.Column(elem_classes=["scrollable"]):
                    radio_components = []
                    for i, (q, opts) in enumerate(zip(questions, options)):
                        with gr.Row():
                            gr.Markdown(f"**{i+1}. {q}**")
                            r = gr.Radio(opts, label="Your answer")
                            radio_components.append(r)
                submit_btn = gr.Button("Submit Answers")

            # Face Analysis
            with gr.TabItem("Face Analysis", id=1):
                with gr.Column():
                    webcam = gr.Image(label="Webcam Feed", sources="webcam", streaming=True)
                    with gr.Row():
                        ear_output = gr.Textbox(label="Avg EAR ± Std", value="---")
                        mar_output = gr.Textbox(label="Avg MAR ± Std", value="---")
                    with gr.Row():
                        pos_emotion_output = gr.Textbox(label="Positive Emotion %", value="---")
                        neu_emotion_output = gr.Textbox(label="Neutral Emotion %", value="---")
                        neg_emotion_output = gr.Textbox(label="Negative Emotion %", value="---")
                    status_output = gr.Textbox(label="Status", value="Ready")
                    start_face_btn = gr.Button("Start Face Analysis")
                    update_stats_btn = gr.Button("Update Stats")
                    next_to_voice_btn = gr.Button("Proceed to Voice Analysis", interactive=False)

            # Voice Analysis
            with gr.TabItem("Voice Analysis", id=2):
                audio_in = gr.Audio(label="Your Recording", sources="microphone", streaming=False)
                record_btn = gr.Button("Analyze Voice")
                text_output = gr.Textbox(label="Transcribed Text")
                sentiment_output = gr.Textbox(label="Sentiment Analysis (Pos, Neu, Neg, Comp)")
                pitch_output = gr.Textbox(label="Pitch (Hz) ± Std Dev")
                proceed_to_results_btn = gr.Button("Proceed to Suggestions")

            # Suggestions
            with gr.TabItem("Suggestions", id=3):
                suggestion_output = gr.Textbox(label="Suggestion based on Burnout Score", interactive=False)


    # Questionnaire submit → store answers + go to Face Analysis tab
    def questionnaire_to_face(*answers):
        stored_data["questionnaire"] = list(answers)
        return gr.Tabs(selected=1)

    submit_btn.click(
        fn=questionnaire_to_face,
        inputs=radio_components,
        outputs=tabs,
    )

    # Start face analysis
    def start_face_analysis():
        face_analyzer.start_run()
        return "Started", gr.update(interactive=False)

    start_face_btn.click(fn=start_face_analysis, outputs=[status_output, next_to_voice_btn])

    # Webcam streaming
    def stream_frame(frame):
        return face_analyzer.process_frame(frame)

    webcam.stream(
        fn=stream_frame,
        inputs=[webcam],
        outputs=[webcam, ear_output, mar_output, status_output],
    )

    # Update stats after face analysis
    def update_stats_button():
        face_stats, emotion_stats = face_analyzer.finalize_stats()
        stored_data["face"] = face_stats
        stored_data["emotion"] = emotion_stats
        stored_data["subtypes"] = dict(emotion_detector.subtype_counts)
        avg_ear, std_ear, avg_mar, std_mar = face_stats
        pos_perc, neu_perc, neg_perc = emotion_stats
        avg_ear_s = f"{avg_ear:.2f} ± {std_ear:.2f}"
        avg_mar_s = f"{avg_mar:.2f} ± {std_mar:.2f}"
        return (
            avg_ear_s,
            avg_mar_s,
            "Finished Face Analysis",
            f"{pos_perc:.1f}%",
            f"{neu_perc:.1f}%",
            f"{neg_perc:.1f}%",
            gr.update(interactive=True),
        )

    update_stats_btn.click(
        fn=update_stats_button,
        outputs=[ear_output, mar_output, status_output,
                 pos_emotion_output, neu_emotion_output, neg_emotion_output,
                 next_to_voice_btn],
    )

    # Proceed to Voice Analysis tab
    def go_to_voice():
        #face_analyzer.stop()
        return gr.Tabs(selected=2)

    next_to_voice_btn.click(fn=go_to_voice, outputs=[tabs])

    # Voice analysis
    def voice_analysis_and_store(audio):
        text, sentiment_str, pitch_display = analyze_voice(audio)
        stored_data["voice"] = (text, sentiment_str, pitch_display)
        return text, sentiment_str, pitch_display

    record_btn.click(
        fn=voice_analysis_and_store,
        inputs=[audio_in],
        outputs=[text_output, sentiment_output, pitch_output],
    )

    # Proceed to Suggestions tab
    def proceed_to_suggestions():
        required_keys = ["questionnaire", "face", "emotion", "voice", "subtypes"]
        missing_keys = [key for key in required_keys if key not in stored_data]

        if missing_keys:
            return f"Error: Missing data: {', '.join(missing_keys)}. Please complete all sections.", gr.Tabs(selected=2)

        try:
            # Defensive cleaning
            subtype_counts = stored_data.get("subtypes", {})
            cleaned_subtypes = {str(emotion): {str(k): int(v) for k, v in subs.items()} 
                                 for emotion, subs in subtype_counts.items()}
            
            results = aggregate_results(
                stored_data["questionnaire"],
                stored_data["face"],
                stored_data["emotion"],
                stored_data["voice"],
                cleaned_subtypes,
            )
            
            # Get burnout score from model prediction
            burnout_score = predict_burnout(results)

            if burnout_score==0.0:
                suggestion_text = """
                Your burnout score lies between 0 to 20.
                You're feeling good.
                You're active and enjoying your work. The goal is to build habits to maintain that feeling and prevent stress from building up.
                Maintain healthy work-life habits. This includes taking proper lunch breaks and not checking work emails after hours.
                Practice gratitude journaling. Write down three things you are grateful for daily to build resilience.
                Promote regular physical activity. Even a 20-minute walk can help reduce stress and improve your mood.
                """
            elif burnout_score==1.0:
                suggestion_text = """
                Your burnout score lies between 20 to 40.
                You're feeling a bit tired.
                You're under a little stress, but not completely burned out. Now is the time to build in small, intentional breaks to prevent burnout.
                Use the Pomodoro Technique: Work for 25 minutes, then take a 5-minute break. This prevents stress from building up.
                Try Box Breathing: Inhale (4s), Hold (4s), Exhale (4s), Hold (4s). This calms your nervous system quickly.
                Schedule "disconnect" periods. Set specific times to turn off work notifications.
                """
            elif burnout_score==2.0:
                suggestion_text = """
                Your burnout score lies between 40 to 60.
                You're starting to burn out.
                You're feeling physical and emotional exhaustion. This is a critical stage. Set firm boundaries and actively manage your stress.
                Implement a Digital Detox. Designate tech-free hours in the evening to mentally separate work from personal life.
                Prioritize Physical Activity: Aim for 20-30 minutes of aerobic exercise, 3 times a week, to reduce stress hormones.
                Focus on Sleep Hygiene: Ensure you get 7-9 hours of quality sleep to improve your body's ability to cope with stress.
                """
            elif burnout_score==3.0:
                suggestion_text = """
                Your burnout score lies between 60 to 80.
                Your burnout symptoms are persistent.
                They won't go away, and you're thinking about work constantly. You need a more structured approach and should consider professional help.
                Take Mandatory Time Off. Use your vacation days to fully disconnect and recharge.
                Consider a Mindfulness-Based Stress Reduction (MBSR) course. This is a more intensive program that provides long-term stress management skills.
                Seek Professional Help. Contact your company's Employee Assistance Program (EAP) or a therapist for confidential support.
                """
            else:
                suggestion_text = """
                Your burnout score lies between 80 to 100.
                You feel completely burned out.
                You feel like you can't go on. Your health and safety are the top priority. You need immediate professional help.
                Access Crisis Resources: If you are having thoughts of self-harm, immediately call a crisis hotline like the National Suicide Prevention Lifeline at 988.
                Consult a Doctor Immediately. Seek urgent help from a doctor, psychologist, or psychiatrist.
                Prioritize Health Over Work. Your health and safety are the most important thing. Work can wait.
                """
            
            return suggestion_text, gr.Tabs(selected=3)
        except Exception as e:
            return f"Error: Failed to generate results: {str(e)}", gr.Tabs(selected=2)


    proceed_to_results_btn.click(
        fn=proceed_to_suggestions,
        outputs=[suggestion_output, tabs],
    )
    
if __name__ == "__main__":
    demo.launch(share=True)
