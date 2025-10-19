# app.py (HYBRID-enabled)
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image
import time
import threading
from gtts import gTTS
import tempfile
import os
from collections import deque, Counter
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay

# ---------------------------
# Constants & Config
# ---------------------------
MOVENET_MODELS = {
    "Thunder": "https://tfhub.dev/google/movenet/singlepose/thunder/4",
    "Lightning": "https://tfhub.dev/google/movenet/singlepose/lightning/4"
}

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# ---------------------------
# Utility math functions
# ---------------------------
def calculate_angle(a, b, c) -> float:
    a, b, c = np.array(a[:2], dtype=float), np.array(b[:2], dtype=float), np.array(c[:2], dtype=float)
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 180.0
    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosang))
    return float(angle)

def calculate_distance(a, b) -> float:
    a, b = np.array(a[:2], dtype=float), np.array(b[:2], dtype=float)
    return float(np.linalg.norm(a - b))

def kp_confident(keypoints: np.ndarray, idxs: List[int], thr: float) -> bool:
    return all(keypoints[i, 2] > thr for i in idxs)

# ---------------------------
# Model loading & preprocessing (robust)
# ---------------------------
@st.cache_resource
def load_movenet_model(model_name: str = "Thunder"):
    model_url = MOVENET_MODELS.get(model_name, MOVENET_MODELS["Thunder"])
    try:
        model = hub.load(model_url)
        if hasattr(model, 'signatures') and 'serving_default' in model.signatures:
            return model.signatures['serving_default']
        if callable(model):
            return model
        raise RuntimeError("Loaded model doesn't expose a usable signature.")
    except Exception as e:
        st.error(f"Failed to load MoveNet model '{model_name}': {e}")
        st.info("Ensure internet access or that a valid SavedModel path is provided.")
        raise

def detect_pose_single(movenet, image_rgb, input_size):
    """
    Run MoveNet and return keypoints array shape (17,3) -> (y, x, score)
    Defensive against different output dict shapes.
    """
    image = tf.image.resize_with_pad(tf.expand_dims(image_rgb, axis=0), input_size, input_size)
    image = tf.cast(image, dtype=tf.int32)
    outputs = movenet(input=image)
    keypoints = None
    if isinstance(outputs, dict):
        for k in ('output_0', 'output', 'output_1'):
            if k in outputs:
                keypoints = outputs[k].numpy()
                break
        if keypoints is None:
            # try first value
            keypoints = list(outputs.values())[0].numpy()
    else:
        keypoints = outputs.numpy()

    keypoints = np.squeeze(keypoints)
    # expected (1,1,17,3) -> squeeze -> (17,3) or already (17,3)
    if keypoints.shape == (17, 3):
        return keypoints.astype(float)
    # attempt reshape if size matches
    if keypoints.size == 17*3:
        return keypoints.reshape((17,3)).astype(float)
    raise RuntimeError(f"Unexpected keypoints shape returned by MoveNet: {keypoints.shape}")

# ---------------------------
# Feature extraction
# ---------------------------
def extract_features(keypoints: np.ndarray, min_confidence: float = 0.3) -> Dict[str, float]:
    f = {}
    try:
        if kp_confident(keypoints, [11,13,15], min_confidence):
            f['left_knee_angle'] = calculate_angle(keypoints[11], keypoints[13], keypoints[15])
        if kp_confident(keypoints, [12,14,16], min_confidence):
            f['right_knee_angle'] = calculate_angle(keypoints[12], keypoints[14], keypoints[16])

        if kp_confident(keypoints, [5,7,9], min_confidence):
            f['left_elbow_angle'] = calculate_angle(keypoints[5], keypoints[7], keypoints[9])
        if kp_confident(keypoints, [6,8,10], min_confidence):
            f['right_elbow_angle'] = calculate_angle(keypoints[6], keypoints[8], keypoints[10])

        if kp_confident(keypoints, [11,12], min_confidence):
            f['hip_width'] = calculate_distance(keypoints[11], keypoints[12])
        if kp_confident(keypoints, [13,14], min_confidence):
            f['knee_width'] = calculate_distance(keypoints[13], keypoints[14])
        if kp_confident(keypoints, [15,16], min_confidence):
            f['ankle_width'] = calculate_distance(keypoints[15], keypoints[16])

        if 'knee_width' in f and 'ankle_width' in f:
            if f['ankle_width'] > 1e-6:
                f['knee_ankle_ratio'] = f['knee_width'] / f['ankle_width']
            else:
                f['knee_ankle_ratio'] = np.inf

        if kp_confident(keypoints, [5,11], min_confidence):
            shoulder = keypoints[5][:2]
            hip = keypoints[11][:2]
            vertical_point = np.array([hip[0] + 1.0, hip[1]])
            f['back_angle'] = calculate_angle(shoulder, hip, vertical_point)
    except Exception as e:
        print("extract_features error:", e)
    return f

# ---------------------------
# Exercise recognition & form checks
# ---------------------------
def recognize_exercise(keypoints: np.ndarray, features: Dict[str,float], min_confidence: float = 0.3) -> Tuple[str, float]:
    if np.max(keypoints[:,2]) < min_confidence:
        return "unknown", 0.0

    exercise = "standing"
    confidence = 0.2

    knee_y_asym = abs(keypoints[13,0] - keypoints[14,0])
    ankle_y_asym = abs(keypoints[15,0] - keypoints[16,0])
    if knee_y_asym > 0.15 or ankle_y_asym > 0.12:
        exercise = "lunge"
        confidence = min(0.98, 0.5 + (knee_y_asym + ankle_y_asym) * 2.5)

    left_elbow = features.get('left_elbow_angle', None)
    right_elbow = features.get('right_elbow_angle', None)
    elbows = [e for e in (left_elbow, right_elbow) if e is not None]
    avg_elbow = np.mean(elbows) if elbows else None

    avg_wrist_y = None
    if keypoints[9,2] > min_confidence and keypoints[10,2] > min_confidence:
        avg_wrist_y = (keypoints[9,0] + keypoints[10,0]) / 2
    avg_shoulder_y = None
    if keypoints[5,2] > min_confidence and keypoints[6,2] > min_confidence:
        avg_shoulder_y = (keypoints[5,0] + keypoints[6,0]) / 2

    if avg_elbow is not None and avg_wrist_y is not None and avg_shoulder_y is not None:
        if avg_elbow < 130 and avg_wrist_y > avg_shoulder_y:
            exercise = "pushup"
            confidence = min(0.98, 0.4 + (180 - avg_elbow) / 200 + 0.2)

    left_knee_angle = features.get('left_knee_angle', None)
    right_knee_angle = features.get('right_knee_angle', None)
    if left_knee_angle is not None and right_knee_angle is not None:
        avg_knee = (left_knee_angle + right_knee_angle) / 2.0
        if avg_knee < 135:
            exercise = "squat"
            confidence = min(0.98, 0.45 + (170 - avg_knee) / 120)

    if (left_knee_angle is None and right_knee_angle is None) and (avg_shoulder_y is not None):
        exercise = "standing"
        confidence = 0.85

    return exercise, float(confidence)

def detect_form_errors(exercise: str, keypoints: np.ndarray, features: Dict[str,float]) -> List[Dict]:
    feedback = []
    if exercise == "squat":
        left = features.get('left_knee_angle', 180)
        right = features.get('right_knee_angle', 180)
        avg_knee = (left + right) / 2.0
        if avg_knee > 140:
            feedback.append({'severity':'warning','metric':'knee_angle','message':'Squat too shallow — go deeper (aim ~90°)','value':avg_knee})
        elif 90 <= avg_knee <= 100:
            feedback.append({'severity':'success','metric':'knee_angle','message':f'Good squat depth ({avg_knee:.1f}°)','value':avg_knee})
        elif avg_knee < 70:
            feedback.append({'severity':'warning','metric':'knee_angle','message':'Going too deep — control depth','value':avg_knee})

        if 'knee_ankle_ratio' in features:
            ratio = features['knee_ankle_ratio']
            if ratio < 0.8:
                feedback.append({'severity':'error','metric':'knee_alignment','message':'Knees caving inward — push knees out','value':ratio})
            elif ratio > 1.2 and ratio != np.inf:
                feedback.append({'severity':'warning','metric':'knee_alignment','message':'Knees too wide','value':ratio})
            else:
                feedback.append({'severity':'success','metric':'knee_alignment','message':'Knee alignment OK','value':ratio})

        back_angle = features.get('back_angle', None)
        if back_angle is not None:
            if back_angle < 45:
                feedback.append({'severity':'error','metric':'back_angle','message':'Back too horizontal — keep chest up','value':back_angle})
            elif back_angle > 80:
                feedback.append({'severity':'warning','metric':'back_angle','message':'Back very vertical — slightly lean forward','value':back_angle})

    elif exercise == "pushup":
        left_e = features.get('left_elbow_angle', 180)
        right_e = features.get('right_elbow_angle', 180)
        avg_elbow = (left_e + right_e) / 2.0
        if avg_elbow > 140:
            feedback.append({'severity':'warning','metric':'elbow_angle','message':'Lower chest more — target ~90° elbow','value':avg_elbow})
        elif 70 <= avg_elbow <= 110:
            feedback.append({'severity':'success','metric':'elbow_angle','message':f'Good pushup depth ({avg_elbow:.1f}°)','value':avg_elbow})

        if kp_confident(keypoints, [5,11,15], 0.3):
            shoulder_y = keypoints[5,0]
            hip_y = keypoints[11,0]
            ankle_y = keypoints[15,0]
            hip_deviation = abs((hip_y - shoulder_y) - (ankle_y - hip_y))
            if hip_deviation > 0.15:
                if hip_y > (shoulder_y + ankle_y) / 2:
                    feedback.append({'severity':'error','metric':'body_alignment','message':'Hips sagging — engage core','value':hip_deviation})
                else:
                    feedback.append({'severity':'warning','metric':'body_alignment','message':'Hips too high — lower them','value':hip_deviation})

    elif exercise == "lunge":
        front_knee_idx = 13 if keypoints[13,0] > keypoints[14,0] else 14
        front_ankle_idx = 15 if front_knee_idx == 13 else 16
        if kp_confident(keypoints, [front_knee_idx, front_ankle_idx], 0.3):
            front_knee_angle = features.get('left_knee_angle' if front_knee_idx == 13 else 'right_knee_angle', None)
            if front_knee_angle and front_knee_angle > 120:
                feedback.append({'severity':'warning','metric':'knee_angle','message':'Lower down more — front knee should be ~90°','value':front_knee_angle})
            elif front_knee_angle and 80 <= front_knee_angle <= 100:
                feedback.append({'severity':'success','metric':'knee_angle','message':f'Good lunge depth ({front_knee_angle:.1f}°)','value':front_knee_angle})

            knee_x = keypoints[front_knee_idx,1]
            ankle_x = keypoints[front_ankle_idx,1]
            knee_ankle_distance = abs(knee_x - ankle_x)
            if knee_ankle_distance > 0.08:
                feedback.append({'severity':'error','metric':'knee_position','message':'Knee past toes — shift weight back','value':knee_ankle_distance})
    else:
        feedback.append({'severity':'success','metric':'neutral','message':'No actionable form detected','value':0.0})
    return feedback

# ---------------------------
# Draw utilities
# ---------------------------
def draw_keypoints_and_skeleton_rgb(image_rgb: np.ndarray, keypoints: np.ndarray, confidence_threshold: float = 0.3) -> np.ndarray:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    for edge in EDGES:
        y1,x1,c1 = keypoints[edge[0]]
        y2,x2,c2 = keypoints[edge[1]]
        if c1 > confidence_threshold and c2 > confidence_threshold:
            start = (int(x1*w), int(y1*h))
            end = (int(x2*w), int(y2*h))
            cv2.line(image_bgr, start, end, (0,200,0), 2)
    for i, (y,x,c) in enumerate(keypoints):
        if c > confidence_threshold:
            center = (int(x*w), int(y*h))
            cv2.circle(image_bgr, center, 4, (0,255,0), -1)
            cv2.circle(image_bgr, center, 6, (255,255,255), 1)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# ---------------------------
# Audio system (server-side)
# ---------------------------
class AudioFeedbackSystem:
    def __init__(self, lang='en'):
        self.lang = lang
        self.is_speaking = False
        self.lock = threading.Lock()

    def speak(self, message: str):
        def _speak():
            try:
                with self.lock:
                    self.is_speaking = True
                    tts = gTTS(text=message, lang=self.lang)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                        tmpname = tmp.name
                    tts.save(tmpname)
                    # try to play via Streamlit (may warn if called in background thread)
                    try:
                        st.audio(tmpname, format="audio/mp3", autoplay=True)
                    except Exception as e:
                        print("st.audio in bg thread may fail:", e)
                    finally:
                        try:
                            os.remove(tmpname)
                        except Exception:
                            pass
            except Exception as e:
                st.error(f"TTS error: {e}")
            finally:
                self.is_speaking = False
        t = threading.Thread(target=_speak, daemon=True)
        t.start()

# Optional offline pyttsx3 implementation (uncomment to use, requires pyttsx3 installed)
import pyttsx3
class AudioFeedbackSystemOffline:
    def __init__(self, lang='en'):
        self.engine = pyttsx3.init()
        self.lock = threading.Lock()
    def speak(self, message):
        def _s():
            with self.lock:
                self.engine.say(message)
                self.engine.runAndWait()
        threading.Thread(target=_s, daemon=True).start()


@st.cache_resource
def get_audio_system():
    return AudioFeedbackSystem()

# ---------------------------
# Temporal smoothing classes
# ---------------------------
class TemporalSmoother:
    def __init__(self, window_size=15, confidence_threshold=0.6, min_change_interval=1.0):
        self.window_size = window_size
        self.exercise_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        self.last_stable = "standing"
        self.last_change_time = time.time()
        self.confidence_threshold = confidence_threshold
        self.min_change_interval = min_change_interval

    def update(self, exercise: str, confidence: float) -> Tuple[str, float]:
        self.exercise_history.append(exercise)
        self.confidence_history.append(confidence)
        if len(self.exercise_history) < max(3, self.window_size//3):
            return self.last_stable, float(np.mean(self.confidence_history)) if len(self.confidence_history) else 0.0
        vote = Counter(self.exercise_history).most_common(1)[0]
        most_common, count = vote
        avg_conf = np.mean([c for e, c in zip(self.exercise_history, self.confidence_history) if e == most_common]) if any(e==most_common for e in self.exercise_history) else 0.0
        majority_threshold = max(1, int(self.window_size * 0.5))
        now = time.time()
        if count >= majority_threshold and avg_conf >= self.confidence_threshold and (now - self.last_change_time) >= self.min_change_interval:
            if most_common != self.last_stable:
                self.last_stable = most_common
                self.last_change_time = now
        return self.last_stable, float(avg_conf)

class FeedbackSmoother:
    def __init__(self, window_size=10, persistence_ratio=0.5, audio_cooldown=5.0):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.persistence_ratio = persistence_ratio
        self.last_spoken = {}
        self.audio_cooldown = audio_cooldown

    def update(self, feedback_list: List[Dict]) -> List[Dict]:
        msgs = [(fb['metric'], fb['severity'], fb['message']) for fb in feedback_list]
        self.history.append(msgs)
        if len(self.history) < max(3, self.window_size//3):
            return []
        counts = Counter()
        example_map = {}
        for msgs in self.history:
            for key in msgs:
                counts[key] += 1
                example_map[key] = key
        threshold = int(len(self.history) * self.persistence_ratio)
        stable = []
        for key, cnt in counts.items():
            if cnt >= threshold:
                metric, severity, message = key
                for fb in reversed(feedback_list):
                    if fb['metric'] == metric and fb['severity'] == severity and fb['message'] == message:
                        stable.append(fb)
                        break
        return stable

    def should_speak(self, metric_key: str) -> bool:
        now = time.time()
        last = self.last_spoken.get(metric_key, 0.0)
        if now - last >= self.audio_cooldown:
            self.last_spoken[metric_key] = now
            return True
        return False

# ---------------------------
# Streamlit UI & main app
# ---------------------------
st.set_page_config(page_title="PosePerfect - Form Correction", layout="wide", initial_sidebar_state="expanded")

# Header CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .exercise-label { font-size: 1.6rem; font-weight: 700; color:#2b6cb0; text-align:center }
    .feedback-error { background: #fee; border-left: 4px solid #f44; padding:0.6rem; border-radius:6px; margin:0.4rem 0;}
    .feedback-warning { background:#fff4e5; border-left:4px solid #f59e0b; padding:0.6rem; border-radius:6px; margin:0.4rem 0;}
    .feedback-success { background:#ecfdf5; border-left:4px solid #10b981; padding:0.6rem; border-radius:6px; margin:0.4rem 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🏋️ PosePerfect — Real-time Form Evaluation</h1><p>MSU-IIT × Kyushu Sangyo University — COIL 2025</p></div>', unsafe_allow_html=True)

# Sidebar configuration (including hybrid camera options)
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("MoveNet Model", options=["Thunder", "Lightning"], index=0)
input_size = 256 if model_choice == "Thunder" else 192
st.sidebar.markdown(f"**Model:** MoveNet {model_choice} ({input_size}x{input_size})")

min_kp_conf = st.sidebar.slider("Keypoint Confidence Threshold", 0.1, 0.9, 0.3, 0.05)
smoothing_window = st.sidebar.slider("Temporal Smoothing Window", 5, 30, 15)
smoothing_conf_thresh = st.sidebar.slider("Smoothing Confidence Threshold", 0.3, 0.95, 0.6, 0.05)
min_time_between_changes = st.sidebar.slider("Min time between exercise changes (s)", 0.5, 3.0, 1.0, 0.1)

# Hybrid camera options
st.sidebar.markdown("---")
st.sidebar.header("Camera / Capture")
use_local_cam = st.sidebar.checkbox("Use local webcam (for development)", value=False)
auto_refresh = st.sidebar.checkbox("Auto-refresh browser camera (pseudo-live)", value=True)
refresh_interval = st.sidebar.slider("Refresh interval (s) — browser camera", 1, 8, 3)

enable_audio = st.sidebar.checkbox("Enable Server TTS (gTTS)", value=False)
audio_cooldown = st.sidebar.slider("Audio Cooldown (s)", 2.0, 10.0, 5.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.subheader("Recording & Metrics")
enable_labeling = st.sidebar.checkbox("Enable live labeling (press button to set true label)", value=True)
auto_log_pred = st.sidebar.checkbox("Auto-log predictions for metrics", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Instructions")
st.sidebar.markdown("""
**Camera placement**
- Place camera ~2–4 meters away, chest height (or slightly lower).
- Side view is best for squats and lunges; slightly angled top-down helps pushups.
- Ensure full body is visible (head to ankles).
""")

st.sidebar.markdown("⚠️ Note: Server TTS plays on the machine running the app. If hosting remotely, enable only if you want audio on that host.")

# Load model
with st.spinner("Loading MoveNet model..."):
    movenet = load_movenet_model(model_choice)

audio_system = get_audio_system() if enable_audio else None

# Main tabs
tab_live, tab_image, tab_metrics = st.tabs(["📹 Live Detection", "📸 Image Analysis", "📊 Metrics & Evaluation"])

# Session state defaults
if 'running' not in st.session_state:
    st.session_state.running = False
if 'y_true' not in st.session_state:
    st.session_state.y_true = []
if 'y_pred' not in st.session_state:
    st.session_state.y_pred = []
if 'label_current' not in st.session_state:
    st.session_state.label_current = "unknown"
if 'rep_count' not in st.session_state:
    st.session_state.rep_count = 0
if 'smoother' not in st.session_state:
    st.session_state.smoother = TemporalSmoother(window_size=smoothing_window, confidence_threshold=smoothing_conf_thresh, min_change_interval=min_time_between_changes)
if 'fb_smoother' not in st.session_state:
    st.session_state.fb_smoother = FeedbackSmoother(window_size=10, persistence_ratio=0.5, audio_cooldown=audio_cooldown)
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = 0.0

# update smoother params
st.session_state.smoother.window_size = smoothing_window
st.session_state.smoother.confidence_threshold = smoothing_conf_thresh
st.session_state.smoother.min_change_interval = min_time_between_changes
st.session_state.fb_smoother.audio_cooldown = audio_cooldown

# ---------------------------
# Live Detection Tab (hybrid)
# ---------------------------
with tab_live:
    st.subheader("Real-time Form Detection (Hybrid Mode)")
    col1, col2 = st.columns([3,1])
    with col1:
        video_placeholder = st.empty()
    with col2:
        st.markdown("### Status")
        status_text = st.empty()
        st.markdown("### Detected Exercise")
        exercise_label_box = st.empty()
        st.markdown("### Confidence")
        confidence_metric = st.empty()
        st.markdown("### FPS")
        fps_metric = st.empty()
        st.markdown("### Live Feedback")
        feedback_box = st.empty()
        st.markdown("### Live Ground Truth Label (for metrics)")
        label_button_cols = st.columns(2)
        with label_button_cols[0]:
            if st.button("Set label: squat"):
                st.session_state.label_current = "squat"
            if st.button("Set label: pushup"):
                st.session_state.label_current = "pushup"
        with label_button_cols[1]:
            if st.button("Set label: lunge"):
                st.session_state.label_current = "lunge"
            if st.button("Set label: standing"):
                st.session_state.label_current = "standing"
        st.markdown("**Current label:**")
        st.info(st.session_state.label_current)

    start_col, stop_col, clear_col = st.columns([1,1,1])
    with start_col:
        if st.button("▶️ Start", key="start_live"):
            st.session_state.running = True
    with stop_col:
        if st.button("⏹️ Stop", key="stop_live"):
            st.session_state.running = False
    with clear_col:
        if st.button("🧹 Clear logged metrics"):
            st.session_state.y_true = []
            st.session_state.y_pred = []
            st.session_state.rep_count = 0
            st.success("Cleared logged labels & predictions")

    # If using local webcam -> run near-real-time OpenCV loop
    if use_local_cam:
        if st.session_state.running:
            status_text.info("Starting local webcam...")
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            time.sleep(0.3)
            if not cap.isOpened():
                status_text.error("Could not open local camera. Check permissions and device.")
                st.session_state.running = False
            else:
                status_text.success("Local camera active")
                prev_time = time.time()
                fps_deque = deque(maxlen=30)
                try:
                    while st.session_state.running:
                        ret, frame_bgr = cap.read()
                        if not ret:
                            status_text.error("Frame read failed. Stopping.")
                            break
                        frame_bgr = cv2.flip(frame_bgr, 1)
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                        # Pose detection and pipeline
                        keypoints = detect_pose_single(movenet, frame_rgb, input_size)
                        features = extract_features(keypoints, min_kp_conf)
                        raw_ex, raw_conf = recognize_exercise(keypoints, features, min_kp_conf)
                        stable_ex, stable_conf = st.session_state.smoother.update(raw_ex, raw_conf)
                        raw_feedback = []
                        if stable_conf > 0.5:
                            raw_feedback = detect_form_errors(stable_ex, keypoints, features)
                        stable_feedback = st.session_state.fb_smoother.update(raw_feedback)

                        if enable_audio and audio_system and stable_feedback:
                            for fb in stable_feedback:
                                if fb['severity'] == 'error' and st.session_state.fb_smoother.should_speak(fb['metric']):
                                    audio_system.speak(fb['message'])

                        annotated = draw_keypoints_and_skeleton_rgb(frame_rgb, keypoints, confidence_threshold=min_kp_conf)
                        overlay = annotated.copy()
                        if stable_conf > 0:
                            cv2.putText(overlay, f"{stable_ex.upper()}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,0), 3, cv2.LINE_AA)
                            cv2.putText(overlay, f"Conf: {stable_conf:.2f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2, cv2.LINE_AA)

                        now = time.time()
                        dt = now - prev_time if (now - prev_time) > 1e-6 else 1e-6
                        fps = 1.0 / dt
                        prev_time = now
                        fps_deque.append(fps)
                        avg_fps = np.mean(fps_deque)

                        video_placeholder.image(overlay, channels="RGB", use_container_width=True)
                        exercise_label_box.markdown(f'<div class="exercise-label">{stable_ex.upper()}</div>', unsafe_allow_html=True)
                        confidence_metric.metric("Confidence", f"{stable_conf:.1%}")
                        fps_metric.metric("FPS", f"{avg_fps:.1f}")

                        if stable_feedback:
                            html = ""
                            for fb in stable_feedback:
                                cls = "feedback-error" if fb['severity']=="error" else ("feedback-warning" if fb['severity']=="warning" else "feedback-success")
                                icon = "❌" if fb['severity']=="error" else ("⚠️" if fb['severity']=="warning" else "✅")
                                html += f'<div class="{cls}">{icon} {fb["message"]}</div>'
                            feedback_box.markdown(html, unsafe_allow_html=True)
                        else:
                            if stable_conf > 0.5:
                                feedback_box.markdown('<div class="feedback-success">✅ Form looks OK</div>', unsafe_allow_html=True)
                            else:
                                feedback_box.info("Perform an exercise for feedback")

                        if auto_log_pred and stable_conf > 0.4:
                            st.session_state.y_pred.append(stable_ex)
                            st.session_state.y_true.append(st.session_state.label_current)

                        time.sleep(0.02)
                finally:
                    cap.release()
                    cv2.destroyAllWindows()
                    status_text.info("Local camera stopped")
        else:
            st.info("Click Start to begin local webcam detection")

    # Browser camera mode (st.camera_input) with auto-refresh for pseudo-live
    else:
        if st.session_state.running:
            status_text.info("Waiting for browser camera frame...")
            frame = st.camera_input("📸 Capture live (browser camera) — allow camera permission")

            if frame is None:
                st.info("Please enable camera in your browser and grant permission.")
                # do not stop; allow user to enable camera
            else:
                # Convert
                img_bytes = np.asarray(bytearray(frame.read()), dtype=np.uint8)
                image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
                if image is None:
                    status_text.error("Failed to decode camera image.")
                else:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    try:
                        keypoints = detect_pose_single(movenet, image_rgb, input_size)
                        features = extract_features(keypoints, min_kp_conf)
                        raw_ex, raw_conf = recognize_exercise(keypoints, features, min_kp_conf)
                        stable_ex, stable_conf = st.session_state.smoother.update(raw_ex, raw_conf)
                        raw_feedback = []
                        if stable_conf > 0.5:
                            raw_feedback = detect_form_errors(stable_ex, keypoints, features)
                        stable_feedback = st.session_state.fb_smoother.update(raw_feedback)

                        if enable_audio and audio_system and stable_feedback:
                            for fb in stable_feedback:
                                if fb['severity'] == 'error' and st.session_state.fb_smoother.should_speak(fb['metric']):
                                    audio_system.speak(fb['message'])

                        annotated = draw_keypoints_and_skeleton_rgb(image_rgb, keypoints, confidence_threshold=min_kp_conf)
                        overlay = annotated.copy()
                        if stable_conf > 0:
                            cv2.putText(overlay, f"{stable_ex.upper()}", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,0), 3, cv2.LINE_AA)
                            cv2.putText(overlay, f"Conf: {stable_conf:.2f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2, cv2.LINE_AA)

                        video_placeholder.image(overlay, channels="RGB", use_container_width=True)
                        exercise_label_box.markdown(f'<div class="exercise-label">{stable_ex.upper()}</div>', unsafe_allow_html=True)
                        confidence_metric.metric("Confidence", f"{stable_conf:.1%}")
                        fps_metric.metric("FPS", "N/A")

                        if stable_feedback:
                            html = ""
                            for fb in stable_feedback:
                                cls = "feedback-error" if fb['severity']=="error" else ("feedback-warning" if fb['severity']=="warning" else "feedback-success")
                                icon = "❌" if fb['severity']=="error" else ("⚠️" if fb['severity']=="warning" else "✅")
                                html += f'<div class="{cls}">{icon} {fb["message"]}</div>'
                            feedback_box.markdown(html, unsafe_allow_html=True)
                        else:
                            if stable_conf > 0.5:
                                feedback_box.markdown('<div class="feedback-success">✅ Form looks OK</div>', unsafe_allow_html=True)
                            else:
                                feedback_box.info("Perform an exercise for feedback")

                        if auto_log_pred and stable_conf > 0.4:
                            st.session_state.y_pred.append(stable_ex)
                            st.session_state.y_true.append(st.session_state.label_current)

                    except Exception as e:
                        status_text.error(f"Inference error: {e}")

            # Auto-refresh behavior
            if auto_refresh and st.session_state.running:
                # wait small interval then rerun to get new camera frame
                time.sleep(refresh_interval)
                st.rerun()
        else:
            st.info("Click Start to begin browser-camera detection")

# ---------------------------
# Image Analysis Tab
# ---------------------------
with tab_image:
    st.subheader("Static Image Analysis")
    colA, colB = st.columns([1,1])
    with colA:
        uploaded = st.file_uploader("Upload an image (jpg/png)", type=['jpg','jpeg','png'])
        model_sel = st.selectbox("Model for inference (image)", ["Thunder","Lightning"], index=0)
        if uploaded:
            pil = Image.open(uploaded).convert("RGB")
            img_np = np.array(pil)
            input_s = 256 if model_sel == "Thunder" else 192
            with st.spinner("Detecting pose & analyzing..."):
                kps = detect_pose_single(load_movenet_model(model_sel), img_np, input_s)
                feats = extract_features(kps, min_kp_conf)
                ex, conf = recognize_exercise(kps, feats, min_kp_conf)
                feedbacks = detect_form_errors(ex, kps, feats)
                annotated = draw_keypoints_and_skeleton_rgb(img_np, kps, min_kp_conf)
            st.image(annotated, caption=f"Detected: {ex} (conf {conf:.2f})", use_container_width=True)
    with colB:
        if uploaded:
            st.markdown("### Features")
            st.json(feats)
            st.markdown("### Feedback")
            if feedbacks:
                for fb in feedbacks:
                    if fb['severity'] == 'error':
                        st.markdown(f"<div class='feedback-error'>❌ {fb['message']}</div>", unsafe_allow_html=True)
                    elif fb['severity']=='warning':
                        st.markdown(f"<div class='feedback-warning'>⚠️ {fb['message']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='feedback-success'>✅ {fb['message']}</div>", unsafe_allow_html=True)
            else:
                st.success("No major issues detected")
            if st.checkbox("Add this analysis to metrics (label & prediction)"):
                true_lbl = st.selectbox("Select ground-truth label for this image", ["squat","pushup","lunge","standing","unknown"], key="img_true_label")
                if st.button("Add to metrics dataset"):
                    st.session_state.y_true.append(true_lbl)
                    st.session_state.y_pred.append(ex)
                    st.success("Added to metrics")

# ---------------------------
# Metrics & Evaluation Tab
# ---------------------------
with tab_metrics:
    st.subheader("Evaluation & Metrics")
    st.markdown("Metrics are computed from logged predictions and ground-truth labels collected during the session.")
    st.write(f"Logged samples: {len(st.session_state.y_true)}")

    if len(st.session_state.y_true) == 0:
        st.info("No recorded labels yet. Enable live labeling or add images to gather labeled data.")
    else:
        labels = st.multiselect("Select labels to include in metrics", options=sorted(list(set(st.session_state.y_true + st.session_state.y_pred))), default=sorted(list(set(st.session_state.y_true + st.session_state.y_pred))))
        if labels:
            y_true_filtered, y_pred_filtered = [], []
            for yt, yp in zip(st.session_state.y_true, st.session_state.y_pred):
                if yt in labels or yp in labels:
                    y_true_filtered.append(yt)
                    y_pred_filtered.append(yp)
            if len(y_true_filtered) == 0:
                st.warning("No samples after filtering. Try selecting more labels.")
            else:
                unique_labels = sorted(list(set(y_true_filtered + y_pred_filtered)))
                cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=unique_labels)
                fig, ax = plt.subplots(figsize=(6,5))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
                disp.plot(ax=ax, xticks_rotation=45)
                st.pyplot(fig)

                report = classification_report(y_true_filtered, y_pred_filtered, labels=unique_labels, zero_division=0, output_dict=True)
                st.markdown("### Classification Report")
                st.dataframe(report)

                overall_acc = accuracy_score(y_true_filtered, y_pred_filtered)
                micro_f1 = f1_score(y_true_filtered, y_pred_filtered, average='micro', zero_division=0)
                macro_f1 = f1_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
                st.metric("Accuracy", f"{overall_acc:.3f}")
                st.metric("Micro F1", f"{micro_f1:.3f}")
                st.metric("Macro F1", f"{macro_f1:.3f}")

    if len(st.session_state.y_true) > 0:
        if st.button("Download logged labels as CSV"):
            import pandas as pd
            df = pd.DataFrame({"y_true": st.session_state.y_true, "y_pred": st.session_state.y_pred})
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV", data=csv, file_name="poseperfect_logged_labels.csv", mime="text/csv")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:#666;'>PosePerfect • Educational tool — not a substitute for professional coaching. For best results follow camera & lighting tips.</div>", unsafe_allow_html=True)
