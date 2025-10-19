# app.py - PosePerfect (Production Ready)
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image
import time
import tempfile
import os
from collections import deque, Counter
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from gtts import gTTS
from dataclasses import dataclass
from threading import Lock
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Configuration
# ---------------------------
@dataclass
class AppConfig:
    """Application configuration"""
    model_name: str = "Lightning"
    min_confidence: float = 0.3
    smoothing_window: int = 15
    smoothing_conf_threshold: float = 0.6
    min_time_between_changes: float = 1.0
    audio_enabled: bool = True
    audio_cooldown: float = 5.0
    auto_refresh: bool = True
    refresh_interval: float = 3.0
    auto_log_predictions: bool = True
    use_local_camera: bool = False
    frame_skip: int = 1
    
    @property
    def input_size(self) -> int:
        """Get input size based on model name"""
        return 192 if self.model_name == "Lightning" else 256

# ---------------------------
# Constants
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

EXERCISE_LABELS = ["squat", "pushup", "lunge", "standing", "unknown"]

# ---------------------------
# Audio Feedback Manager
# ---------------------------
class AudioFeedbackManager:
    """Manages audio feedback with queueing and rate limiting"""
    
    def __init__(self, cooldown_seconds: float = 5.0, max_queue_size: int = 3):
        self.cooldown = cooldown_seconds
        self.last_spoken = {}
        self.audio_queue = deque(maxlen=max_queue_size)
        self.lock = Lock()
        self.last_play_time = 0
        self.min_gap_between_audio = 2.0
        self.enabled = True
    
    def should_speak(self, metric_key: str) -> bool:
        """Check if enough time has passed for this metric"""
        now = time.time()
        last = self.last_spoken.get(metric_key, 0.0)
        return (now - last) >= self.cooldown
    
    def can_play_now(self) -> bool:
        """Check if enough time has passed since last audio"""
        now = time.time()
        return (now - self.last_play_time) >= self.min_gap_between_audio
    
    def add_to_queue(self, metric_key: str, message: str, priority: str = 'warning') -> bool:
        """Add audio message to queue with priority"""
        if not self.enabled:
            return False
            
        with self.lock:
            if not self.should_speak(metric_key):
                return False
            
            priority_map = {'error': 3, 'warning': 2, 'success': 1}
            self.audio_queue.append({
                'metric': metric_key,
                'message': message,
                'priority': priority_map.get(priority, 1),
                'timestamp': time.time()
            })
            return True
    
    def play_next(self, placeholder=None) -> Optional[str]:
        """Play the highest priority message from queue"""
        if not self.enabled:
            return None
            
        with self.lock:
            if not self.audio_queue or not self.can_play_now():
                return None
            
            sorted_queue = sorted(self.audio_queue, 
                                key=lambda x: (-x['priority'], x['timestamp']))
            
            if sorted_queue:
                audio_item = sorted_queue[0]
                self.audio_queue.remove(audio_item)
                
                success = self._speak_text(audio_item['message'], placeholder)
                
                if success:
                    self.last_spoken[audio_item['metric']] = time.time()
                    self.last_play_time = time.time()
                    return audio_item['message']
        
        return None
    
    def _speak_text(self, message: str, placeholder=None) -> bool:
        """Generate and play audio using gTTS"""
        tmpname = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmpname = tmp.name
            
            tts = gTTS(text=message, lang="en", slow=False)
            tts.save(tmpname)
            
            with open(tmpname, "rb") as f:
                audio_bytes = f.read()
            
            if placeholder:
                with placeholder:
                    st.audio(audio_bytes, format="audio/mp3")
            else:
                st.audio(audio_bytes, format="audio/mp3")
            
            return True
            
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")
            if placeholder:
                with placeholder:
                    st.warning(f"Audio error: {str(e)[:50]}")
            return False
        finally:
            if tmpname and os.path.exists(tmpname):
                try:
                    os.remove(tmpname)
                except Exception:
                    pass
    
    def clear_queue(self):
        """Clear all pending audio messages"""
        with self.lock:
            self.audio_queue.clear()

# ---------------------------
# Camera Refresh Manager
# ---------------------------
class CameraRefreshManager:
    """Manages browser camera refresh without infinite loops"""
    
    def __init__(self, refresh_interval: float = 3.0, max_consecutive_fails: int = 5):
        self.refresh_interval = refresh_interval
        self.last_refresh = time.time()
        self.last_frame_time = time.time()
        self.consecutive_fails = 0
        self.max_consecutive_fails = max_consecutive_fails
        self.is_stale = False
    
    def should_refresh(self) -> bool:
        """Check if it's time to refresh the camera"""
        current_time = time.time()
        return (current_time - self.last_refresh) >= self.refresh_interval
    
    def mark_refreshed(self):
        """Mark that a refresh has occurred"""
        self.last_refresh = time.time()
        self.consecutive_fails = 0
    
    def mark_frame_received(self):
        """Mark that a frame was successfully received"""
        self.last_frame_time = time.time()
        self.consecutive_fails = 0
        self.is_stale = False
    
    def mark_frame_failed(self):
        """Mark that frame retrieval failed"""
        self.consecutive_fails += 1
        if self.consecutive_fails >= self.max_consecutive_fails:
            self.is_stale = True
    
    def get_status(self) -> Dict:
        """Get current status information"""
        current_time = time.time()
        return {
            'time_since_refresh': current_time - self.last_refresh,
            'time_since_frame': current_time - self.last_frame_time,
            'consecutive_fails': self.consecutive_fails,
            'is_stale': self.is_stale,
            'next_refresh_in': max(0, self.refresh_interval - (current_time - self.last_refresh))
        }

# ---------------------------
# Performance Monitor
# ---------------------------
class PerformanceMonitor:
    """Monitor app performance metrics"""
    
    def __init__(self, window_size: int = 30):
        self.fps_history = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        self.last_time = time.time()
    
    def record_frame(self):
        """Record a frame and calculate FPS"""
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 1e-6:
            fps = 1.0 / dt
            self.fps_history.append(fps)
        self.last_time = current_time
    
    def record_inference(self, duration: float):
        """Record inference time"""
        self.inference_times.append(duration)
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        return {
            'avg_fps': float(np.mean(self.fps_history)) if self.fps_history else 0.0,
            'avg_inference_ms': float(np.mean(self.inference_times) * 1000) if self.inference_times else 0.0,
            'frame_count': len(self.fps_history)
        }

# ---------------------------
# Utility Functions
# ---------------------------
def calculate_angle(a, b, c) -> float:
    """Calculate angle between three points"""
    try:
        a, b, c = np.array(a[:2], dtype=float), np.array(b[:2], dtype=float), np.array(c[:2], dtype=float)
        v1, v2 = a - b, c - b
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 180.0
        cosang = np.dot(v1, v2) / (n1 * n2)
        cosang = np.clip(cosang, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosang))
        return float(angle)
    except Exception as e:
        logger.warning(f"Angle calculation error: {e}")
        return 180.0

def calculate_distance(a, b) -> float:
    """Calculate distance between two points"""
    try:
        a, b = np.array(a[:2], dtype=float), np.array(b[:2], dtype=float)
        return float(np.linalg.norm(a - b))
    except Exception as e:
        logger.warning(f"Distance calculation error: {e}")
        return 0.0

def kp_confident(keypoints: np.ndarray, idxs: List[int], thr: float) -> bool:
    """Check if keypoints meet confidence threshold"""
    try:
        return all(keypoints[i, 2] > thr for i in idxs)
    except Exception:
        return False

# ---------------------------
# Model Loading
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_movenet_model(model_name: str = "Thunder"):
    """Load MoveNet model with error handling"""
    model_url = MOVENET_MODELS.get(model_name, MOVENET_MODELS["Thunder"])
    try:
        logger.info(f"Loading MoveNet {model_name} model...")
        model = hub.load(model_url)
        if hasattr(model, 'signatures') and 'serving_default' in model.signatures:
            logger.info("Model loaded successfully")
            return model.signatures['serving_default']
        if callable(model):
            logger.info("Model loaded successfully")
            return model
        raise RuntimeError("Loaded model doesn't expose a usable signature.")
    except Exception as e:
        logger.error(f"Failed to load MoveNet model '{model_name}': {e}")
        st.error(f"❌ Failed to load model: {e}")
        st.info("Ensure internet access for model download.")
        raise

def detect_pose_single(movenet, image_rgb, input_size: int):
    """Run MoveNet and return keypoints"""
    try:
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
                keypoints = list(outputs.values())[0].numpy()
        else:
            keypoints = outputs.numpy()

        keypoints = np.squeeze(keypoints)
        if keypoints.shape == (17, 3):
            return keypoints.astype(float)
        if keypoints.size == 17*3:
            return keypoints.reshape((17,3)).astype(float)
        raise RuntimeError(f"Unexpected keypoints shape: {keypoints.shape}")
    except Exception as e:
        logger.error(f"Pose detection error: {e}")
        return np.zeros((17, 3), dtype=float)

# ---------------------------
# Feature Extraction
# ---------------------------
def extract_features(keypoints: np.ndarray, min_confidence: float = 0.3) -> Dict[str, float]:
    """Extract features from keypoints"""
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
        logger.warning(f"Feature extraction error: {e}")
    return f

# ---------------------------
# Exercise Recognition
# ---------------------------
def recognize_exercise(keypoints: np.ndarray, features: Dict[str,float], min_confidence: float = 0.3) -> Tuple[str, float]:
    """Recognize exercise from keypoints and features"""
    try:
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
    except Exception as e:
        logger.warning(f"Exercise recognition error: {e}")
        return "unknown", 0.0

# ---------------------------
# Form Error Detection
# ---------------------------
def detect_form_errors(exercise: str, keypoints: np.ndarray, features: Dict[str,float]) -> List[Dict]:
    """Detect form errors for specific exercises"""
    feedback = []
    try:
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
    except Exception as e:
        logger.warning(f"Form error detection error: {e}")
    
    return feedback

# ---------------------------
# Drawing Functions
# ---------------------------
def draw_keypoints_and_skeleton_rgb(image_rgb: np.ndarray, keypoints: np.ndarray, confidence_threshold: float = 0.3) -> np.ndarray:
    """Draw keypoints and skeleton on RGB image"""
    try:
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
    except Exception as e:
        logger.error(f"Drawing error: {e}")
        return image_rgb

# ---------------------------
# Temporal Smoothing
# ---------------------------
class TemporalSmoother:
    """Temporal smoothing for stable predictions"""
    
    def __init__(self, window_size: int = 15, confidence_threshold: float = 0.6, min_change_interval: float = 1.0):
        self.window_size = window_size
        self.exercise_history = deque(maxlen=window_size)
        self.confidence_history = deque(maxlen=window_size)
        self.last_stable = "standing"
        self.last_change_time = time.time()
        self.confidence_threshold = confidence_threshold
        self.min_change_interval = min_change_interval

    def update(self, exercise: str, confidence: float) -> Tuple[str, float]:
        """Update with new prediction and return smoothed result"""
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
    """Temporal smoothing for feedback stability"""
    
    def __init__(self, window_size: int = 10, persistence_ratio: float = 0.5):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.persistence_ratio = persistence_ratio

    def update(self, feedback_list: List[Dict]) -> List[Dict]:
        """Update with new feedback and return stable feedback"""
        msgs = [(fb['metric'], fb['severity'], fb['message']) for fb in feedback_list]
        self.history.append(msgs)
        
        if len(self.history) < max(3, self.window_size//3):
            return []
        
        counts = Counter()
        for msgs in self.history:
            for key in msgs:
                counts[key] += 1
        
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

# ---------------------------
# Feedback Rendering
# ---------------------------
def render_feedback_ui(feedback_list: List[Dict], audio_manager: Optional[AudioFeedbackManager] = None, audio_placeholder=None) -> str:
    """Render feedback with HTML and optional audio"""
    if not feedback_list:
        return '<div class="feedback-success">✅ Form looks good!</div>'
    
    html_output = ""
    audio_triggered = False
    
    severity_order = {'error': 0, 'warning': 1, 'success': 2}
    sorted_feedback = sorted(feedback_list, 
                            key=lambda x: severity_order.get(x['severity'], 3))
    
    for fb in sorted_feedback:
        severity = fb['severity']
        message = fb['message']
        metric = fb['metric']
        
        if severity == 'error':
            css_class = 'feedback-error'
            icon = '❌'
        elif severity == 'warning':
            css_class = 'feedback-warning'
            icon = '⚠️'
        else:
            css_class = 'feedback-success'
            icon = '✅'
        
        html_output += f'<div class="{css_class}">{icon} {message}</div>'
        
        if audio_manager and severity == 'error' and not audio_triggered:
            if audio_manager.add_to_queue(metric, message, severity):
                audio_manager.play_next(audio_placeholder)
                audio_triggered = True
    
    return html_output

# ---------------------------
# Session State Initialization
# ---------------------------
def initialize_session_state(config: AppConfig):
    """Initialize all session state variables"""
    if 'audio_manager' not in st.session_state:
        st.session_state.audio_manager = AudioFeedbackManager(
            cooldown_seconds=config.audio_cooldown,
            max_queue_size=3
        )
    
    if 'camera_manager' not in st.session_state:
        st.session_state.camera_manager = CameraRefreshManager(
            refresh_interval=config.refresh_interval,
            max_consecutive_fails=5
        )
    
    if 'perf_monitor' not in st.session_state:
        st.session_state.perf_monitor = PerformanceMonitor(window_size=30)
    
    if 'smoother' not in st.session_state:
        st.session_state.smoother = TemporalSmoother(
            window_size=config.smoothing_window,
            confidence_threshold=config.smoothing_conf_threshold,
            min_change_interval=config.min_time_between_changes
        )
    
    if 'fb_smoother' not in st.session_state:
        st.session_state.fb_smoother = FeedbackSmoother(
            window_size=10,
            persistence_ratio=0.5
        )
    
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
    
    if 'frame_counter' not in st.session_state:
        st.session_state.frame_counter = 0

# ---------------------------
# Main Application
# ---------------------------
def main():
    """Main application entry point"""
    
    # Page configuration
    st.set_page_config(
        page_title="PosePerfect - Production Ready",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "PosePerfect - AI-powered exercise form analysis"
        }
    )
    
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            text-align: center;
            padding: 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .exercise-label {
            font-size: 1.6rem;
            font-weight: 700;
            color: #2b6cb0;
            text-align: center;
            padding: 0.5rem;
            background: #e6f7ff;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        .feedback-error {
            background: #fee;
            border-left: 4px solid #f44;
            padding: 0.8rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            animation: slideIn 0.3s ease;
        }
        .feedback-warning {
            background: #fff4e5;
            border-left: 4px solid #f59e0b;
            padding: 0.8rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            animation: slideIn 0.3s ease;
        }
        .feedback-success {
            background: #ecfdf5;
            border-left: 4px solid #10b981;
            padding: 0.8rem;
            border-radius: 6px;
            margin: 0.5rem 0;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border: 1px solid #e0e0e0;
        }
        .status-badge {
            display: inline-block;
            padding: 0.3rem 0.8rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 600;
            margin: 0.2rem;
        }
        .status-active { background: #d1fae5; color: #065f46; }
        .status-inactive { background: #fee2e2; color: #991b1b; }
        .status-waiting { background: #fef3c7; color: #92400e; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('''
    <div class="main-header">
        <h1>🏋️ PosePerfect</h1>
        <p style="font-size:1.1rem; margin:0;">AI-Powered Exercise Form Analysis</p>
        <p style="font-size:0.9rem; margin:0.5rem 0 0 0; opacity:0.9;">
            Real-time pose detection with MoveNet • Intelligent feedback • Browser audio
        </p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar Configuration
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.markdown("### Model Settings")
    model_choice = st.sidebar.selectbox(
        "MoveNet Model",
        options=["Lightning", "Thunder"],
        index=0,
        help="Lightning: Faster (192x192) | Thunder: More accurate (256x256)"
    )
    
    min_kp_conf = st.sidebar.slider(
        "Keypoint Confidence",
        0.1, 0.9, 0.3, 0.05,
        help="Minimum confidence threshold for keypoint detection"
    )
    
    st.sidebar.markdown("### Temporal Smoothing")
    smoothing_window = st.sidebar.slider(
        "Smoothing Window",
        5, 30, 15,
        help="Number of frames to average for stable predictions"
    )
    smoothing_conf_thresh = st.sidebar.slider(
        "Smoothing Confidence",
        0.3, 0.95, 0.6, 0.05,
        help="Confidence threshold for exercise classification"
    )
    min_time_between_changes = st.sidebar.slider(
        "Min Change Interval (s)",
        0.5, 3.0, 1.0, 0.1,
        help="Minimum time between exercise type changes"
    )
    
    st.sidebar.markdown("### Camera Settings")
    use_local_cam = st.sidebar.checkbox(
        "Use Local Webcam",
        value=False,
        help="Enable for development with local camera"
    )
    
    if not use_local_cam:
        auto_refresh = st.sidebar.checkbox(
            "Auto-refresh Browser Camera",
            value=True,
            help="Automatically refresh camera for pseudo-live detection"
        )
        refresh_interval = st.sidebar.slider(
            "Refresh Interval (s)",
            1.0, 8.0, 3.0, 0.5,
            help="Time between camera refreshes"
        )
    else:
        auto_refresh = False
        refresh_interval = 3.0
        frame_skip = st.sidebar.slider(
            "Frame Skip",
            1, 5, 1,
            help="Process every Nth frame (1 = all frames)"
        )
    
    st.sidebar.markdown("### Audio Feedback")
    enable_audio = st.sidebar.checkbox(
        "Enable Voice Feedback",
        value=True,
        help="Text-to-speech feedback for form errors"
    )
    audio_cooldown = st.sidebar.slider(
        "Audio Cooldown (s)",
        2.0, 10.0, 5.0, 1.0,
        help="Minimum time between audio messages"
    )
    
    st.sidebar.markdown("### Data Logging")
    auto_log_pred = st.sidebar.checkbox(
        "Auto-log Predictions",
        value=True,
        help="Automatically log predictions for metrics"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Tip:** Allow camera permissions in your browser for best results")
    
    # Create configuration
    config = AppConfig(
        model_name=model_choice,
        min_confidence=min_kp_conf,
        smoothing_window=smoothing_window,
        smoothing_conf_threshold=smoothing_conf_thresh,
        min_time_between_changes=min_time_between_changes,
        audio_enabled=enable_audio,
        audio_cooldown=audio_cooldown,
        auto_refresh=auto_refresh,
        refresh_interval=refresh_interval,
        auto_log_predictions=auto_log_pred,
        use_local_camera=use_local_cam,
        frame_skip=frame_skip if use_local_cam else 1
    )
    
    # Initialize session state
    initialize_session_state(config)
    
    # Update managers with new config
    st.session_state.audio_manager.cooldown = config.audio_cooldown
    st.session_state.audio_manager.enabled = config.audio_enabled
    st.session_state.camera_manager.refresh_interval = config.refresh_interval
    st.session_state.smoother.window_size = config.smoothing_window
    st.session_state.smoother.confidence_threshold = config.smoothing_conf_threshold
    st.session_state.smoother.min_change_interval = config.min_time_between_changes
    
    # Load model with error handling
    try:
        with st.spinner(f"🔄 Loading MoveNet {config.model_name} model..."):
            movenet = load_movenet_model(config.model_name)
        st.sidebar.success(f"✅ Model loaded: {config.model_name}")
    except Exception as e:
        st.error(f"❌ Failed to load model. Please refresh the page.")
        st.stop()
        return
    
    # Create tabs
    tab_live, tab_image, tab_metrics, tab_about = st.tabs([
        "📹 Live Detection",
        "📸 Image Analysis",
        "📊 Metrics & Evaluation",
        "ℹ️ About"
    ])
    
    # ---------------------------
    # Live Detection Tab
    # ---------------------------
    with tab_live:
        st.subheader("Real-time Form Detection")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            video_placeholder = st.empty()
        
        with col2:
            st.markdown("### 📊 Status")
            status_text = st.empty()
            
            st.markdown("### 🏃 Exercise")
            exercise_label_box = st.empty()
            
            col_conf, col_fps = st.columns(2)
            with col_conf:
                confidence_metric = st.empty()
            with col_fps:
                fps_metric = st.empty()
            
            st.markdown("### 💬 Feedback")
            feedback_box = st.empty()
            
            st.markdown("### 🏷️ Ground Truth Label")
            st.caption("For metrics collection")
            
            label_cols = st.columns(2)
            with label_cols[0]:
                if st.button("🏋️ Squat", use_container_width=True):
                    st.session_state.label_current = "squat"
                if st.button("🤸 Pushup", use_container_width=True):
                    st.session_state.label_current = "pushup"
            with label_cols[1]:
                if st.button("🦵 Lunge", use_container_width=True):
                    st.session_state.label_current = "lunge"
                if st.button("🧍 Standing", use_container_width=True):
                    st.session_state.label_current = "standing"
            
            current_label = st.session_state.label_current
            st.markdown(f'<div class="metric-card"><strong>Current:</strong> {current_label.upper()}</div>', unsafe_allow_html=True)
        
        # Control buttons
        st.markdown("---")
        col_start, col_stop, col_clear = st.columns([1, 1, 1])
        
        with col_start:
            if st.button("▶️ Start Detection", type="primary", use_container_width=True):
                st.session_state.running = True
                st.rerun()
        
        with col_stop:
            if st.button("⏹️ Stop Detection", use_container_width=True):
                st.session_state.running = False
                st.rerun()
        
        with col_clear:
            if st.button("🧹 Clear Metrics", use_container_width=True):
                st.session_state.y_true = []
                st.session_state.y_pred = []
                st.session_state.rep_count = 0
                st.success("✅ Metrics cleared!")
                time.sleep(1)
                st.rerun()
        
        # Audio placeholder
        audio_placeholder = st.empty()
        
        # Local webcam mode
        if config.use_local_camera:
            if st.session_state.running:
                run_local_camera(
                    video_placeholder, status_text, exercise_label_box,
                    confidence_metric, fps_metric, feedback_box,
                    audio_placeholder, movenet, config
                )
            else:
                status_text.info("Click 'Start Detection' to begin")
                video_placeholder.info("📹 Ready to start local webcam detection")
        
        # Browser camera mode
        else:
            if st.session_state.running:
                run_browser_camera(
                    video_placeholder, status_text, exercise_label_box,
                    confidence_metric, fps_metric, feedback_box,
                    audio_placeholder, movenet, config
                )
            else:
                status_text.info("Click 'Start Detection' to begin")
                video_placeholder.info("📸 Ready to start browser camera detection")
    
    # ---------------------------
    # Image Analysis Tab
    # ---------------------------
    with tab_image:
        st.subheader("📸 Static Image Analysis")
        
        col_a, col_b = st.columns([1, 1])
        
        with col_a:
            st.markdown("### Upload Image")
            uploaded = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a photo of your exercise form"
            )
            
            if uploaded:
                try:
                    pil_img = Image.open(uploaded).convert("RGB")
                    img_np = np.array(pil_img)
                    
                    st.image(img_np, caption="Uploaded Image", use_container_width=True)
                    
                    if st.button("🔍 Analyze Form", type="primary"):
                        with st.spinner("Analyzing pose and form..."):
                            input_s = config.input_size
                            
                            start_time = time.time()
                            kps = detect_pose_single(movenet, img_np, input_s)
                            inference_time = time.time() - start_time
                            
                            feats = extract_features(kps, config.min_confidence)
                            ex, conf = recognize_exercise(kps, feats, config.min_confidence)
                            feedbacks = detect_form_errors(ex, kps, feats)
                            annotated = draw_keypoints_and_skeleton_rgb(img_np, kps, config.min_confidence)
                        
                        st.session_state.last_image_result = {
                            'annotated': annotated,
                            'exercise': ex,
                            'confidence': conf,
                            'features': feats,
                            'feedback': feedbacks,
                            'inference_time': inference_time
                        }
                        st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error processing image: {e}")
        
        with col_b:
            if 'last_image_result' in st.session_state:
                result = st.session_state.last_image_result
                
                st.markdown("### 🎯 Analysis Results")
                st.image(
                    result['annotated'],
                    caption=f"Detected: {result['exercise'].upper()}",
                    use_container_width=True
                )
                
                st.markdown(f"**Exercise:** {result['exercise'].upper()}")
                st.markdown(f"**Confidence:** {result['confidence']:.1%}")
                st.markdown(f"**Inference Time:** {result['inference_time']*1000:.1f}ms")
                
                st.markdown("### 📏 Features")
                with st.expander("View extracted features"):
                    st.json(result['features'])
                
                st.markdown("### 💬 Feedback")
                if result['feedback']:
                    feedback_html = render_feedback_ui(result['feedback'])
                    st.markdown(feedback_html, unsafe_allow_html=True)
                else:
                    st.success("✅ No major issues detected")
                
                st.markdown("### 📊 Add to Metrics")
                with st.expander("Label and save to metrics"):
                    true_label = st.selectbox(
                        "Ground truth label",
                        EXERCISE_LABELS,
                        key="img_label"
                    )
                    if st.button("➕ Add to Metrics Dataset"):
                        st.session_state.y_true.append(true_label)
                        st.session_state.y_pred.append(result['exercise'])
                        st.success(f"✅ Added: {true_label} (predicted: {result['exercise']})")
    
    # ---------------------------
    # Metrics Tab
    # ---------------------------
    with tab_metrics:
        st.subheader("📊 Performance Metrics & Evaluation")
        
        total_samples = len(st.session_state.y_true)
        st.metric("Total Logged Samples", total_samples)
        
        if total_samples == 0:
            st.info("📝 No data yet. Enable auto-logging or manually add images to collect metrics.")
        else:
            st.markdown("---")
            
            # Filter options
            all_labels = sorted(list(set(st.session_state.y_true + st.session_state.y_pred)))
            selected_labels = st.multiselect(
                "Select labels to include",
                options=all_labels,
                default=all_labels,
                help="Filter which exercise types to include in metrics"
            )
            
            if not selected_labels:
                st.warning("⚠️ Select at least one label")
            else:
                # Filter data
                y_true_filt = []
                y_pred_filt = []
                for yt, yp in zip(st.session_state.y_true, st.session_state.y_pred):
                    if yt in selected_labels or yp in selected_labels:
                        y_true_filt.append(yt)
                        y_pred_filt.append(yp)
                
                if len(y_true_filt) == 0:
                    st.warning("No samples after filtering")
                else:
                    col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                    
                    overall_acc = accuracy_score(y_true_filt, y_pred_filt)
                    micro_f1 = f1_score(y_true_filt, y_pred_filt, average='micro', zero_division=0)
                    macro_f1 = f1_score(y_true_filt, y_pred_filt, average='macro', zero_division=0)
                    
                    with col_metrics1:
                        st.metric("Accuracy", f"{overall_acc:.3f}")
                    with col_metrics2:
                        st.metric("Micro F1", f"{micro_f1:.3f}")
                    with col_metrics3:
                        st.metric("Macro F1", f"{macro_f1:.3f}")
                    
                    st.markdown("### Confusion Matrix")
                    unique_labels = sorted(list(set(y_true_filt + y_pred_filt)))
                    cm = confusion_matrix(y_true_filt, y_pred_filt, labels=unique_labels)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
                    disp.plot(ax=ax, xticks_rotation=45, cmap='Blues')
                    plt.title("Confusion Matrix")
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    st.markdown("### Classification Report")
                    report = classification_report(
                        y_true_filt, y_pred_filt,
                        labels=unique_labels,
                        zero_division=0,
                        output_dict=True
                    )
                    st.dataframe(report, use_container_width=True)
                    
                    # Download button
                    st.markdown("### 📥 Export Data")
                    if st.button("Download Logged Data as CSV"):
                        import pandas as pd
                        df = pd.DataFrame({
                            "y_true": st.session_state.y_true,
                            "y_pred": st.session_state.y_pred
                        })
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "⬇️ Download CSV",
                            data=csv,
                            file_name="poseperfect_metrics.csv",
                            mime="text/csv"
                        )
    
    # ---------------------------
    # About Tab
    # ---------------------------
    with tab_about:
        st.subheader("ℹ️ About PosePerfect")
        
        st.markdown("""
        ### 🎯 Features
        - **Real-time Pose Detection**: Uses Google's MoveNet for accurate pose estimation
        - **Exercise Recognition**: Automatically detects squats, pushups, lunges, and standing poses
        - **Form Analysis**: Provides instant feedback on exercise form and technique
        - **Voice Feedback**: Text-to-speech audio alerts for form corrections
        - **Metrics Tracking**: Log and evaluate performance with detailed analytics
        
        ### 🔧 Technical Details
        - **Model**: MoveNet (Lightning/Thunder variants)
        - **Framework**: TensorFlow, Streamlit
        - **Audio**: gTTS (Google Text-to-Speech)
        - **Analysis**: Real-time joint angle and alignment calculations
        
        ### 📋 Exercise Form Guidelines
        
        #### Squat
        - Knees should reach ~90° angle
        - Keep knees aligned with toes
        - Maintain upright chest position
        
        #### Pushup
        - Lower until elbows are ~90°
        - Keep body in straight line
        - Engage core to prevent sagging
        
        #### Lunge
        - Front knee at ~90° angle
        - Keep knee behind toes
        - Maintain upright torso
        
        ### ⚠️ Important Notes
        - This app is for **educational purposes only**
        - Not a substitute for professional coaching
        - Ensure proper warm-up before exercising
        - Stop if you feel pain or discomfort
        
        ### 🛠️ System Requirements
        - Modern web browser (Chrome, Firefox, Safari, Edge)
        - Webcam access (for live detection)
        - Internet connection (for model loading and TTS)
        
        ### 📞 Support
        For issues or questions, please check the documentation or contact support.
        
        ---
        
        **Version**: 2.0 Production | **Built with**: Streamlit + TensorFlow + MoveNet
        """)

# ---------------------------
# Camera Processing Functions
# ---------------------------
def run_local_camera(video_placeholder, status_text, exercise_label_box,
                    confidence_metric, fps_metric, feedback_box,
                    audio_placeholder, movenet, config: AppConfig):
    """Run local webcam detection loop"""
    
    status_text.info("🔄 Starting local webcam...")
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    time.sleep(0.5)
    
    if not cap.isOpened():
        status_text.error("❌ Could not open webcam. Check permissions.")
        st.session_state.running = False
        return
    
    status_text.success("✅ Webcam active")
    
    audio_mgr = st.session_state.audio_manager
    perf_mon = st.session_state.perf_monitor
    smoother = st.session_state.smoother
    fb_smoother = st.session_state.fb_smoother
    
    frame_count = 0
    
    try:
        while st.session_state.running:
            ret, frame_bgr = cap.read()
            if not ret:
                status_text.error("❌ Frame read failed")
                break
            
            frame_count += 1
            
            # Frame skipping
            if frame_count % config.frame_skip != 0:
                continue
            
            perf_mon.record_frame()
            
            frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            
            # Pose detection
            infer_start = time.time()
            keypoints = detect_pose_single(movenet, frame_rgb, config.input_size)
            infer_time = time.time() - infer_start
            perf_mon.record_inference(infer_time)
            
            # Feature extraction and recognition
            features = extract_features(keypoints, config.min_confidence)
            raw_ex, raw_conf = recognize_exercise(keypoints, features, config.min_confidence)
            stable_ex, stable_conf = smoother.update(raw_ex, raw_conf)
            
            # Form feedback
            raw_feedback = []
            if stable_conf > 0.5:
                raw_feedback = detect_form_errors(stable_ex, keypoints, features)
            stable_feedback = fb_smoother.update(raw_feedback)
            
            # Audio feedback
            if config.audio_enabled and stable_feedback:
                for fb in stable_feedback:
                    if fb['severity'] == 'error':
                        audio_mgr.add_to_queue(fb['metric'], fb['message'], fb['severity'])
                audio_mgr.play_next(audio_placeholder)
            
            # Draw annotations
            annotated = draw_keypoints_and_skeleton_rgb(frame_rgb, keypoints, config.min_confidence)
            overlay = annotated.copy()
            if stable_conf > 0:
                cv2.putText(overlay, f"{stable_ex.upper()}", (10,50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,0), 3, cv2.LINE_AA)
                cv2.putText(overlay, f"Conf: {stable_conf:.2f}", (10,90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2, cv2.LINE_AA)
            
            # Update UI
            video_placeholder.image(overlay, channels="RGB", use_container_width=True)
            exercise_label_box.markdown(
                f'<div class="exercise-label">{stable_ex.upper()}</div>',
                unsafe_allow_html=True
            )
            
            stats = perf_mon.get_stats()
            confidence_metric.metric("Confidence", f"{stable_conf:.1%}")
            fps_metric.metric("FPS", f"{stats['avg_fps']:.1f}")
            
            # Feedback display
            if stable_feedback:
                feedback_html = render_feedback_ui(stable_feedback, audio_mgr, audio_placeholder)
                feedback_box.markdown(feedback_html, unsafe_allow_html=True)
            else:
                if stable_conf > 0.5:
                    feedback_box.markdown(
                        '<div class="feedback-success">✅ Form looks good!</div>',
                        unsafe_allow_html=True
                    )
                else:
                    feedback_box.info("Perform an exercise for feedback")
            
            # Log predictions
            if config.auto_log_predictions and stable_conf > 0.4:
                st.session_state.y_pred.append(stable_ex)
                st.session_state.y_true.append(st.session_state.label_current)
            
            time.sleep(0.01)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        status_text.info("⏸️ Webcam stopped")

def run_browser_camera(video_placeholder, status_text, exercise_label_box,
                       confidence_metric, fps_metric, feedback_box,
                       audio_placeholder, movenet, config: AppConfig):
    """Run browser camera detection with auto-refresh"""
    
    camera_mgr = st.session_state.camera_manager
    audio_mgr = st.session_state.audio_manager
    smoother = st.session_state.smoother
    fb_smoother = st.session_state.fb_smoother
    
    status_text.info("📸 Browser camera active")
    
    # Camera input with unique key for refresh
    camera_key = f"cam_{int(camera_mgr.last_refresh * 1000)}"
    frame = st.camera_input("📸 Show your movement", key=camera_key)
    
    if frame is None:
        status_text.warning("⏳ Waiting for camera permission...")
        camera_mgr.mark_frame_failed()
        
        cam_status = camera_mgr.get_status()
        st.info(f"Next refresh in: {cam_status['next_refresh_in']:.1f}s")
        video_placeholder.info("📷 Please allow camera access in your browser")
    
    else:
        try:
            # Decode image
            img_bytes = np.asarray(bytearray(frame.read()), dtype=np.uint8)
            image = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                camera_mgr.mark_frame_failed()
                status_text.error("❌ Failed to decode frame")
            else:
                camera_mgr.mark_frame_received()
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process frame
                infer_start = time.time()
                keypoints = detect_pose_single(movenet, image_rgb, config.input_size)
                infer_time = time.time() - infer_start
                
                features = extract_features(keypoints, config.min_confidence)
                raw_ex, raw_conf = recognize_exercise(keypoints, features, config.min_confidence)
                stable_ex, stable_conf = smoother.update(raw_ex, raw_conf)
                
                # Form feedback
                raw_feedback = []
                if stable_conf > 0.5:
                    raw_feedback = detect_form_errors(stable_ex, keypoints, features)
                stable_feedback = fb_smoother.update(raw_feedback)
                
                # Audio feedback
                if config.audio_enabled and stable_feedback:
                    for fb in stable_feedback:
                        if fb['severity'] == 'error':
                            audio_mgr.add_to_queue(fb['metric'], fb['message'], fb['severity'])
                    audio_mgr.play_next(audio_placeholder)
                
                # Draw annotations
                annotated = draw_keypoints_and_skeleton_rgb(image_rgb, keypoints, config.min_confidence)
                overlay = annotated.copy()
                if stable_conf > 0:
                    cv2.putText(overlay, f"{stable_ex.upper()}", (10,50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,200,0), 3, cv2.LINE_AA)
                    cv2.putText(overlay, f"Conf: {stable_conf:.2f}", (10,90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2, cv2.LINE_AA)
                
                # Update UI
                video_placeholder.image(overlay, channels="RGB", use_container_width=True)
                exercise_label_box.markdown(
                    f'<div class="exercise-label">{stable_ex.upper()}</div>',
                    unsafe_allow_html=True
                )
                
                confidence_metric.metric("Confidence", f"{stable_conf:.1%}")
                fps_metric.metric("Inference", f"{infer_time*1000:.0f}ms")
                
                # Feedback display
                if stable_feedback:
                    feedback_html = render_feedback_ui(stable_feedback, audio_mgr, audio_placeholder)
                    feedback_box.markdown(feedback_html, unsafe_allow_html=True)
                else:
                    if stable_conf > 0.5:
                        feedback_box.markdown(
                            '<div class="feedback-success">✅ Form looks good!</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        feedback_box.info("Perform an exercise for feedback")
                
                # Log predictions
                if config.auto_log_predictions and stable_conf > 0.4:
                    st.session_state.y_pred.append(stable_ex)
                    st.session_state.y_true.append(st.session_state.label_current)
                
                status_text.success("✅ Processing frame")
        
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            camera_mgr.mark_frame_failed()
            status_text.error(f"❌ Processing error: {str(e)[:50]}")
    
    # Auto-refresh logic
    if config.auto_refresh and st.session_state.running:
        if camera_mgr.should_refresh():
            camera_mgr.mark_refreshed()
            time.sleep(0.5)
            st.rerun()

# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"❌ Application Error: {e}")
        st.info("Please refresh the page to restart the application.")