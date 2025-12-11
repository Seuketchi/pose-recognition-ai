"""
AI-Powered Gym Workout Tracker

Full-featured workout tracking application with real-time exercise classification,
rep counting, session management, and voice coaching.

Uses the pre-trained Gym-Workout-Classifier-SigLIP2 model from HuggingFace.
Model: https://huggingface.co/prithivMLmods/Gym-Workout-Classifier-SigLIP2
DOI: 10.57967/hf/5391

See docs/MODEL_ATTRIBUTION.md for model details and attribution.
"""

import sys
from pathlib import Path

# Add parent directory to path for src imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
from datetime import datetime, timedelta
import json
import os
from collections import deque
import threading
import queue
import logging
import warnings
import time
import atexit
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# Setup structured logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"workout_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AdaptiveThresholdManager:
    """Dynamically adjusts confidence thresholds based on detection stability"""
    
    def __init__(self, initial_threshold=0.4, min_threshold=0.2, max_threshold=0.8):
        self.threshold = initial_threshold
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.detection_history = deque(maxlen=30)
        self.stability_score = 0.0
        
    def update(self, confidence: float, is_stable: bool) -> float:
        """Update threshold based on detection quality"""
        self.detection_history.append((confidence, is_stable))
        
        # Calculate stability score
        if len(self.detection_history) >= 10:
            recent = list(self.detection_history)[-10:]
            stable_count = sum(1 for _, stable in recent if stable)
            avg_confidence = sum(conf for conf, _ in recent) / len(recent)
            
            self.stability_score = (stable_count / len(recent)) * avg_confidence
            
            # Adjust threshold
            if self.stability_score > 0.7:
                # High stability - can be more selective
                self.threshold = min(self.threshold + 0.01, self.max_threshold)
            elif self.stability_score < 0.3:
                # Low stability - be more permissive
                self.threshold = max(self.threshold - 0.01, self.min_threshold)
        
        return self.threshold
    
    def get_threshold(self) -> float:
        return self.threshold


class SmartInsightsEngine:
    """Enhanced insight generator with multiple TTS backends and caching"""
    
    def __init__(self, tts_mode="fast", insight_interval=60, use_ai_generation=False):
        self.tts_mode = tts_mode
        self.insight_interval = insight_interval
        self.use_ai_generation = use_ai_generation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.insight_queue = queue.Queue(maxsize=5)
        self.history_lock = threading.Lock()
        self.running = True
        self.last_insight_time = time.time()
        
        # Cache for generated insights
        self.insight_cache = {}
        
        if self.use_ai_generation:
            self._load_text_model()
        else:
            logger.info("🧠 Using rule-based insights (faster, more reliable)")
            
        self.setup_tts()
    
    def _load_text_model(self):
        """Load AI text generation model with error handling"""
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            model_name = "google/flan-t5-small"
            
            logger.info(f"🧠 Loading text model ({model_name}) on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.text_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.text_model.to(self.device)
            self.text_model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("✅ Text model ready")
        except Exception as e:
            logger.error(f"❌ Failed to load text model: {e}")
            self.use_ai_generation = False
    
    def setup_tts(self):
        """Setup TTS with automatic fallback"""
        self.tts_engine = None
        
        if self.tts_mode == "fast":
            # Try pyttsx3 first (offline, instant)
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 180)
                self.tts_engine.setProperty('volume', 0.9)
                logger.info("✅ pyttsx3 TTS ready (instant, offline)")
                return
            except (ImportError, RuntimeError) as e:
                logger.warning(f"⚠️ pyttsx3 not available: {e}")
            
            # Fallback to gTTS
            try:
                from gtts import gTTS
                self.tts_engine = "gtts"
                logger.info("✅ gTTS ready (fast, requires internet)")
                return
            except ImportError:
                logger.warning("⚠️ gTTS not available")
        
        logger.warning("⚠️ No TTS available. Install: pip install pyttsx3 or gtts")
    
    def generate_text_insight(self, workout_history: List[Dict]) -> str:
        """Generate motivational insight with caching"""
        
        # Create cache key from recent workout data
        cache_key = self._get_cache_key(workout_history)
        if cache_key in self.insight_cache:
            return self.insight_cache[cache_key]
        
        insight = self._generate_rule_based_insight(workout_history)
        
        # Cache the insight
        self.insight_cache[cache_key] = insight
        if len(self.insight_cache) > 100:  # Limit cache size
            self.insight_cache.pop(next(iter(self.insight_cache)))
        
        return insight
    
    def _get_cache_key(self, history: List[Dict]) -> str:
        """Generate cache key from workout data"""
        if not history:
            return "empty"
        
        recent = history[-3:]
        total_reps = sum(s['reps'] for session in recent for s in session.get('sets', []))
        exercises = tuple(sorted(set(s['exercise'] for session in recent for s in session.get('sets', []))))
        return f"{total_reps}_{exercises}"
    
    def _generate_rule_based_insight(self, workout_history: List[Dict]) -> str:
        """Enhanced rule-based insights with more patterns"""
        if not workout_history:
            return self._generate_generic_motivation()
        
        try:
            recent = workout_history[-5:]
            total_reps = sum(s['reps'] for session in recent for s in session.get('sets', []))
            exercises = list(set(s['exercise'] for session in recent for s in session.get('sets', [])))
            total_sets = sum(len(session.get('sets', [])) for session in recent)
            
            insights = []
            
            # Volume-based insights
            if total_reps > 200:
                insights.append(f"Outstanding volume! {total_reps} reps across recent sessions!")
            elif total_reps > 100:
                insights.append(f"Solid effort! {total_reps} reps completed!")
            elif total_reps > 50:
                insights.append(f"Great consistency! {total_reps} reps and counting!")
            
            # Progression tracking
            if len(workout_history) >= 6:
                prev_week = sum(s['reps'] for session in workout_history[-6:-3] for s in session.get('sets', []))
                this_week = sum(s['reps'] for session in workout_history[-3:] for s in session.get('sets', []))
                if this_week > prev_week * 1.1:
                    improvement = ((this_week - prev_week) / prev_week) * 100
                    insights.append(f"Progressive overload! {improvement:.1f}% increase from last period!")
            
            # Exercise variety
            if len(exercises) >= 4:
                insights.append(f"Excellent variety! {len(exercises)} exercises for balanced development!")
            
            # Consistency rewards
            if len(workout_history) >= 10:
                insights.append(f"Consistency champion! {len(workout_history)} sessions logged!")
            
            # Specific exercise praise
            for ex in exercises:
                ex_lower = ex.lower()
                if 'squat' in ex_lower:
                    insights.append("Squat dedication! Building that foundation!")
                elif 'deadlift' in ex_lower:
                    insights.append("Deadlift power! The ultimate strength builder!")
                elif 'bench' in ex_lower or 'press' in ex_lower:
                    insights.append("Press mastery! Upper body strength developing!")
            
            if insights:
                import random
                return " ".join(random.sample(insights, min(2, len(insights))))
            
            return self._generate_generic_motivation()
            
        except Exception as e:
            logger.error(f"⚠️ Insight generation error: {e}")
            return self._generate_generic_motivation()
    
    def _generate_generic_motivation(self) -> str:
        """Context-aware generic motivation"""
        motivations = [
            "Welcome to your workout! Let's make it count!",
            "Time to build strength! Every rep matters!",
            "Consistency is key! You're doing great!",
            "Push your limits! You've got this!",
            "Strong body, strong mind! Let's go!",
            "Progress over perfection! Keep showing up!",
            "Your dedication is impressive! Keep it up!",
            "Every workout is a win! Stay committed!",
        ]
        import random
        return random.choice(motivations)
    
    def speak_text(self, text: str):
        """TTS with error handling"""
        if not text or not self.tts_engine:
            return
        
        try:
            if isinstance(self.tts_engine, str) and self.tts_engine == "gtts":
                self._speak_gtts(text)
            else:
                self._speak_pyttsx3(text)
        except Exception as e:
            logger.error(f"⚠️ TTS error: {e}")
    
    def _speak_pyttsx3(self, text: str):
        """Instant offline TTS"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"⚠️ pyttsx3 error: {e}")
    
    def _speak_gtts(self, text: str):
        """Online TTS with caching"""
        try:
            from gtts import gTTS
            import tempfile
            
            tts = gTTS(text=text, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                temp_file = f.name
                tts.save(temp_file)
            
            self._play_audio_file(temp_file)
            
            try:
                os.remove(temp_file)
            except:
                pass
        except Exception as e:
            logger.error(f"⚠️ gTTS error: {e}")
    
    def _play_audio_file(self, filepath: str):
        """Play audio with multiple backend support"""
        players = [
            ("play", f"play -q '{filepath}' 2>/dev/null &"),
            ("aplay", f"aplay -q '{filepath}' 2>/dev/null &"),
            ("afplay", f"afplay '{filepath}' 2>/dev/null &"),
        ]
        
        for player, cmd in players:
            if os.system(f"which {player} > /dev/null 2>&1") == 0:
                os.system(cmd)
                return
        
        logger.warning("⚠️ No audio player found")
    
    def schedule_insight(self, history: List[Dict]):
        """Non-blocking insight scheduling"""
        try:
            self.insight_queue.put_nowait(history.copy())
        except queue.Full:
            logger.warning("⚠️ Insight queue full")
    
    def process_insights_worker(self):
        """Background worker"""
        while self.running:
            try:
                history = self.insight_queue.get(timeout=1)
                text = self.generate_text_insight(history)
                logger.info(f"🎙️ Insight: {text}")
                self.speak_text(text)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"⚠️ Worker error: {e}")
    
    def start_background(self, tracker):
        """Start background processing"""
        worker = threading.Thread(target=self.process_insights_worker, daemon=True)
        worker.start()
        
        def scheduler():
            while self.running:
                try:
                    time.sleep(self.insight_interval)
                    if tracker.history and (time.time() - self.last_insight_time) >= self.insight_interval:
                        self.schedule_insight(tracker.history)
                        self.last_insight_time = time.time()
                except Exception as e:
                    logger.error(f"⚠️ Scheduler error: {e}")
        
        threading.Thread(target=scheduler, daemon=True).start()
    
    def shutdown(self):
        """Graceful shutdown"""
        self.running = False
        logger.info("🛑 Insights engine stopped")


class WorkoutTracker:
    """Enhanced workout tracking with adaptive thresholds and auto-recovery"""
    
    def __init__(self, 
                 data_file="workout_history.json",
                 metrics_file="realtime_metrics.json",
                 enable_insights=True,
                 tts_mode="fast",
                 use_ai_generation=False,
                 camera_id=0,
                 auto_save_interval=30):
        
        self.data_file = Path(data_file)
        self.metrics_file = Path(metrics_file)
        self.enable_insights = enable_insights
        self.camera_id = camera_id
        self.auto_save_interval = auto_save_interval
        self.history_lock = threading.Lock()
        self.last_auto_save = time.time()
        
        # Adaptive threshold manager
        self.threshold_manager = AdaptiveThresholdManager()
        
        # Load exercise classifier with error handling
        self._load_classifier()
        
        # Enhanced tracking parameters
        self.movement_threshold = 15.0
        self.rep_cooldown = 10
        self.cooldown_counter = 0
        self.motion_history = deque(maxlen=30)
        self.rep_state = "down"
        
        # Exercise classification with voting
        self.exercise_votes = deque(maxlen=5)
        self.stable_exercise_count = 0
        self.exercise_change_threshold = 3
        
        # Session state
        self.current_exercise = None
        self.current_reps = 0
        self.current_sets = []
        self.set_start_time = None
        self.last_rep_time = None
        
        self.prev_frame = None
        self.running = True
        self._classification_counter = 0
        self._frame_skip = 0
        
        # Performance metrics
        self.fps = 0
        self.frame_times = deque(maxlen=30)
        
        # Load history
        self.load_history()
        
        # Initialize insights engine
        if self.enable_insights:
            self._init_insights(tts_mode, use_ai_generation)
        
        # Register cleanup handler
        atexit.register(self.cleanup_handler)
        
        # Start auto-save thread
        self._start_auto_save()
    
    def _load_classifier(self):
        """Load exercise classifier with error handling"""
        logger.info("🏋️ Loading Gym Workout Classifier...")
        try:
            model_name = "prithivMLmods/Gym-Workout-Classifier-SigLIP2"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ Classifier ready on {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to load classifier: {e}")
            raise
    
    def _init_insights(self, tts_mode: str, use_ai: bool):
        """Initialize insights engine with error handling"""
        try:
            self.insights = SmartInsightsEngine(
                tts_mode=tts_mode,
                insight_interval=60,
                use_ai_generation=use_ai
            )
            self.insights.start_background(self)
        except Exception as e:
            logger.error(f"⚠️ Insights engine disabled: {e}")
            self.insights = None
    
    def _start_auto_save(self):
        """Start background auto-save thread"""
        def auto_saver():
            while self.running:
                time.sleep(self.auto_save_interval)
                if self.current_sets and (time.time() - self.last_auto_save) >= self.auto_save_interval:
                    self.save_history()
                    self.export_realtime_metrics()
                    self.last_auto_save = time.time()
        
        threading.Thread(target=auto_saver, daemon=True).start()
    
    def load_history(self):
        """Load workout history with validation"""
        with self.history_lock:
            if self.data_file.exists():
                try:
                    with open(self.data_file, "r") as f:
                        data = json.load(f)
                    
                    # Validate data structure
                    self.history = [s for s in data if isinstance(s, dict) and 'sets' in s]
                    logger.info(f"📂 Loaded {len(self.history)} sessions")
                except Exception as e:
                    logger.error(f"⚠️ Error loading history: {e}")
                    self.history = []
                    # Try to load backup
                    self._load_backup()
            else:
                self.history = []
    
    def _load_backup(self):
        """Load backup file if main file is corrupted"""
        backup_file = self.data_file.with_suffix('.backup.json')
        if backup_file.exists():
            try:
                with open(backup_file, "r") as f:
                    self.history = json.load(f)
                logger.info(f"✅ Loaded backup: {len(self.history)} sessions")
            except Exception as e:
                logger.error(f"⚠️ Backup also corrupted: {e}")
    
    def save_history(self):
        """Save with backup and atomic write"""
        with self.history_lock:
            try:
                # Create backup of existing file
                if self.data_file.exists():
                    backup_file = self.data_file.with_suffix('.backup.json')
                    import shutil
                    shutil.copy2(self.data_file, backup_file)
                
                # Atomic write using temp file
                temp_file = self.data_file.with_suffix('.tmp')
                with open(temp_file, "w") as f:
                    json.dump(self.history, f, indent=2)
                
                # Replace original with temp
                temp_file.replace(self.data_file)
                logger.info("💾 History saved")
            except Exception as e:
                logger.error(f"⚠️ Error saving history: {e}")
    
    def export_realtime_metrics(self):
        """Export real-time metrics for dashboard"""
        try:
            metrics = {
                "last_updated": datetime.now().isoformat(),
                "current_exercise": self.current_exercise,
                "current_reps": self.current_reps,
                "current_sets_count": len(self.current_sets),
                "session_duration": (datetime.now() - self.set_start_time).seconds if self.set_start_time else 0,
                "confidence_threshold": self.threshold_manager.get_threshold(),
                "stability_score": self.threshold_manager.stability_score,
                "fps": self.fps,
                "total_sessions": len(self.history),
                "total_reps_today": self._get_today_reps(),
            }
            
            with open(self.metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            logger.error(f"⚠️ Error exporting metrics: {e}")
    
    def _get_today_reps(self) -> int:
        """Calculate reps completed today"""
        today = datetime.now().date()
        total = 0
        with self.history_lock:
            for session in self.history:
                session_date = datetime.fromisoformat(session.get('date', '')).date()
                if session_date == today:
                    total += sum(s['reps'] for s in session.get('sets', []))
        return total
    
    def classify_exercise(self, frame: np.ndarray) -> Tuple[str, float]:
        """Enhanced classification with ensemble voting"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get top prediction
                top_probs, top_indices = torch.topk(probs[0], k=3)
                idx = top_indices[0].item()
                conf = top_probs[0].item()
                label = self.model.config.id2label[idx]
                
                # Add to voting queue
                self.exercise_votes.append((label, conf))
                
                # Ensemble voting for stability
                if len(self.exercise_votes) >= 3:
                    from collections import Counter
                    vote_labels = [v[0] for v in self.exercise_votes if v[1] > 0.3]
                    if vote_labels:
                        most_common = Counter(vote_labels).most_common(1)[0][0]
                        avg_conf = np.mean([v[1] for v in self.exercise_votes if v[0] == most_common])
                        return most_common, avg_conf
                
                # Log periodically
                self._classification_counter += 1
                if self._classification_counter % 15 == 0:
                    logger.info(f"🔍 Detection: {label} ({conf:.2%})")
                    if len(top_indices) >= 3:
                        for i in range(1, 3):
                            alt_label = self.model.config.id2label[top_indices[i].item()]
                            alt_conf = top_probs[i].item()
                            logger.info(f"   Alt: {alt_label} ({alt_conf:.2%})")
            
            return label, conf
        except Exception as e:
            logger.error(f"⚠️ Classification error: {e}")
            return "Unknown", 0.0
    
    def detect_motion(self, frame: np.ndarray) -> float:
        """Enhanced motion detection"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if self.prev_frame is None:
                self.prev_frame = gray
                return 0.0
            
            frame_delta = cv2.absdiff(self.prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Calculate motion intensity
            motion_intensity = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
            self.prev_frame = gray
            
            return motion_intensity * 100
        except Exception as e:
            logger.error(f"⚠️ Motion detection error: {e}")
            return 0.0
    
    def count_reps(self, motion_intensity: float) -> bool:
        """Improved rep counting with adaptive threshold"""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False
        
        self.motion_history.append(motion_intensity)
        if len(self.motion_history) < 30:
            return False
        
        recent_motion = np.mean(list(self.motion_history)[-10:])
        
        # Adaptive motion threshold
        motion_threshold = self.movement_threshold
        if np.std(list(self.motion_history)[-20:]) > 5:
            motion_threshold *= 1.2  # Increase threshold for noisy motion
        
        if self.rep_state == "down" and recent_motion > motion_threshold:
            self.rep_state = "up"
        elif self.rep_state == "up" and recent_motion < motion_threshold * 0.6:
            self.rep_state = "down"
            self.cooldown_counter = self.rep_cooldown
            self.last_rep_time = datetime.now()
            return True
        
        return False
    
    def start_new_set(self, exercise_name: str):
        """Start new set with validation"""
        self.current_exercise = exercise_name
        self.current_reps = 0
        self.set_start_time = datetime.now()
        logger.info(f"🏋️ New set: {exercise_name}")
    
    def end_current_set(self):
        """End set with validation"""
        if self.current_exercise and self.current_reps > 0:
            duration = (datetime.now() - self.set_start_time).seconds
            set_data = {
                "exercise": self.current_exercise,
                "reps": self.current_reps,
                "duration": duration,
                "timestamp": self.set_start_time.isoformat(),
            }
            self.current_sets.append(set_data)
            logger.info(f"✅ Set: {self.current_reps} reps in {duration}s")
            self.current_reps = 0
            self.set_start_time = None
    
    def save_session(self):
        """Save current session"""
        if self.current_sets:
            session = {
                "date": datetime.now().isoformat(),
                "sets": self.current_sets
            }
            with self.history_lock:
                self.history.append(session)
            
            self.save_history()
            logger.info(f"💾 Session: {len(self.current_sets)} sets")
            
            if self.insights:
                self.insights.schedule_insight(self.history)
            
            self.current_sets = []
    
    def calculate_fps(self, frame_time: float):
        """Calculate FPS"""
        self.frame_times.append(frame_time)
        if len(self.frame_times) >= 10:
            avg_time = np.mean(list(self.frame_times))
            self.fps = int(1.0 / avg_time) if avg_time > 0 else 0
    
    def run(self):
        """Main loop with enhanced error handling"""
        cap = None
        
        try:
            # Try multiple camera IDs
            for cam_id in [self.camera_id, 0, 1]:
                cap = cv2.VideoCapture(cam_id)
                if cap.isOpened():
                    logger.info(f"🎥 Camera {cam_id} opened")
                    break
                cap.release()
            
            if not cap or not cap.isOpened():
                logger.error("❌ No camera available")
                return
            
            # Optimize camera settings
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("🎬 Tracking started. Controls: s=save | i=insight | +/- =threshold | a=toggle AI | q=quit")
            
            frame_count = 0
            classify_interval = 30
            
            while self.running:
                frame_start = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    logger.warning("⚠️ Frame read failed")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Skip frames for performance
                self._frame_skip = (self._frame_skip + 1) % 2
                if self._frame_skip == 0:
                    # Classify exercise periodically
                    if frame_count % classify_interval == 0:
                        exercise, conf = self.classify_exercise(frame)
                        
                        # Update adaptive threshold
                        current_threshold = self.threshold_manager.update(
                            conf, 
                            exercise == self.current_exercise
                        )
                        
                        if conf > current_threshold:
                            if exercise != self.current_exercise:
                                self.stable_exercise_count += 1
                                
                                if self.stable_exercise_count >= self.exercise_change_threshold:
                                    self.end_current_set()
                                    self.start_new_set(exercise)
                                    self.stable_exercise_count = 0
                            else:
                                self.stable_exercise_count = 0
                
                # Motion detection and rep counting
                motion = self.detect_motion(frame)
                if self.current_exercise and self.count_reps(motion):
                    self.current_reps += 1
                    logger.info(f"✅ Rep #{self.current_reps}")
                
                # Draw UI
                self.draw_ui(frame, motion)
                cv2.imshow("AI Gym Tracker - Enhanced", frame)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("👋 Quit")
                    break
                elif key == ord("s"):
                    self.end_current_set()
                    self.save_session()
                elif key == ord("i"):
                    if self.insights and self.history:
                        self.insights.schedule_insight(self.history)
                elif key == ord("+") or key == ord("="):
                    self.threshold_manager.threshold = min(0.9, self.threshold_manager.threshold + 0.05)
                    logger.info(f"🎚️ Threshold: {self.threshold_manager.threshold:.2%}")
                elif key == ord("-"):
                    self.threshold_manager.threshold = max(0.1, self.threshold_manager.threshold - 0.05)
                    logger.info(f"🎚️ Threshold: {self.threshold_manager.threshold:.2%}")
                elif key == ord("a"):
                    if self.insights:
                        self.insights.use_ai_generation = not self.insights.use_ai_generation
                        mode = "AI" if self.insights.use_ai_generation else "Rule-based"
                        logger.info(f"🔄 Mode: {mode}")
                
                # Calculate FPS
                frame_time = time.time() - frame_start
                self.calculate_fps(frame_time)
                
                # Export metrics periodically
                if frame_count % 60 == 0:
                    self.export_realtime_metrics()
        
        except KeyboardInterrupt:
            logger.info("👋 Interrupted")
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}", exc_info=True)
        finally:
            self.cleanup(cap)
    
    def draw_ui(self, frame: np.ndarray, motion: float):
        """Enhanced UI with more metrics"""
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (550, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        y = 30
        line_height = 30
        
        # Main metrics
        cv2.putText(frame, f"Exercise: {self.current_exercise or 'Detecting...'}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += line_height
        
        cv2.putText(frame, f"Reps: {self.current_reps}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += line_height
        
        cv2.putText(frame, f"Motion: {motion:.1f}%",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y += line_height
        
        cv2.putText(frame, f"Sets Today: {len(self.current_sets)}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y += line_height
        
        # Advanced metrics
        cv2.putText(frame, f"Confidence: {self.threshold_manager.get_threshold():.1%}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 25
        
        cv2.putText(frame, f"Stability: {self.threshold_manager.stability_score:.2f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 25
        
        cv2.putText(frame, f"FPS: {self.fps}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += 25
        
        if self.insights:
            mode = "AI" if self.insights.use_ai_generation else "Rule"
            cv2.putText(frame, f"Insights: {mode}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Motion bar
        bar_width = int(min(motion * 3, 500))
        cv2.rectangle(frame, (10, 240), (10 + bar_width, 260), (0, 255, 0), -1)
        
        # Instructions
        cv2.putText(frame, "s=save | i=insight | +/-=threshold | a=AI | q=quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def cleanup(self, cap):
        """Enhanced cleanup"""
        logger.info("🧹 Cleanup started...")
        self.running = False
        
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        
        self.end_current_set()
        self.save_session()
        self.export_realtime_metrics()
        
        if self.insights:
            self.insights.shutdown()
        
        logger.info("✅ Cleanup complete")
    
    def cleanup_handler(self):
        """Atexit cleanup handler"""
        if self.running:
            logger.info("⚠️ Emergency cleanup")
            self.running = False
            self.end_current_set()
            self.save_session()


def main():
    """Enhanced entry point with configuration"""
    try:
        logger.info("=" * 70)
        logger.info("🏋️  AI-POWERED GYM WORKOUT TRACKER v2.5 ENHANCED")
        logger.info("=" * 70)
        
        tracker = WorkoutTracker(
            data_file="workout_history.json",
            metrics_file="realtime_metrics.json",
            enable_insights=True,
            tts_mode="fast",
            use_ai_generation=False,
            camera_id=0,
            auto_save_interval=30
        )
        
        logger.info("🚀 System ready!")
        logger.info("")
        logger.info("NEW FEATURES:")
        logger.info("  • Adaptive confidence threshold")
        logger.info("  • Auto-save every 30s")
        logger.info("  • Real-time metrics export")
        logger.info("  • Crash recovery with backup")
        logger.info("  • FPS monitoring")
        logger.info("")
        logger.info("KEYBOARD CONTROLS:")
        logger.info("  's' - Save current set")
        logger.info("  'i' - Trigger insight")
        logger.info("  '+/-' - Adjust threshold")
        logger.info("  'a' - Toggle AI insights")
        logger.info("  'q' - Quit")
        logger.info("")
        
        tracker.run()
        
    except KeyboardInterrupt:
        logger.info("👋 Stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)
    finally:
        logger.info("💪 Stay strong!")


if __name__ == "__main__":
    main()