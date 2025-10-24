import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification, pipeline
import torch
from PIL import Image
from datetime import datetime
import json
import os
from collections import deque
import threading
import queue
import logging
from pathlib import Path
import warnings
import time

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartInsightsEngine:
    """
    Production-ready insight generator with fast TTS
    Supports: Rule-based insights (fast) OR AI-generated (creative)
    TTS Options: pyttsx3 (instant), gTTS (fast), Bark (quality)
    """

    def __init__(self, 
                 text_model_name="google/flan-t5-small",
                 tts_mode="fast",  # "fast", "quality", or "preloaded"
                 insight_interval=60,
                 use_ai_generation=True):  # Toggle AI text generation
        self.text_model_name = text_model_name
        self.tts_mode = tts_mode
        self.insight_interval = insight_interval
        self.use_ai_generation = use_ai_generation
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.insight_queue = queue.Queue()
        self.history_lock = threading.Lock()
        self.running = True
        
        # Only load text model if AI generation is enabled
        if self.use_ai_generation:
            logger.info(f"🧠 Loading text model ({text_model_name}) on {self.device}...")
            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                
                self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
                self.text_model = AutoModelForSeq2SeqLM.from_pretrained(text_model_name)
                self.text_model.to(self.device)
                self.text_model.eval()
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                logger.info("✅ Text model ready.")
            except Exception as e:
                logger.error(f"❌ Failed to load text model: {e}")
                self.tokenizer = None
                self.text_model = None
                self.use_ai_generation = False
        else:
            logger.info("🧠 Using rule-based insights (faster, more reliable)")
            self.tokenizer = None
            self.text_model = None

        # Initialize TTS based on mode
        self.setup_tts()

    def setup_tts(self):
        """Setup TTS with optimization based on mode"""
        if self.tts_mode == "preloaded":
            # Fastest: Use preloaded motivational audio clips
            logger.info("🎧 Using preloaded audio clips (fastest)")
            self.tts_pipeline = None
            self.preloaded_audio = self._create_preloaded_audio()
            
        elif self.tts_mode == "fast":
            # Fast: Use lightweight TTS (gTTS or pyttsx3)
            logger.info("🎧 Using fast TTS (gTTS/pyttsx3)")
            try:
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', 180)  # Speed up
                self.tts_engine.setProperty('volume', 0.9)
                self.tts_pipeline = None
                logger.info("✅ pyttsx3 TTS ready (instant)")
            except ImportError:
                logger.warning("⚠️ pyttsx3 not found, trying gTTS...")
                try:
                    from gtts import gTTS
                    self.tts_engine = "gtts"
                    self.tts_pipeline = None
                    logger.info("✅ gTTS ready (fast)")
                except ImportError:
                    logger.error("❌ No fast TTS available. Install: pip install pyttsx3 or gtts")
                    self.tts_engine = None
                    self.tts_pipeline = None
                    
        else:  # quality mode
            # High quality but slower
            logger.info("🎧 Using high-quality TTS (Bark - slower)")
            try:
                self.tts_pipeline = pipeline(
                    "text-to-speech",
                    model="suno/bark-small",
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("✅ Bark TTS ready.")
            except Exception as e:
                logger.error(f"❌ Failed to load TTS: {e}")
                self.tts_pipeline = None

    def _create_preloaded_audio(self):
        """Create preloaded audio clips for common messages"""
        clips = {
            "welcome": "Welcome to your workout! Let's get started!",
            "great_work": "Great work! Keep pushing!",
            "milestone": "Awesome progress! You're crushing it!",
            "finish_strong": "Finish strong! You've got this!",
        }
        
        # In production, these would be pre-recorded audio files
        # For now, we'll generate them on first run and cache
        audio_dir = Path("audio_cache")
        audio_dir.mkdir(exist_ok=True)
        
        preloaded = {}
        for key, text in clips.items():
            audio_file = audio_dir / f"{key}.wav"
            if audio_file.exists():
                preloaded[key] = str(audio_file)
            else:
                # Generate once using fast TTS
                preloaded[key] = None  # Will use text fallback
        
        return preloaded

    def generate_text_insight(self, workout_history):
        """Generate motivational insight - with AI or rule-based option"""
        
        # Try AI generation if enabled and model is loaded
        if self.use_ai_generation and self.text_model:
            try:
                ai_insight = self._generate_ai_insight(workout_history)
                if ai_insight and len(ai_insight) > 10 and "great work" not in ai_insight.lower():
                    logger.info(f"🤖 AI Insight: {ai_insight}")
                    return ai_insight
                else:
                    logger.warning("⚠️ AI generated poor response, using rule-based")
            except Exception as e:
                logger.error(f"⚠️ AI generation failed: {e}")
        
        # Use rule-based insights (default or fallback)
        return self._generate_rule_based_insight(workout_history)
    
    def _generate_ai_insight(self, workout_history):
        """Generate insight using Flan-T5 AI model"""
        if not workout_history:
            return None
        
        recent = workout_history[-3:]
        total_reps = sum(s['reps'] for session in recent for s in session.get('sets', []))
        exercises = list(set(s['exercise'] for session in recent for s in session.get('sets', [])))
        total_sets = sum(len(session.get('sets', [])) for session in recent)
        
        # Simpler, more direct prompt for small models
        prompt = f"You completed {total_reps} reps doing {', '.join(exercises)}. Give motivational feedback:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.text_model.generate(
                **inputs,
                max_length=50,
                min_length=10,
                temperature=0.9,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                attention_mask=inputs.attention_mask,
                num_beams=1
            )
        
        insight = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return insight.strip()
    
    def _generate_rule_based_insight(self, workout_history):
        """Generate smart rule-based insights from workout data"""
        if not workout_history:
            return self._generate_generic_motivation()
        
        try:
            # Analyze workout data
            recent = workout_history[-3:]
            total_reps = sum(s['reps'] for session in recent for s in session.get('sets', []))
            exercises = list(set(s['exercise'] for session in recent for s in session.get('sets', [])))
            total_sets = sum(len(session.get('sets', [])) for session in recent)
            
            # Generate smart insights based on data patterns
            insights = []
            
            # Rep-based insights
            if total_reps > 100:
                insights.append(f"Incredible! {total_reps} reps completed. You're a machine!")
            elif total_reps > 50:
                insights.append(f"Solid work! {total_reps} reps in your recent sessions!")
            elif total_reps > 20:
                insights.append(f"Great consistency! {total_reps} reps and counting!")
            
            # Exercise variety insights
            if len(exercises) >= 3:
                insights.append(f"Love the variety! {len(exercises)} different exercises. Well-rounded training!")
            elif len(exercises) == 2:
                insights.append(f"Nice focus on {' and '.join(exercises)}. Keep it up!")
            elif len(exercises) == 1:
                insights.append(f"Mastering {exercises[0]}! Consider adding variety next time.")
            
            # Set-based insights
            if total_sets >= 10:
                insights.append(f"{total_sets} sets completed! Your dedication is impressive!")
            elif total_sets >= 5:
                insights.append(f"{total_sets} sets down! You're building real strength!")
            
            # Time-based insights
            if len(workout_history) >= 5:
                insights.append("Five sessions logged! Consistency is key to progress!")
            elif len(workout_history) >= 3:
                insights.append("Building a great routine! Keep showing up!")
            
            # Specific exercise insights
            for ex in exercises:
                if 'push' in ex.lower() or 'press' in ex.lower():
                    insights.append("Chest and triceps getting stronger! Push power building!")
                elif 'pull' in ex.lower() or 'row' in ex.lower():
                    insights.append("Back development on point! Pull exercises are crucial!")
                elif 'squat' in ex.lower():
                    insights.append("Leg day warrior! Squats build total body strength!")
                elif 'deadlift' in ex.lower():
                    insights.append("Deadlifts! The king of exercises. Great choice!")
                elif 'curl' in ex.lower():
                    insights.append("Arm gains incoming! Biceps love the work!")
            
            # Select best insight or combine
            if insights:
                import random
                if len(insights) >= 2:
                    return f"{random.choice(insights[:2])} {random.choice(insights[1:]) if len(insights) > 1 else ''}"
                return random.choice(insights)
            
            # Fallback to motivational quotes
            return self._generate_generic_motivation()
            
        except Exception as e:
            logger.error(f"⚠️ Insight generation error: {e}")
            return self._generate_generic_motivation()
    
    def _generate_generic_motivation(self):
        """Generate context-aware generic motivation"""
        motivations = [
            "Welcome to your workout! Let's crush those goals today!",
            "Ready to build strength? Let's get started!",
            "Time to push your limits! You've got this!",
            "Let's make today count! Start strong!",
            "Every rep counts! Give it your all!",
            "Consistency beats perfection! Keep showing up!",
            "Your future self will thank you for this workout!",
            "Strong body, strong mind! Let's go!",
            "Progress, not perfection! You're doing great!",
            "The only bad workout is the one you didn't do!"
        ]
        import random
        return random.choice(motivations)

    def speak_text(self, text):
        """Convert text to speech - OPTIMIZED VERSION"""
        if not text:
            return
        
        logger.info(f"🎙️ Speaking: {text[:50]}...")
        
        try:
            if self.tts_mode == "preloaded":
                # Fastest: Play preloaded audio
                self._play_preloaded(text)
                
            elif self.tts_mode == "fast" and hasattr(self, 'tts_engine'):
                if self.tts_engine == "gtts":
                    # gTTS (requires internet but fast)
                    self._speak_gtts(text)
                else:
                    # pyttsx3 (offline, instant)
                    self._speak_pyttsx3(text)
                    
            elif self.tts_pipeline:
                # High quality but slower
                self._speak_bark(text)
            else:
                logger.warning("⚠️ No TTS available")
                
        except Exception as e:
            logger.error(f"⚠️ TTS error: {e}")

    def _speak_pyttsx3(self, text):
        """Instant offline TTS using pyttsx3"""
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def _speak_gtts(self, text):
        """Fast online TTS using gTTS"""
        from gtts import gTTS
        import tempfile
        
        # Generate speech
        tts = gTTS(text=text, lang='en', slow=False)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
            temp_file = f.name
            tts.save(temp_file)
        
        # Play audio
        self._play_audio_file(temp_file)
        
        # Cleanup
        try:
            os.remove(temp_file)
        except:
            pass

    def _speak_bark(self, text):
        """High quality TTS using Bark (slower)"""
        speech = self.tts_pipeline(text, forward_params={"do_sample": True})
        
        audio_path = Path("insight.wav")
        import scipy.io.wavfile as wavfile
        wavfile.write(
            str(audio_path),
            rate=speech["sampling_rate"],
            data=speech["audio"].squeeze()
        )
        
        self._play_audio_file(str(audio_path))

    def _play_preloaded(self, text):
        """Play preloaded audio clip"""
        # Match text to closest preloaded clip
        if "welcome" in text.lower() or "start" in text.lower():
            key = "welcome"
        elif "milestone" in text.lower() or "progress" in text.lower():
            key = "milestone"
        elif "finish" in text.lower():
            key = "finish_strong"
        else:
            key = "great_work"
        
        audio_file = self.preloaded_audio.get(key)
        if audio_file and os.path.exists(audio_file):
            self._play_audio_file(audio_file)
        else:
            # Fallback to fast TTS
            logger.info("⚠️ Preloaded audio not found, using text")

    def _play_audio_file(self, filepath):
        """Play audio file using available player"""
        if os.system("which play > /dev/null 2>&1") == 0:
            os.system(f"play -q '{filepath}' 2>/dev/null &")  # Background play
        elif os.system("which aplay > /dev/null 2>&1") == 0:
            os.system(f"aplay -q '{filepath}' 2>/dev/null &")
        elif os.system("which afplay > /dev/null 2>&1") == 0:  # macOS
            os.system(f"afplay '{filepath}' 2>/dev/null &")
        else:
            logger.warning("⚠️ No audio player found")

    def run_once(self, history):
        """Generate and speak one insight"""
        with self.history_lock:
            history_copy = history.copy()
        
        text = self.generate_text_insight(history_copy)
        self.speak_text(text)
        return text

    def process_insights_worker(self):
        """Background worker for processing insight requests"""
        logger.info(f"🧠 Insight worker started (interval: {self.insight_interval}s)")
        while self.running:
            try:
                history = self.insight_queue.get(timeout=1)
                self.run_once(history)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"⚠️ Insight worker error: {e}")

    def schedule_insight(self, history):
        """Schedule an insight generation (non-blocking)"""
        try:
            self.insight_queue.put_nowait(history.copy())
        except queue.Full:
            logger.warning("⚠️ Insight queue full, skipping...")

    def start_background(self, tracker):
        """Start background insight generation"""
        worker = threading.Thread(target=self.process_insights_worker, daemon=True)
        worker.start()
        
        def scheduler():
            while self.running:
                try:
                    time.sleep(self.insight_interval)
                    if tracker.history:
                        self.schedule_insight(tracker.history)
                except Exception as e:
                    logger.error(f"⚠️ Scheduler error: {e}")
        
        threading.Thread(target=scheduler, daemon=True).start()

    def shutdown(self):
        """Graceful shutdown"""
        self.running = False
        logger.info("🛑 Insights engine shutting down...")


class WorkoutTracker:
    """Production-ready workout tracking system with AI insights"""
    
    def __init__(self, 
                 data_file="workout_history.json",
                 enable_insights=True,
                 tts_mode="fast",  # "fast", "quality", or "preloaded"
                 use_ai_generation=False,  # Toggle AI vs rule-based insights
                 camera_id=0):
        
        self.data_file = Path(data_file)
        self.enable_insights = enable_insights
        self.camera_id = camera_id
        self.history_lock = threading.Lock()
        
        # Load exercise classifier
        logger.info("🏋️ Loading Gym Workout Classifier...")
        try:
            model_name = "prithivMLmods/Gym-Workout-Classifier-SigLIP2"
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✅ Gym Classifier ready on {self.device}")
        except Exception as e:
            logger.error(f"❌ Failed to load classifier: {e}")
            raise

        # Tracking parameters
        self.movement_threshold = 15.0
        self.rep_cooldown = 10
        self.cooldown_counter = 0
        self.motion_history = deque(maxlen=30)
        self.rep_state = "down"

        # Session state
        self.current_exercise = None
        self.current_reps = 0
        self.current_sets = []
        self.set_start_time = None
        self.last_rep_time = None
        self.confidence_threshold = 0.4  # Lowered for better detection
        self.stable_exercise_count = 0
        self.exercise_change_threshold = 3  # Faster response

        self.prev_frame = None
        self.running = True
        self._classification_counter = 0

        # Load history
        self.load_history()

        # Initialize insights engine with TTS mode
        if self.enable_insights:
            try:
                self.insights = SmartInsightsEngine(
                    insight_interval=60,
                    tts_mode=tts_mode,
                    use_ai_generation=use_ai_generation
                )
                self.insights.start_background(self)
            except Exception as e:
                logger.error(f"⚠️ Insights engine disabled: {e}")
                self.insights = None
        else:
            self.insights = None

    def load_history(self):
        """Load workout history from JSON"""
        with self.history_lock:
            if self.data_file.exists():
                try:
                    with open(self.data_file, "r") as f:
                        self.history = json.load(f)
                    logger.info(f"📂 Loaded {len(self.history)} previous sessions")
                except Exception as e:
                    logger.error(f"⚠️ Error loading history: {e}")
                    self.history = []
            else:
                self.history = []

    def save_history(self):
        """Save workout history to JSON"""
        with self.history_lock:
            try:
                with open(self.data_file, "w") as f:
                    json.dump(self.history, f, indent=2)
                logger.info("💾 History saved")
            except Exception as e:
                logger.error(f"⚠️ Error saving history: {e}")

    def classify_exercise(self, frame):
        """Classify exercise from frame with debugging"""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                
                # Get top 3 predictions for debugging
                top_probs, top_indices = torch.topk(probs[0], k=3)
                
                idx = top_indices[0].item()
                conf = top_probs[0].item()
                
                # Log top predictions periodically
                self._classification_counter += 1
                
                if self._classification_counter % 10 == 0:  # Every 10 classifications
                    logger.info("🔍 Top 3 predictions:")
                    for i in range(3):
                        label = self.model.config.id2label[top_indices[i].item()]
                        prob = top_probs[i].item()
                        logger.info(f"   {i+1}. {label}: {prob:.2%}")
            
            return self.model.config.id2label[idx], conf
        except Exception as e:
            logger.error(f"⚠️ Classification error: {e}")
            return "Unknown", 0.0

    def detect_motion(self, frame):
        """Detect motion intensity in frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if self.prev_frame is None:
                self.prev_frame = gray
                return 0.0
            
            frame_delta = cv2.absdiff(self.prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            motion_intensity = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
            self.prev_frame = gray
            
            return motion_intensity * 100
        except Exception as e:
            logger.error(f"⚠️ Motion detection error: {e}")
            return 0.0

    def count_reps(self, motion_intensity):
        """Count reps based on motion patterns"""
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return False

        self.motion_history.append(motion_intensity)
        if len(self.motion_history) < 30:
            return False

        recent_motion = np.mean(list(self.motion_history)[-10:])
        
        if self.rep_state == "down" and recent_motion > self.movement_threshold:
            self.rep_state = "up"
        elif self.rep_state == "up" and recent_motion < self.movement_threshold * 0.6:
            self.rep_state = "down"
            self.cooldown_counter = self.rep_cooldown
            self.last_rep_time = datetime.now()
            return True
        
        return False

    def start_new_set(self, exercise_name):
        """Start tracking a new set"""
        self.current_exercise = exercise_name
        self.current_reps = 0
        self.set_start_time = datetime.now()
        logger.info(f"🏋️ Started new set: {exercise_name}")

    def end_current_set(self):
        """End current set and save data"""
        if self.current_exercise and self.current_reps > 0:
            duration = (datetime.now() - self.set_start_time).seconds
            set_data = {
                "exercise": self.current_exercise,
                "reps": self.current_reps,
                "duration": duration,
                "timestamp": self.set_start_time.isoformat(),
            }
            self.current_sets.append(set_data)
            logger.info(f"✅ Set completed: {self.current_reps} reps in {duration}s")
            self.current_reps = 0
            self.set_start_time = None

    def save_session(self):
        """Save current workout session"""
        if self.current_sets:
            session = {
                "date": datetime.now().isoformat(),
                "sets": self.current_sets
            }
            with self.history_lock:
                self.history.append(session)
            self.save_history()
            
            logger.info(f"💾 Session saved with {len(self.current_sets)} sets")
            
            # Trigger immediate insight
            if self.insights:
                self.insights.schedule_insight(self.history)
            
            self.current_sets = []

    def run(self):
        """Main tracking loop"""
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            logger.error(f"❌ Cannot open camera {self.camera_id}")
            return
        
        logger.info("🎥 Camera ready. Controls: s=save | i=insight | c=lower threshold | r=raise threshold | q=quit")
        frame_count = 0
        classify_interval = 30

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("⚠️ Failed to read frame")
                    break

                frame_count += 1
                
                # Classify exercise periodically
                if frame_count % classify_interval == 0:
                    exercise, conf = self.classify_exercise(frame)
                    
                    # Debug output
                    logger.info(f"📊 Detection: {exercise} (confidence: {conf:.2%})")
                    
                    if conf > self.confidence_threshold:
                        if exercise != self.current_exercise:
                            self.stable_exercise_count += 1
                            logger.info(f"🔄 New exercise detected: {exercise} (stability: {self.stable_exercise_count}/{self.exercise_change_threshold})")
                            
                            if self.stable_exercise_count >= self.exercise_change_threshold:
                                self.end_current_set()
                                self.start_new_set(exercise)
                                self.stable_exercise_count = 0
                        else:
                            self.stable_exercise_count = 0
                    else:
                        logger.info(f"⚠️ Low confidence: {conf:.2%} < {self.confidence_threshold:.2%}")

                # Detect motion and count reps
                motion = self.detect_motion(frame)
                if self.current_exercise and self.count_reps(motion):
                    self.current_reps += 1
                    logger.info(f"✅ Rep #{self.current_reps}")

                # Draw UI
                self.draw_ui(frame, motion)
                cv2.imshow("Gym Workout Tracker - AI Powered", frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("👋 Quit requested")
                    break
                elif key == ord("s"):
                    self.end_current_set()
                    self.save_session()
                elif key == ord("i"):
                    logger.info("🎤 Manual insight triggered")
                    if self.insights and self.history:
                        self.insights.schedule_insight(self.history)
                    else:
                        logger.warning("⚠️ No workout history yet for insights")
                elif key == ord("c"):
                    # Lower confidence threshold for better detection
                    self.confidence_threshold = max(0.1, self.confidence_threshold - 0.1)
                    logger.info(f"🎚️ Confidence threshold lowered to {self.confidence_threshold:.1%}")
                elif key == ord("r"):
                    # Raise confidence threshold
                    self.confidence_threshold = min(0.9, self.confidence_threshold + 0.1)
                    logger.info(f"🎚️ Confidence threshold raised to {self.confidence_threshold:.1%}")
                elif key == ord("a"):
                    # Toggle AI generation
                    if self.insights:
                        self.insights.use_ai_generation = not self.insights.use_ai_generation
                        mode = "AI" if self.insights.use_ai_generation else "Rule-based"
                        logger.info(f"🔄 Switched to {mode} insight generation")

        except KeyboardInterrupt:
            logger.info("👋 Interrupted by user")
        except Exception as e:
            logger.error(f"❌ Runtime error: {e}")
        finally:
            self.cleanup(cap)

    def draw_ui(self, frame, motion):
        """Draw UI overlay on frame with enhanced debugging"""
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (500, 240), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        
        y_offset = 30
        cv2.putText(frame, f"Exercise: {self.current_exercise or 'Detecting...'}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Reps: {self.current_reps}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Motion: {motion:.1f}%",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        y_offset += 30
        cv2.putText(frame, f"Sets: {len(self.current_sets)}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Show confidence threshold
        y_offset += 30
        cv2.putText(frame, f"Confidence: {self.confidence_threshold:.1%}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show insight mode
        y_offset += 25
        if self.insights:
            mode = "AI" if self.insights.use_ai_generation else "Rule-based"
            cv2.putText(frame, f"Insights: {mode}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Show stability counter
        y_offset += 25
        if self.stable_exercise_count > 0:
            cv2.putText(frame, f"Stability: {self.stable_exercise_count}/{self.exercise_change_threshold}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        
        # Motion bar
        bar_width = int(min(motion * 3, 450))
        cv2.rectangle(frame, (10, 210), (10 + bar_width, 230), (0, 255, 0), -1)
        
        # Instructions
        cv2.putText(frame, "s=save | i=insight | c/r=threshold | a=toggle AI | q=quit",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def cleanup(self, cap):
        """Cleanup resources"""
        logger.info("🧹 Cleaning up...")
        self.running = False
        self.end_current_set()
        self.save_session()
        cap.release()
        cv2.destroyAllWindows()
        
        if self.insights:
            self.insights.shutdown()
        
        logger.info("✅ Cleanup complete")


def main():
    """
    Entry point for AI-Powered Gym Workout Tracker
    
    Configuration Options:
    - tts_mode: "fast" (pyttsx3/gTTS), "quality" (Bark), "preloaded" (cached audio)
    - use_ai_generation: False (rule-based), True (Flan-T5 AI)
    - camera_id: 0 (default webcam) or other camera index
    """
    try:
        logger.info("=" * 60)
        logger.info("🏋️  AI-POWERED GYM WORKOUT TRACKER v2.0")
        logger.info("=" * 60)
        
        tracker = WorkoutTracker(
            data_file="workout_history.json",
            enable_insights=True,
            tts_mode="fast",  # Options: "fast", "quality", "preloaded"
            use_ai_generation=False,  # Toggle: True for AI, False for rule-based
            camera_id=0
        )
        
        logger.info("🚀 System ready! Starting workout tracker...")
        logger.info("")
        logger.info("KEYBOARD CONTROLS:")
        logger.info("  's' - Save current set")
        logger.info("  'i' - Trigger insight/motivation")
        logger.info("  'c' - Lower confidence threshold (more detections)")
        logger.info("  'r' - Raise confidence threshold (fewer detections)")
        logger.info("  'a' - Toggle AI/Rule-based insights")
        logger.info("  'q' - Quit")
        logger.info("")
        
        tracker.run()
        
    except KeyboardInterrupt:
        logger.info("👋 Workout tracker stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        logger.info("💪 Thanks for working out! Stay strong!")


if __name__ == "__main__":
    main()