# 🏋️ AI-Powered Gym Workout Tracker

> **Production-ready workout tracking system with real-time exercise detection, AI coaching, and advanced analytics**

Transform your webcam into an intelligent personal trainer. This system automatically detects exercises, counts reps, tracks your progress, and provides AI-powered insights with voice coaching.

![Version](https://img.shields.io/badge/version-2.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## ✨ Features

### 🎥 Real-Time Tracking
- **Automatic Exercise Detection** - AI identifies exercises using computer vision
- **Motion-Based Rep Counting** - Intelligent motion analysis tracks each rep
- **Set & Session Management** - Automatic set detection and history logging
- **Live Camera Feed** - Visual feedback with overlay statistics

### 🧠 AI-Powered Insights
- **Dual Insight Modes:**
  - **Rule-Based** (Default) - Fast, contextual, data-driven motivational feedback
  - **AI-Generated** (Optional) - Creative insights using Flan-T5 language model
- **Voice Coaching** - Text-to-speech motivation during workouts
- **Smart Fatigue Detection** - Analyzes performance drops
- **Rest Recommendations** - Suggests recovery based on trends
- **Progress Predictions** - Linear regression forecasts

### 📊 Advanced Analytics Dashboard
- **GitHub-Style Activity Heatmap** - Visualize workout consistency
- **Muscle Group Distribution** - Track which muscles you're training
- **Personal Records (PRs)** - Automatic tracking of best performances
- **Volume Trends** - Weekly aggregated performance charts
- **Recovery Score** - 0-100 score based on fatigue and rest metrics
- **Streak Tracking** - Current and best workout streaks
- **Achievement Badges** - Gamification with unlockable badges

### 🎯 Key Capabilities
- Multi-threaded architecture for smooth performance
- Adjustable confidence thresholds
- Export dashboard as high-resolution image
- Persistent workout history (JSON)
- Cross-platform support (Windows, macOS, Linux)

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Webcam
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ai-workout-tracker.git
cd ai-workout-tracker
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the tracker:**
```bash
python workout_tracker.py
```

4. **Open the dashboard:**
```bash
# Option 1: Serve locally (recommended)
python -m http.server 8000
# Then visit: http://localhost:8000/dashboard.html

# Option 2: Open directly in browser
# Open dashboard.html and use "Load Data" button
```

---

## 📦 Installation Details

### Required Python Packages
```bash
pip install opencv-python numpy transformers torch pillow pyttsx3
```

### Optional Packages
```bash
# For alternative TTS options:
pip install gtts scipy

# For GPU acceleration (CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### System Dependencies

**Linux (Debian/Ubuntu):**
```bash
sudo apt-get install python3-dev portaudio19-dev
# For audio playback:
sudo apt-get install sox espeak
```

**macOS:**
```bash
brew install portaudio
# Audio playback uses built-in afplay
```

**Windows:**
```bash
# No additional dependencies needed
# Uses built-in SAPI5 TTS
```

---

## 🎮 Usage Guide

### Tracker Controls

| Key | Action |
|-----|--------|
| `s` | Save current set and session |
| `i` | Trigger manual insight/motivation |
| `c` | Lower confidence threshold (more detections) |
| `r` | Raise confidence threshold (fewer detections) |
| `a` | Toggle AI/Rule-based insight generation |
| `q` | Quit and save |

### Configuration Options

Edit the `main()` function in `workout_tracker.py`:

```python
tracker = WorkoutTracker(
    data_file="workout_history.json",     # Data storage file
    enable_insights=True,                  # Enable voice coaching
    tts_mode="fast",                       # Options: "fast", "quality", "preloaded"
    use_ai_generation=False,               # True for AI, False for rule-based
    camera_id=0                            # Camera index (0 = default)
)
```

### TTS Modes

| Mode | Speed | Quality | Requirements |
|------|-------|---------|--------------|
| `fast` | ⚡ Instant | Good | pyttsx3 (offline) or gTTS (online) |
| `quality` | 🐌 5-10s | Excellent | Bark model (~500MB download) |
| `preloaded` | ⚡⚡ Instant | Good | Pre-cached audio files |

### Insight Generation Modes

**Rule-Based (Default)** ✅ Recommended
- Instant generation
- Contextual based on actual workout data
- Examples: "67 reps! Back development on point!"

**AI-Generated** 🤖 Experimental
- Uses Flan-T5 small model
- Creative varied responses
- May produce generic output
- Enable with `use_ai_generation=True`

---

## 🏗️ Project Structure

```
ai-workout-tracker/
├── workout_tracker.py          # Main tracking application
├── dashboard.html              # Analytics dashboard
├── workout_history.json        # Workout data (auto-generated)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── audio_cache/               # TTS audio cache (auto-generated)
```

---

## 📊 Supported Exercises

The system uses the **Gym-Workout-Classifier-SigLIP2** model with **96.38% accuracy** to detect 22 different exercises:

### Upper Body
**Chest:**
- Bench Press
- Decline Bench Press
- Incline Bench Press
- Chest Fly Machine
- Push-ups
- Tricep Dips

**Back:**
- Deadlift
- Romanian Deadlift
- Pull-ups
- Lat Pulldown
- T-Bar Row

**Shoulders:**
- Shoulder Press
- Lateral Raises

**Arms:**
- Barbell Biceps Curl
- Hammer Curl
- Tricep Pushdown

### Lower Body
- Squat
- Hip Thrust
- Leg Extension
- Leg Raises

### Core
- Plank
- Russian Twist

### Model Performance
- **Overall Accuracy:** 96.38%
- **Precision:** 96.47% (weighted avg)
- **Recall:** 96.38% (weighted avg)
- **F1-Score:** 96.39% (weighted avg)
- **Training Set:** 9,687 images across 22 classes

*The model is fine-tuned from Google's SigLIP2-base-patch16-224 architecture*

---

## 🔧 Advanced Configuration

### Adjusting Detection Sensitivity

```python
# In WorkoutTracker.__init__()
self.confidence_threshold = 0.4        # Lower = more sensitive (0.1-0.9)
self.exercise_change_threshold = 3     # Frames needed to confirm exercise
self.movement_threshold = 15.0         # Motion sensitivity for rep counting
```

### Custom Muscle Group Mapping

The dashboard maps the 22 exercises to muscle groups. Edit `EX_TO_MUSCLE` in `dashboard.html`:

```javascript
const EX_TO_MUSCLE = {
  // Chest
  'bench press': 'Chest',
  'decline bench press': 'Chest',
  'incline bench press': 'Chest',
  'chest fly machine': 'Chest',
  'push up': 'Chest',
  
  // Back
  'deadlift': 'Back',
  'romanian deadlift': 'Back',
  'pull up': 'Back',
  'lat pulldown': 'Back',
  't bar row': 'Back',
  
  // Shoulders
  'shoulder press': 'Shoulders',
  'lateral raises': 'Shoulders',
  
  // Arms
  'barbell biceps curl': 'Arms',
  'hammer curl': 'Arms',
  'tricep pushdown': 'Arms',
  'tricep dips': 'Arms',
  
  // Legs
  'squat': 'Legs',
  'hip thrust': 'Legs',
  'leg extension': 'Legs',
  'leg raises': 'Legs',
  
  // Core
  'plank': 'Core',
  'russian twist': 'Core'
};
```

### Weekly Goal Configuration

In `dashboard.html`, modify the weekly goal:

```javascript
function renderWeekProgress(){
  const weeklyGoal = 200;  // Change to your target reps
  // ...
}
```

---

## 📈 Dashboard Features

### Smart Insights

1. **Fatigue Detection**
   - Analyzes 6-day rolling performance
   - Detects drops >10% as potential fatigue
   - Visual indicators: High/Moderate/Low/None

2. **Rest Recommendations**
   - Compares recent vs. previous performance
   - Suggests rest days when performance drops >20%
   - Encourages consistency when trending up

3. **Muscle Balance Analysis**
   - Identifies undertrained muscle groups
   - Compares against average distribution
   - Provides specific recommendations

4. **Performance Prediction**
   - Uses linear regression on workout history
   - Predicts next session volume
   - Shows trending direction (up/down/stable)

5. **Recovery Score**
   - 0-100 scale based on fatigue and rest metrics
   - Visual progress bar
   - Updates in real-time

### Badge System

Unlock achievements by reaching milestones:

| Badge | Requirement |
|-------|-------------|
| 🔥 7-Day Streak | Workout 7 consecutive days |
| 🔥 30-Day Streak | Workout 30 consecutive days |
| 🔥 100-Day Streak | Workout 100 consecutive days |
| 🏆 First PR | Set your first personal record |
| 🏆 5 PRs | Set 5 personal records |
| 💪 500 Reps | Complete 500 total reps |
| 💪 1000 Reps | Complete 1000 total reps |
| 💪 5000 Reps | Complete 5000 total reps |
| 📊 10 Sessions | Complete 10 workout sessions |
| 📊 50 Sessions | Complete 50 workout sessions |

*Badges are stored in browser localStorage*

---

## 🐛 Troubleshooting

### Camera Not Opening
```bash
# List available cameras
python -c "import cv2; print([i for i in range(10) if cv2.VideoCapture(i).isOpened()])"

# Use different camera
tracker = WorkoutTracker(camera_id=1)  # Try 1, 2, etc.
```

### Exercise Not Detected

**Common Issues:**

1. **Low confidence scores** (< 40%)
   - Press `c` to lower threshold
   - Ensure good lighting (avoid shadows)
   - Position camera 5-8 feet away
   - Capture full body in frame

2. **Wrong exercise detected**
   - Model may confuse similar exercises
   - Check console for top 3 predictions
   - Ensure clear view of movement
   - Avoid cluttered backgrounds

3. **Exercise not in supported list**
   - Model only recognizes 22 specific exercises
   - Check the [Supported Exercises](#-supported-exercises) section
   - Consider similar alternatives (e.g., use "shoulder press" instead of "military press")

4. **Confidence fluctuating**
   - Increase `exercise_change_threshold` (default: 3 frames)
   - More stable but slower to detect changes

**Debug Mode:**
Check console output for real-time detection info:
```
🔍 Top 3 predictions:
   1. bench press: 92.5%
   2. incline bench press: 5.2%
   3. decline bench press: 1.8%
```

### TTS Not Working
```bash
# Test pyttsx3
python -c "import pyttsx3; engine=pyttsx3.init(); engine.say('test'); engine.runAndWait()"

# Install alternative
pip install gtts
```

### Dashboard Not Loading Data
1. Ensure `workout_history.json` exists
2. Serve via HTTP: `python -m http.server 8000`
3. Use "Load Data" button for local files
4. Check browser console for errors

### Low FPS / Lag
1. Use CPU-only mode (faster on some systems)
2. Reduce camera resolution
3. Close other applications
4. Disable AI insights: `enable_insights=False`

---

## 🔬 Technical Details

### Architecture

```
┌─────────────────────────────────────────┐
│         Main Thread (GUI Loop)          │
│  - Video capture                        │
│  - Exercise classification              │
│  - Motion detection                     │
│  - Rep counting                         │
│  - UI rendering                         │
└────────────┬────────────────────────────┘
             │
             ├──────────────────────────────┐
             │                              │
┌────────────▼────────────┐   ┌────────────▼──────────┐
│  Insight Worker Thread  │   │  Scheduler Thread     │
│  - Generate insights    │   │  - Trigger insights   │
│  - Text-to-speech       │   │    every 60s          │
│  - Queue processing     │   │  - Non-blocking       │
└─────────────────────────┘   └───────────────────────┘
```

### Models Used

1. **Exercise Classifier: Gym-Workout-Classifier-SigLIP2**
   - Base Model: `google/siglip2-base-patch16-224`
   - Fine-tuned by: prithivMLmods
   - Architecture: SigLIP (Vision-Language Encoder)
   - Input: 224x224 RGB images
   - Output: 22 exercise classes with confidence scores
   - Accuracy: 96.38%
   - Model Size: ~400MB
   - [HuggingFace Model Card](https://huggingface.co/prithivMLmods/Gym-Workout-Classifier-SigLIP2)

2. **Text Generation (Optional)**
   - Model: `google/flan-t5-small`
   - Type: Seq2Seq Transformer
   - Size: ~80MB
   - Purpose: Generate motivational insights
   - Parameters: 60M

3. **Text-to-Speech**
   - **pyttsx3**: Offline SAPI5/NSSpeechSynthesizer/espeak
   - **gTTS**: Google Text-to-Speech API (requires internet)
   - **Bark**: Neural codec language model (~500MB, high quality)

### Motion Detection Algorithm

```python
1. Convert frame to grayscale
2. Apply Gaussian blur (reduce noise)
3. Calculate frame difference (current - previous)
4. Threshold difference (binary mask)
5. Dilate mask (connect regions)
6. Calculate motion intensity (% of changed pixels)
7. Track intensity over 30-frame rolling window
8. Detect rep: up-peak → down-peak transition
```

### Data Format

`workout_history.json` structure:
```json
[
  {
    "date": "2025-10-24T14:30:00.000000",
    "sets": [
      {
        "exercise": "bench_press",
        "reps": 12,
        "duration": 45,
        "timestamp": "2025-10-24T14:30:00.000000"
      }
    ]
  }
]
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report Bugs** - Open an issue with detailed reproduction steps
2. **Suggest Features** - Share your ideas for improvements
3. **Add Exercises** - Expand the exercise classification mapping
4. **Improve Models** - Train better detection models
5. **Enhance Dashboard** - Add new visualizations or analytics

### Development Setup
```bash
git clone https://github.com/yourusername/ai-workout-tracker.git
cd ai-workout-tracker
pip install -e .
```

---

## 🙏 Acknowledgments

- **Exercise Classifier Model** - [prithivMLmods/Gym-Workout-Classifier-SigLIP2](https://huggingface.co/prithivMLmods/Gym-Workout-Classifier-SigLIP2) - 96.38% accuracy on 22 exercise classes
- **SigLIP Base Model** - [Google's SigLIP2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224) - Vision-language encoder
- **Flan-T5** - Google's instruction-tuned T5 model for text generation
- **Bark TTS** - Suno's high-quality neural text-to-speech model
- **Transformers Library** - Hugging Face for model deployment
- **Chart.js** - Beautiful, responsive data visualizations
- **html2canvas** - Client-side screenshot functionality
- **OpenCV** - Computer vision and motion detection algorithms

---

## 🗺️ Roadmap

### v2.1 (Planned)
- [ ] Mobile app (React Native)
- [ ] Cloud sync for multi-device support
- [ ] Social features (share workouts, compete with friends)
- [ ] Custom workout plans and scheduling
- [ ] Form correction using pose estimation

### v2.2 (Future)
- [ ] Wearable device integration (heart rate, calories)
- [ ] Video recording of sets
- [ ] Advanced biomechanics analysis
- [ ] Nutrition tracking integration
- [ ] AI-powered workout plan generation

---

## ⭐ Star History

If this project helped you, please consider giving it a star! ⭐

---

**Built with 💪 by the fitness tech community**

*Train smarter, not harder.*