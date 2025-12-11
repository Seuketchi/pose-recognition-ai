# Exercise Classification Using Vision-Language Models

Real-time gym exercise classification from webcam footage using a pre-trained vision-language model.

## Model Attribution

This project uses [Gym-Workout-Classifier-SigLIP2](https://huggingface.co/prithivMLmods/Gym-Workout-Classifier-SigLIP2) by prithivMLmods.

| Metric | Value |
|--------|-------|
| Accuracy | 96.38% |
| Classes | 22 |
| Test Images | 9,687 |

*Metrics reported by original model authors. See [docs/MODEL_ATTRIBUTION.md](docs/MODEL_ATTRIBUTION.md) for details.*

## Installation

```bash
# Clone the repository
git clone https://github.com/Seuketchi/pose-recognition-ai.git
cd pose-recognition-ai

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- Webcam (for real-time classification)
- CUDA-capable GPU (optional, for faster inference)

## Usage

### Real-time Demo

```bash
python scripts/run_demo.py
```

Options:
- `--camera 0` - Camera device ID (default: 0)
- `--threshold 0.4` - Confidence threshold (default: 0.4)
- `--no-smoothing` - Disable temporal smoothing

### Single Image Inference

```bash
# Classify a single image
python scripts/inference.py image.jpg

# Classify with top-3 predictions
python scripts/inference.py image.jpg --top-k 3

# Classify a directory of images
python scripts/inference.py images/ --output results.json

# List supported exercises
python scripts/inference.py --list-classes
```

### Full Workout Tracker

```bash
python app/workout_tracker.py
```

Keyboard controls:
- `s` - Save current set
- `i` - Trigger voice insight
- `+/-` - Adjust confidence threshold
- `a` - Toggle AI/rule-based insights
- `q` - Quit

### Analytics Dashboard

```bash
# Start local server
python -m http.server 8000

# Open in browser
# http://localhost:8000/app/dashboard.html
```

## Project Structure

```
pose-recognition-ai/
├── src/
│   ├── classifier.py        # Exercise classification module
│   └── motion_detector.py   # Motion detection for rep counting
├── scripts/
│   ├── run_demo.py          # Simple webcam demo
│   └── inference.py         # Single image inference
├── app/
│   ├── workout_tracker.py   # Full workout application
│   └── dashboard.html       # Analytics dashboard
├── docs/
│   └── MODEL_ATTRIBUTION.md # Model credits and details
├── requirements.txt
├── LICENSE
└── CITATION.cff
```

## Our Contributions

We did not train the classification model. Our contributions are:

1. **Real-time webcam integration** - Video capture and processing with OpenCV
2. **Temporal smoothing** - 3-frame consensus mechanism to reduce false positives during transitions between exercises
3. **Motion-based repetition counting** - Frame differencing approach for detecting exercise repetitions (proposed)
4. **Analytics dashboard** - Web-based visualization of workout history and progress

## Supported Exercises

The model classifies 22 gym exercises:

| | | |
|---|---|---|
| barbell biceps curl | bench press | chest fly machine |
| deadlift | decline bench press | hammer curl |
| hip thrust | incline bench press | lat pulldown |
| lateral raises | leg extension | leg raises |
| plank | pull up | push up |
| romanian deadlift | russian twist | shoulder press |
| squat | t bar row | tricep dips |
| tricep pushdown | | |

## Citation

```bibtex
@inproceedings{jadman2025exercise,
  title={Exercise Classification Using Vision-Language Models: A Deep Learning Approach for Automated Gym Exercise Recognition},
  author={Jadman, Tristan and Cruza, Ian James},
  year={2025}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
