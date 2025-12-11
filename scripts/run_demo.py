#!/usr/bin/env python3
"""
Real-time Exercise Classification Demo

A simple demonstration of the exercise classifier using webcam input.
Press 'q' to quit.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --camera 1
    python scripts/run_demo.py --threshold 0.5
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classifier import ExerciseClassifier


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time exercise classification demo"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.4,
        help="Confidence threshold (default: 0.4)"
    )
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable temporal smoothing"
    )
    return parser.parse_args()


def main() -> None:
    """Run the real-time exercise classification demo."""
    args = parse_args()
    
    print("=" * 60)
    print("Exercise Classification Demo")
    print("=" * 60)
    print(f"Camera: {args.camera}")
    print(f"Confidence threshold: {args.threshold}")
    print(f"Temporal smoothing: {'disabled' if args.no_smoothing else 'enabled'}")
    print("Press 'q' to quit")
    print("=" * 60)
    
    # Initialize classifier
    print("\nLoading model...")
    classifier = ExerciseClassifier(confidence_threshold=args.threshold)
    print("Model ready!\n")
    
    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        sys.exit(1)
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    fps = 0
    fps_timer = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: Failed to read frame")
                continue
            
            frame_count += 1
            
            # Classify every 15 frames to reduce load
            if frame_count % 15 == 0:
                if args.no_smoothing:
                    exercise, confidence = classifier.classify_frame(frame)
                else:
                    exercise, confidence = classifier.classify_with_smoothing(frame)
            else:
                exercise = classifier.current_exercise
                confidence = 0.0
            
            # Calculate FPS
            if time.time() - fps_timer >= 1.0:
                fps = frame_count
                frame_count = 0
                fps_timer = time.time()
            
            # Draw UI
            draw_overlay(frame, exercise, confidence, fps)
            
            cv2.imshow("Exercise Classification Demo", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Demo ended")


def draw_overlay(
    frame,
    exercise: str | None,
    confidence: float,
    fps: int
) -> None:
    """Draw classification results on frame."""
    h, w = frame.shape[:2]
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (400, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    
    # Exercise name
    display_name = exercise or "Detecting..."
    cv2.putText(
        frame, f"Exercise: {display_name}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )
    
    # Confidence
    if confidence > 0:
        cv2.putText(
            frame, f"Confidence: {confidence:.1%}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )
    
    # FPS
    cv2.putText(
        frame, f"FPS: {fps}",
        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
    )
    
    # Instructions
    cv2.putText(
        frame, "Press 'q' to quit",
        (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
    )


if __name__ == "__main__":
    main()
