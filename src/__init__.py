"""
Source modules for Exercise Classification System.
"""

from .classifier import ExerciseClassifier, EXERCISE_CLASSES
from .motion_detector import MotionDetector

__all__ = ["ExerciseClassifier", "MotionDetector", "EXERCISE_CLASSES"]
