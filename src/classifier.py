"""
Exercise Classification Module

Uses the pre-trained Gym-Workout-Classifier-SigLIP2 model from HuggingFace.
Model: https://huggingface.co/prithivMLmods/Gym-Workout-Classifier-SigLIP2
DOI: 10.57967/hf/5391

Classification metrics (reported by original model authors):
- Accuracy: 96.38%
- Test set: 9,687 images
- Classes: 22 gym exercises

We did not train this model. Our contribution is the real-time integration,
temporal smoothing (3-frame consensus), and proposed system architecture.
"""

from collections import deque, Counter
from typing import Tuple, Optional, List
import logging

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

logger = logging.getLogger(__name__)

# 22 Exercise classes supported by the model
EXERCISE_CLASSES = [
    "barbell biceps curl",
    "bench press",
    "chest fly machine",
    "deadlift",
    "decline bench press",
    "hammer curl",
    "hip thrust",
    "incline bench press",
    "lat pulldown",
    "lateral raises",
    "leg extension",
    "leg raises",
    "plank",
    "pull up",
    "push up",
    "romanian deadlift",
    "russian twist",
    "shoulder press",
    "squat",
    "t bar row",
    "tricep dips",
    "tricep pushdown",
]


class ExerciseClassifier:
    """
    Classifies gym exercises from images using a pre-trained vision model.
    
    Implements temporal smoothing via ensemble voting over recent predictions
    to reduce noise and false positives during exercise transitions.
    
    Attributes:
        model_name: HuggingFace model identifier
        device: Computation device (cuda or cpu)
        confidence_threshold: Minimum confidence for valid predictions
    """
    
    MODEL_NAME = "prithivMLmods/Gym-Workout-Classifier-SigLIP2"
    
    def __init__(
        self,
        confidence_threshold: float = 0.4,
        vote_window: int = 5,
        consensus_threshold: int = 3,
        device: Optional[str] = None
    ):
        """
        Initialize the exercise classifier.
        
        Args:
            confidence_threshold: Minimum confidence score for predictions
            vote_window: Number of recent predictions to consider for voting
            consensus_threshold: Minimum votes needed to change exercise
            device: Force specific device (cuda/cpu), auto-detect if None
        """
        self.confidence_threshold = confidence_threshold
        self.consensus_threshold = consensus_threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Voting mechanism for temporal smoothing
        self._vote_history: deque = deque(maxlen=vote_window)
        self._current_exercise: Optional[str] = None
        self._stable_count: int = 0
        
        # Load model
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the pre-trained model from HuggingFace."""
        logger.info(f"Loading exercise classifier from {self.MODEL_NAME}")
        
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForImageClassification.from_pretrained(self.MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Classifier ready on {self.device}")
    
    def classify_frame(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Classify a single video frame.
        
        Args:
            frame: BGR image as numpy array (OpenCV format)
            
        Returns:
            Tuple of (exercise_name, confidence_score)
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            top_prob, top_idx = torch.max(probs[0], dim=0)
            label = self.model.config.id2label[top_idx.item()]
            confidence = top_prob.item()
        
        return label, confidence
    
    def classify_with_smoothing(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Classify with temporal smoothing using ensemble voting.
        
        Uses a sliding window of recent predictions and requires consensus
        before confirming an exercise change. This reduces false positives
        during transitions between exercises.
        
        Args:
            frame: BGR image as numpy array (OpenCV format)
            
        Returns:
            Tuple of (exercise_name, average_confidence)
            Returns (None, 0.0) if no stable prediction available
        """
        label, confidence = self.classify_frame(frame)
        
        # Only consider predictions above threshold
        if confidence < self.confidence_threshold:
            return self._current_exercise, 0.0
        
        self._vote_history.append((label, confidence))
        
        if len(self._vote_history) < self.consensus_threshold:
            return self._current_exercise, 0.0
        
        # Count votes for each exercise
        vote_labels = [v[0] for v in self._vote_history]
        vote_counts = Counter(vote_labels)
        most_common, count = vote_counts.most_common(1)[0]
        
        # Calculate average confidence for winning label
        matching_confs = [v[1] for v in self._vote_history if v[0] == most_common]
        avg_confidence = np.mean(matching_confs)
        
        # Update current exercise if consensus reached
        if count >= self.consensus_threshold:
            if most_common != self._current_exercise:
                self._stable_count += 1
                if self._stable_count >= self.consensus_threshold:
                    self._current_exercise = most_common
                    self._stable_count = 0
            else:
                self._stable_count = 0
        
        return self._current_exercise, avg_confidence
    
    def get_top_k_predictions(
        self, 
        frame: np.ndarray, 
        k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get top-k predictions for a frame.
        
        Args:
            frame: BGR image as numpy array
            k: Number of top predictions to return
            
        Returns:
            List of (exercise_name, confidence) tuples
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            top_probs, top_indices = torch.topk(probs[0], k=min(k, len(probs[0])))
            
            results = []
            for prob, idx in zip(top_probs, top_indices):
                label = self.model.config.id2label[idx.item()]
                results.append((label, prob.item()))
        
        return results
    
    def reset(self) -> None:
        """Reset the voting history and current exercise state."""
        self._vote_history.clear()
        self._current_exercise = None
        self._stable_count = 0
    
    @property
    def current_exercise(self) -> Optional[str]:
        """Get the currently detected exercise."""
        return self._current_exercise
