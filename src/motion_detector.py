"""
Motion Detection Module

Frame differencing approach for detecting movement and counting repetitions.
Note: This component is proposed in the paper and requires experimental validation.

The motion detector uses temporal differencing to estimate movement intensity
and a state machine to count repetitions based on motion peaks.
"""

from collections import deque
from typing import Optional
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class MotionDetector:
    """
    Detects motion and counts exercise repetitions using frame differencing.
    
    Uses a state machine approach:
    - "down" state: Waiting for upward motion (motion intensity > threshold)
    - "up" state: Waiting for return to rest (motion intensity < threshold * decay)
    
    Attributes:
        movement_threshold: Motion intensity threshold for detecting movement
        history_size: Number of frames to consider for motion analysis
        rep_cooldown: Frames to wait between rep counts (prevents double counting)
    """
    
    def __init__(
        self,
        movement_threshold: float = 15.0,
        history_size: int = 30,
        rep_cooldown: int = 10
    ):
        """
        Initialize the motion detector.
        
        Args:
            movement_threshold: Minimum motion intensity to trigger state change
            history_size: Size of motion history buffer for smoothing
            rep_cooldown: Cooldown frames between repetition counts
        """
        self.movement_threshold = movement_threshold
        self.history_size = history_size
        self.rep_cooldown = rep_cooldown
        
        self._prev_frame: Optional[np.ndarray] = None
        self._motion_history: deque = deque(maxlen=history_size)
        self._rep_state: str = "down"
        self._cooldown_counter: int = 0
        self._rep_count: int = 0
    
    def detect_motion(self, frame: np.ndarray) -> float:
        """
        Calculate motion intensity from frame differencing.
        
        Args:
            frame: BGR image as numpy array (OpenCV format)
            
        Returns:
            Motion intensity as percentage (0-100)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self._prev_frame is None:
            self._prev_frame = gray
            return 0.0
        
        # Calculate frame difference
        frame_delta = cv2.absdiff(self._prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Calculate motion intensity as percentage of changed pixels
        motion_intensity = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
        
        self._prev_frame = gray
        return motion_intensity * 100
    
    def count_rep(self, motion_intensity: float) -> bool:
        """
        Check if a repetition was completed based on motion intensity.
        
        Uses a state machine with cooldown to prevent double-counting.
        
        Args:
            motion_intensity: Motion intensity from detect_motion()
            
        Returns:
            True if a new repetition was counted
        """
        # Handle cooldown
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
            return False
        
        self._motion_history.append(motion_intensity)
        
        # Need sufficient history
        if len(self._motion_history) < self.history_size:
            return False
        
        # Calculate recent motion average
        recent_motion = np.mean(list(self._motion_history)[-10:])
        
        # Adaptive threshold based on motion variance
        threshold = self.movement_threshold
        variance = np.std(list(self._motion_history)[-20:])
        if variance > 5:
            threshold *= 1.2  # Increase threshold for noisy environments
        
        # State machine logic
        if self._rep_state == "down" and recent_motion > threshold:
            self._rep_state = "up"
        elif self._rep_state == "up" and recent_motion < threshold * 0.6:
            self._rep_state = "down"
            self._cooldown_counter = self.rep_cooldown
            self._rep_count += 1
            return True
        
        return False
    
    def process_frame(self, frame: np.ndarray) -> tuple[float, bool]:
        """
        Process a frame and return motion info and rep status.
        
        Convenience method that combines detect_motion and count_rep.
        
        Args:
            frame: BGR image as numpy array
            
        Returns:
            Tuple of (motion_intensity, rep_counted)
        """
        motion = self.detect_motion(frame)
        rep_counted = self.count_rep(motion)
        return motion, rep_counted
    
    def reset(self) -> None:
        """Reset detector state."""
        self._prev_frame = None
        self._motion_history.clear()
        self._rep_state = "down"
        self._cooldown_counter = 0
        self._rep_count = 0
    
    @property
    def rep_count(self) -> int:
        """Get current repetition count."""
        return self._rep_count
    
    @rep_count.setter
    def rep_count(self, value: int) -> None:
        """Set repetition count (for reset or manual adjustment)."""
        self._rep_count = value
    
    @property
    def current_state(self) -> str:
        """Get current state machine state ('up' or 'down')."""
        return self._rep_state
