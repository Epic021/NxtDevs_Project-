import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class AttentionState(Enum):
    FOCUSED = "Focused"
    THINKING = "Thinking"
    LOSING_ATTENTION = "Losing Attention"
    STUCK = "Stuck"

@dataclass
class EyeMetrics:
    ear: float  # Eye Aspect Ratio
    gaze_direction: Tuple[float, float]  # (x, y) direction vector
    blink_rate: float  # Blinks per second
    fixation_duration: float  # Seconds looking at same point
    saccade_frequency: float  # Saccades per second

class EyeTracker:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Constants
        self.EAR_THRESHOLD = 0.22
        self.FIXATION_THRESHOLD = 1.5
        self.SACCADE_THRESHOLD = 0.08
        self.BLINK_RATE_THRESHOLD = 0.4
        self.GAZE_AWAY_THRESHOLD = 0.25
        
        # State tracking
        self.blink_history = deque(maxlen=30)
        self.gaze_history = deque(maxlen=30)
        self.fixation_start_time = None
        self.last_gaze_point = None
        self.frame_count = 0
        self.last_blink_time = 0
        self.blink_count = 0
        
    def calculate_ear(self, landmarks, eye_indices: List[int]) -> float:
        """Calculate Eye Aspect Ratio (EAR) for given eye landmarks."""
        points = np.array([(landmarks[idx].x, landmarks[idx].y) for idx in eye_indices])
        
        # Calculate vertical distances
        v1 = np.linalg.norm(points[1] - points[5])
        v2 = np.linalg.norm(points[2] - points[4])
        
        # Calculate horizontal distance
        h = np.linalg.norm(points[0] - points[3])
        
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def calculate_gaze_direction(self, landmarks, eye_indices: List[int]) -> Tuple[float, float]:
        """Calculate gaze direction vector for given eye landmarks."""
        # Get eye corners and center
        left_corner = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
        right_corner = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
        center = np.array([landmarks[eye_indices[4]].x, landmarks[eye_indices[4]].y])
        
        # Calculate direction vector
        direction = center - (left_corner + right_corner) / 2
        return tuple(direction / np.linalg.norm(direction))

    def detect_blink(self, ear: float) -> bool:
        """Detect if a blink occurred based on EAR."""
        is_blink = ear < self.EAR_THRESHOLD
        self.blink_history.append(is_blink)
        return is_blink

    def calculate_metrics(self, frame: np.ndarray) -> Optional[EyeMetrics]:
        """Calculate all eye metrics for the current frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        # Define eye landmark indices (MediaPipe Face Mesh)
        left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(landmarks, left_eye_indices)
        right_ear = self.calculate_ear(landmarks, right_eye_indices)
        ear = (left_ear + right_ear) / 2.0

        # Calculate gaze direction
        left_gaze = self.calculate_gaze_direction(landmarks, left_eye_indices)
        right_gaze = self.calculate_gaze_direction(landmarks, right_eye_indices)
        gaze = ((left_gaze[0] + right_gaze[0]) / 2, (left_gaze[1] + right_gaze[1]) / 2)

        # Update blink rate
        is_blink = self.detect_blink(ear)
        if is_blink:
            self.blink_count += 1

        # Calculate blink rate (blinks per second)
        current_time = self.frame_count / 30.0  # Assuming 30 FPS
        time_since_last_blink = current_time - self.last_blink_time

        if time_since_last_blink >= 1.0:
            blink_rate = self.blink_count
            self.blink_count = 0
            self.last_blink_time = current_time
        else:
            blink_rate = self.blink_count / (time_since_last_blink if time_since_last_blink > 1e-6 else 1e-6)

        # Calculate fixation duration and saccade frequency
        current_gaze = np.array(gaze)
        if self.last_gaze_point is not None:
            gaze_distance = np.linalg.norm(current_gaze - self.last_gaze_point)
            if gaze_distance < self.SACCADE_THRESHOLD:
                if self.fixation_start_time is None:
                    self.fixation_start_time = current_time
                fixation_duration = current_time - self.fixation_start_time
            else:
                self.fixation_start_time = None
                fixation_duration = 0
        else:
            fixation_duration = 0
            self.fixation_start_time = current_time

        self.last_gaze_point = current_gaze
        self.frame_count += 1

        # Calculate saccade frequency
        saccade_frequency = 1.0 / (time_since_last_blink if time_since_last_blink > 1e-6 else 1e-6)

        return EyeMetrics(
            ear=ear,
            gaze_direction=gaze,
            blink_rate=blink_rate,
            fixation_duration=fixation_duration,
            saccade_frequency=saccade_frequency
        )

    def classify_attention_state(self, metrics: EyeMetrics) -> AttentionState:
        """Classify attention state based on eye metrics."""
        # Check for losing attention
        if (abs(metrics.gaze_direction[0]) > self.GAZE_AWAY_THRESHOLD or 
            metrics.gaze_direction[1] < -self.GAZE_AWAY_THRESHOLD) and \
           metrics.blink_rate > self.BLINK_RATE_THRESHOLD:
            return AttentionState.LOSING_ATTENTION
        
        # Check for stuck state
        if metrics.fixation_duration > self.FIXATION_THRESHOLD and \
           metrics.blink_rate < self.BLINK_RATE_THRESHOLD:
            return AttentionState.STUCK
        
        # Check for thinking state
        if (metrics.gaze_direction[1] > 0.2 or  # Looking up
            abs(metrics.gaze_direction[0]) > 0.3) and \
            metrics.ear < 0.35:  # Slight squint
            return AttentionState.THINKING
        
        return AttentionState.FOCUSED

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[AttentionState], Optional[EyeMetrics]]:
        """Process a single frame and return attention state and metrics."""
        metrics = self.calculate_metrics(frame)
        if metrics is None:
            return None, None
            
        attention_state = self.classify_attention_state(metrics)
        return attention_state, metrics

def main():
    tracker = EyeTracker()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        attention_state, metrics = tracker.process_frame(frame)
        
        if attention_state and metrics:
            # Display attention state with color coding
            state_colors = {
                AttentionState.FOCUSED: (0, 255, 0),  # Green
                AttentionState.THINKING: (255, 255, 0),  # Yellow
                AttentionState.LOSING_ATTENTION: (0, 0, 255),  # Red
                AttentionState.STUCK: (255, 0, 0)  # Blue
            }
            
            color = state_colors[attention_state]
            
            # Display attention state with background
            cv2.rectangle(frame, (5, 5), (300, 180), (0, 0, 0), -1)
            cv2.putText(frame, f"State: {attention_state.value}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Display metrics with better formatting
            cv2.putText(frame, f"EAR: {metrics.ear:.2f}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Blink Rate: {metrics.blink_rate:.1f}/s", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Fixation: {metrics.fixation_duration:.1f}s", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display gaze direction
            gaze_x, gaze_y = metrics.gaze_direction
            center_x, center_y = frame.shape[1]//2, frame.shape[0]//2
            end_x = int(center_x + gaze_x * 100)
            end_y = int(center_y + gaze_y * 100)
            # cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y), (0, 255, 255), 2)
        
        cv2.imshow("Eye Tracking", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
