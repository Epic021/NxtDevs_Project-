import cv2
import torch
import numpy as np 
from PIL import Image
from collections import deque
from typing import List, Dict, Optional
from transformers import AutoImageProcessor, AutoModelForImageClassification
from contextlib import contextmanager

# Constants
FRAME_SKIP = 2  # Process every nth frame
TARGET_WIDTH = 640  # Target width for resizing frames
EMOTION_HISTORY_SIZE = 30
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for emotion classification

@contextmanager
def open_camera():
    """Context manager for camera handling."""
    cap = cv2.VideoCapture(0)
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        yield cap
    finally:
        cap.release()
        cv2.destroyAllWindows()

def load_model() -> tuple:
    """Load and return the model and processor."""
    try:
        processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
        model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
        model.eval()
        return processor, model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def classify_frame(frame: np.ndarray, processor, model) -> str:
    """Classify emotion in a single frame with confidence check."""
    try:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        max_prob, predicted_class = torch.max(probabilities, dim=-1)
        
        if max_prob.item() < CONFIDENCE_THRESHOLD:
            return "unknown"
            
        emotion = model.config.id2label[predicted_class.item()]
        return emotion.lower()
    except Exception as e:
        print(f"Error in frame classification: {e}")
        return "unknown"

def infer_modified_emotion(history: List[str]) -> str:
    """Infer modified emotion based on history with improved logic."""
    if not history:
        return "Analyzing..."

    # Filter out unknown emotions
    history = [e for e in history if e != "unknown"]
    if not history:
        return "Unknown"

    counts = {e: history.count(e) for e in set(history)}
    total = sum(counts.values())
    sorted_emotions = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    top_emotion, top_count = sorted_emotions[0]

    ratio = top_count / total

    # Let neutral stand alone if it's dominant enough
    if top_emotion == "neutral" and ratio > 0.6:
        return "Neutral"

    # Only say 'Confident' when happy dominates and neutral supports
    if "happy" in counts and counts["happy"] > 0.4 * total:
        if "neutral" in counts and counts["neutral"] > 0.2 * total:
            return "Confident"

    if "sad" in counts and counts["sad"] > 0.4 * total:
        return "Anxious"

    if "angry" in counts or "disgust" in counts:
        if counts.get("angry", 0) + counts.get("disgust", 0) > 0.35 * total:
            return "Stressed"

    if "fear" in counts and counts["fear"] > 0.3 * total:
        return "Nervous"

    if "surprised" in counts and counts["surprised"] > 0.3 * total:
        return "Alert"

    return top_emotion.capitalize()

def main(return_data: bool = False, frame: Optional[np.ndarray] = None) -> Optional[Dict]:
    """Main function for facial analysis.
    
    Args:
        return_data: Whether to return the analysis data instead of displaying it
        frame: Optional frame to analyze. If None, will capture from camera
        
    Returns:
        Dict containing analysis results if return_data is True, None otherwise
    """
    try:
        processor, model = load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

    emotion_history = deque(maxlen=EMOTION_HISTORY_SIZE)
    frame_count = 0

    try:
        if frame is None:
            # Use camera if no frame provided
            with open_camera() as cap:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Could not read frame")
                        break

                    frame_count += 1
                    if frame_count % FRAME_SKIP != 0:
                        continue

                    height, width = frame.shape[:2]
                    aspect_ratio = width / height
                    new_height = int(TARGET_WIDTH / aspect_ratio)
                    frame = cv2.resize(frame, (TARGET_WIDTH, new_height))

                    emotion = classify_frame(frame, processor, model)
                    emotion_history.append(emotion)

                    modified_emotion = infer_modified_emotion(list(emotion_history))

                    cv2.putText(frame, f"Behavior: {modified_emotion}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    cv2.imshow("Live Emotion Detection", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        else:
            # Analyze provided frame
            height, width = frame.shape[:2]
            aspect_ratio = width / height
            new_height = int(TARGET_WIDTH / aspect_ratio)
            frame = cv2.resize(frame, (TARGET_WIDTH, new_height))

            emotion = classify_frame(frame, processor, model)
            emotion_history.append(emotion)

            modified_emotion = infer_modified_emotion(list(emotion_history))

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        if return_data:
            from collections import Counter
            return {
                "emotion": modified_emotion,
                "confidence": 0.8 if emotion != "unknown" else 0.3,
                "emotion_history": list(emotion_history),
                "emotion_counts": dict(Counter(emotion_history))
            }

if __name__ == "__main__":
    main()
