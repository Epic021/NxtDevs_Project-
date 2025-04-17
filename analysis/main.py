import cv2
import time
from facial import main as facial_analysis
from eye import EyeTracker
from speech import analyze_speech
from data import InterviewDataManager
import pyaudio
import wave
import threading
import queue
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterviewAnalyzer:
    def __init__(self):
        self.data_manager = InterviewDataManager()
        self.eye_tracker = EyeTracker()
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.audio_frames = []
        
        # Audio recording parameters
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        
        # Initialize video capture
        self.cap = None
        self.audio_thread = None
        
    def start_interview(self):
        """Start a new interview session."""
        try:
            interview_id = self.data_manager.start_new_interview()
            logger.info(f"Started interview session: {interview_id}")
            
            # Start audio recording thread
            self.is_recording = True
            self.audio_thread = threading.Thread(target=self._record_audio)
            self.audio_thread.start()
            
            # Start video capture
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open video capture device")
            
            self._run_interview_loop()
            
        except Exception as e:
            logger.error(f"Error starting interview: {str(e)}")
            self.stop_interview()
            raise
    
    def _run_interview_loop(self):
        """Main interview analysis loop."""
        last_analysis_time = time.time()
        analysis_interval = 1.0  # Analyze every second
        
        # Initialize variables
        facial_data = None
        eye_state = None
        eye_metrics = None
        speech_data = None
        
        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    break
                
                current_time = time.time()
                
                # Only analyze if enough time has passed
                if current_time - last_analysis_time >= analysis_interval:
                    try:
                        # Run facial analysis with current frame
                        facial_data = facial_analysis(return_data=True, frame=frame)
                        if not facial_data:
                            logger.warning("No facial data received")
                            facial_data = {"emotion": "unknown", "confidence": 0.0}
                    except Exception as e:
                        logger.error(f"Error in facial analysis: {str(e)}")
                        facial_data = {"emotion": "unknown", "confidence": 0.0}
                    
                    try:
                        # Run eye tracking
                        eye_state, eye_metrics = self.eye_tracker.process_frame(frame)
                        if not eye_state:
                            logger.warning("No eye tracking data received")
                            eye_state = type('obj', (object,), {'value': 'unknown'})
                            eye_metrics = type('obj', (object,), {'confidence': 0.0})
                    except Exception as e:
                        logger.error(f"Error in eye tracking: {str(e)}")
                        eye_state = type('obj', (object,), {'value': 'unknown'})
                        eye_metrics = type('obj', (object,), {'confidence': 0.0})
                    
                    try:
                        # Process audio if available
                        if not self.audio_queue.empty():
                            audio_data = self.audio_queue.get()
                            speech_data = analyze_speech(audio_data)
                    except Exception as e:
                        logger.error(f"Error analyzing speech: {str(e)}")
                        speech_data = None
                    
                    # Add metrics to data manager if we have all data
                    if facial_data and eye_state and speech_data:
                        try:
                            self.data_manager.add_metrics(
                                facial_data=facial_data,
                                eye_data={
                                    "attention_state": eye_state.value,
                                    "confidence_score": eye_metrics.confidence if hasattr(eye_metrics, 'confidence') else 0.5
                                },
                                speech_data=speech_data
                            )
                        except Exception as e:
                            logger.error(f"Error adding metrics: {str(e)}")
                    
                    last_analysis_time = current_time
                
                # Display frame with metrics
                self._display_metrics(frame, facial_data, eye_state, speech_data)
                
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            logger.error(f"Error in interview loop: {str(e)}")
        finally:
            self.stop_interview()
    
    def stop_interview(self):
        """Stop the interview and save data."""
        logger.info("Stopping interview...")
        
        self.is_recording = False
        
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2.0)
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        try:
            # Save interview data
            filepath = self.data_manager.save_interview()
            logger.info(f"Interview data saved to: {filepath}")
        except Exception as e:
            logger.error(f"Error saving interview data: {str(e)}")
    
    def _record_audio(self):
        """Record audio in a separate thread."""
        p = pyaudio.PyAudio()
        stream = None
        
        try:
            stream = p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK
            )
            
            while self.is_recording:
                try:
                    data = stream.read(self.CHUNK)
                    self.audio_frames.append(data)
                    
                    # Process audio in chunks of 5 seconds
                    if len(self.audio_frames) >= (self.RATE * 5) // self.CHUNK:
                        audio_data = b''.join(self.audio_frames)
                        self.audio_queue.put(audio_data)
                        self.audio_frames = []
                        
                except Exception as e:
                    logger.error(f"Error recording audio: {str(e)}")
                    break
                    
        except Exception as e:
            logger.error(f"Error setting up audio recording: {str(e)}")
        finally:
            if stream:
                stream.stop_stream()
                stream.close()
            p.terminate()
    
    def _display_metrics(self, frame, facial_data, eye_state, speech_data):
        """Display current metrics on the frame."""
        try:
            # Display facial expression
            if facial_data:
                cv2.putText(frame, f"Facial: {facial_data.get('emotion', 'unknown')}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display eye tracking
            if eye_state:
                cv2.putText(frame, f"Eye: {eye_state.value}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display speech metrics
            if speech_data:
                cv2.putText(frame, f"Speech: {speech_data.get('analysis', {}).get('speech_quality', 'unknown')}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Interview Analysis", frame)
        except Exception as e:
            logger.error(f"Error displaying metrics: {str(e)}")

def main():
    try:
        analyzer = InterviewAnalyzer()
        logger.info("Starting interview analysis...")
        logger.info("Press 'q' to end the interview")
        analyzer.start_interview()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    main() 