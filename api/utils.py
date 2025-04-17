# api/utils.py
import cv2
import numpy as np
import pyaudio
import wave
import io
import logging

logger = logging.getLogger(__name__)

# Audio recording parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

def capture_frame():
    """Capture a frame from the webcam using OpenCV."""
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("Could not open webcam")

        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise Exception("Could not read frame from webcam")
        return frame
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        return None

def record_audio(duration: int = 5) -> bytes:
    """Record audio for a specified duration and return the audio data as bytes."""
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
        
        logger.info("Recording audio...")
        frames = []
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            frames.append(data)
        
        logger.info("Finished recording")
        
        # Convert frames to bytes
        audio_data = b''.join(frames)
        return audio_data
        
    except Exception as e:
        logger.error(f"Error recording audio: {e}")
        return b''  # Return empty bytes if there's an error
    finally:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()

def save_audio_to_wav(audio_data: bytes, filename: str = "temp_audio.wav"):
    """Save audio data to a temporary WAV file."""
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(audio_data)
        logger.info(f"Audio saved to {filename}")
        return filename
    except Exception as e:
        logger.error(f"Error saving audio to file: {e}")
        return None