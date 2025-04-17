# api/endpoints.py
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from analysis.eye import EyeTracker, AttentionState
from analysis.facial import main as facial_analysis
from analysis.speech import analyze_speech
from analysis.data import InterviewDataManager
from analysis.visual import InterviewVisualizer
from api.models import FacialAnalysisResult, SpeechAnalysisResult, InterviewData
from api.utils import capture_frame, record_audio, save_audio_to_wav
from typing import Optional, Dict, List # Import Dict and List
import cv2
import numpy as np
import io
import os
import logging
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

data_manager = InterviewDataManager(base_dir=config.DATA_DIR)
eye_tracker = EyeTracker()  # Initialize EyeTracker
visualizer = InterviewVisualizer(data_dir=config.DATA_DIR) # Initialize Visualizer
# Create necessary directories if they don't exist
os.makedirs(config.DATA_DIR, exist_ok=True)
os.makedirs(config.VISUALIZATIONS_DIR, exist_ok=True)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Interview Analysis API"}

@app.post("/start_interview")
async def start_interview():
    """Start a new interview session."""
    try:
        interview_id = data_manager.start_new_interview()
        return {"interview_id": interview_id, "message": "Interview started successfully"}
    except Exception as e:
        logger.error(f"Error starting interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_frame")
async def analyze_frame():
    """Analyze a single frame from the webcam."""
    try:
        # Capture frame from webcam
        frame = capture_frame()
        if frame is None:
            raise HTTPException(status_code=500, detail="Failed to capture frame from webcam")
        
        # Perform facial analysis
        facial_result = facial_analysis(return_data=True, frame=frame)
        if facial_result is None:
            logger.warning("Facial analysis returned None")
            facial_result = {"emotion": "unknown", "confidence": 0.0}
        
        # Perform eye tracking
        eye_state, eye_metrics = eye_tracker.process_frame(frame)
        if eye_state is None or eye_metrics is None:
            logger.warning("Eye tracking returned None")
            eye_state = type('obj', (object,), {'value': 'unknown'})
            eye_metrics = type('obj', (object,), {'confidence': 0.0})
        
        return {
            "facial_analysis": facial_result,
            "eye_tracking": {
                "attention_state": eye_state.value if eye_state else 'unknown',
                "confidence": eye_metrics.confidence if hasattr(eye_metrics, 'confidence') else 0.5
            }
        }
    except Exception as e:
        logger.error(f"Error analyzing frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_audio")
async def analyze_audio():
    """Analyze audio from a recorded segment."""
    try:
        # Record audio for 5 seconds
        audio_data = record_audio(duration=5)

        # Perform speech analysis
        speech_result = analyze_speech(audio_data)

        return speech_result
    except Exception as e:
        logger.error(f"Error analyzing audio: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_interview_metrics")
async def add_interview_metrics(
    facial_data: Dict,
    eye_data: Dict,
    speech_data: Dict
):
    """Add metrics from facial, eye, and speech analysis to the current interview."""
    try:
        data_manager.add_metrics(
            facial_data=facial_data,
            eye_data=eye_data,
            speech_data=speech_data
        )
        return {"message": "Metrics added to interview successfully"}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error adding metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/end_interview")
async def end_interview():
    """End the current interview session and save the data."""
    try:
        file_path = data_manager.save_interview()
        return {"message": "Interview ended and data saved successfully", "file_path": file_path}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error ending interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_interview_data/{interview_number}", response_model=InterviewData)
async def get_interview_data(interview_number: int):
    """Retrieve interview data by interview number."""
    try:
        data = data_manager.load_interview_data(interview_number)
        if not data['metrics']:
            raise HTTPException(status_code=404, detail="Interview data not found")
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Interview data not found")
    except Exception as e:
        logger.error(f"Error getting interview data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualize_interview/{interview_number}", response_class=HTMLResponse)
async def visualize_interview(interview_number: int):
    """Visualize interview data."""
    try:
        # Generate visualizations
        fig = visualizer.create_comprehensive_dashboard(interview_number)
        
        # Convert the plot to an HTML string
        html = fig.to_html(full_html=False, include_plotlyjs='cdn')
        return HTMLResponse(content=html)
    except Exception as e:
        logger.error(f"Error visualizing interview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_insights/{interview_number}")
async def generate_insights(interview_number: int):
    """
    Generates insights and recommendations based on the interview data.
    """
    try:
        data = data_manager.load_interview_data(interview_number)
        if not data['metrics']:
            raise HTTPException(status_code=404, detail="Interview data not found")

        # Here you would integrate your insights generation logic (summary.py)
        # For now, just return the data for demonstration purposes

        return {"message": "Insights generated successfully", "data": data}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Interview data not found")
    except Exception as e:
        logger.error(f"Error generating insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))