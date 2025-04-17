# api/models.py
from pydantic import BaseModel, Dict, List, Optional, validator
from enum import Enum
from typing import Tuple
from datetime import datetime

class AttentionState(str, Enum):
    FOCUSED = "Focused"
    THINKING = "Thinking"
    LOSING_ATTENTION = "Losing Attention"
    STUCK = "Stuck"

class EyeMetrics(BaseModel):
    ear: float
    gaze_direction: Tuple[float, float]
    blink_rate: float
    fixation_duration: float
    saccade_frequency: float

class FacialAnalysisResult(BaseModel):
    emotion: str
    confidence: float
    emotion_history: List[str]
    emotion_counts: Dict[str, int]

class SpeechAnalysisResult(BaseModel):
    transcript: str
    metrics: Dict
    analysis: Dict
    sentiment: str
    emotion: str

class InterviewMetrics(BaseModel):
    timestamp: datetime
    facial_expression: Dict
    eye_tracking: Dict
    speech_analysis: Dict
    overall_confidence: float
    engagement_score: float
    improvement_suggestions: List[str]

class InterviewData(BaseModel):
    interview_id: str
    timestamp: datetime
    metrics: List[Dict]
    summary: Dict
    key_insights: List[str]
    improvement_areas: Dict