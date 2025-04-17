import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import uuid
import numpy as np

@dataclass
class InterviewMetrics:
    timestamp: str
    facial_expression: Dict
    eye_tracking: Dict
    speech_analysis: Dict
    overall_confidence: float
    engagement_score: float
    improvement_suggestions: List[str]

class InterviewDataManager:
    def __init__(self, base_dir: str = "interview_data"):
        self.base_dir = base_dir
        self.current_interview_id = None
        self.current_data = []
        
        # Create base directory if it doesn't exist
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
    
    def start_new_interview(self) -> str:
        """Start a new interview session and return its ID."""
        self.current_interview_id = str(uuid.uuid4())
        self.current_data = []
        return self.current_interview_id
    
    def add_metrics(self, 
                   facial_data: Dict,
                   eye_data: Dict,
                   speech_data: Dict) -> None:
        """Add new metrics to the current interview."""
        if not self.current_interview_id:
            raise ValueError("No active interview session. Call start_new_interview() first.")
        
        # Calculate overall metrics
        overall_confidence = self._calculate_overall_confidence(
            facial_data, eye_data, speech_data
        )
        
        engagement_score = self._calculate_engagement_score(
            facial_data, eye_data, speech_data
        )
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(
            facial_data, eye_data, speech_data
        )
        
        # Create metrics entry
        metrics = InterviewMetrics(
            timestamp=datetime.now().isoformat(),
            facial_expression=self._compress_facial_data(facial_data),
            eye_tracking=self._compress_eye_data(eye_data),
            speech_analysis=self._compress_speech_data(speech_data),
            overall_confidence=round(overall_confidence, 2),
            engagement_score=round(engagement_score, 2),
            improvement_suggestions=suggestions
        )
        
        self.current_data.append(asdict(metrics))
    
    def save_interview(self) -> str:
        """Save the current interview data to a JSON file."""
        if not self.current_interview_id:
            raise ValueError("No active interview session.")
        
        # Find the next available interview number
        existing_files = [f for f in os.listdir(self.base_dir) 
                         if f.startswith("interview_") and f.endswith(".json")]
        next_number = 1
        if existing_files:
            numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
            next_number = max(numbers) + 1
        
        filename = f"interview_{next_number}.json"
        filepath = os.path.join(self.base_dir, filename)
        
        # Prepare data for Gemini model
        gemini_data = {
            "interview_id": self.current_interview_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": self.current_data,
            "summary": self._generate_summary(),
            "key_insights": self._generate_key_insights(),
            "improvement_areas": self._generate_improvement_areas()
        }
        
        # Save to file with compression
        with open(filepath, 'w') as f:
            json.dump(gemini_data, f, indent=2, default=self._json_serializer)
        
        return filepath
    
    def _compress_facial_data(self, data: Dict) -> Dict:
        """Compress facial expression data."""
        return {
            "emotion": data.get("emotion", "neutral"),
            "confidence": round(data.get("confidence", 0.0), 2)
        }
    
    def _compress_eye_data(self, data: Dict) -> Dict:
        """Compress eye tracking data."""
        return {
            "attention_state": data.get("attention_state", "unknown"),
            "confidence": round(data.get("confidence_score", 0.0), 2)
        }
    
    def _compress_speech_data(self, data: Dict) -> Dict:
        """Compress speech analysis data."""
        metrics = data.get("metrics", {})
        return {
            "wpm": round(metrics.get("wpm", 0.0), 1),
            "filler_count": metrics.get("filler_count", 0),
            "confidence": round(metrics.get("confidence_score", 0.0), 2),
            "sentiment": data.get("analysis", {}).get("sentiment", "neutral"),
            "emotion": data.get("analysis", {}).get("emotion", "neutral")
        }
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types and other special cases."""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        elif isinstance(obj, (datetime)):
            return obj.isoformat()
        elif isinstance(obj, (type(None))):
            return None
        raise TypeError(f"Type {type(obj)} not serializable")
    
    def _calculate_overall_confidence(self,
                                   facial_data: Dict,
                                   eye_data: Dict,
                                   speech_data: Dict) -> float:
        """Calculate overall confidence score from all metrics."""
        # Weighted average of different confidence scores
        facial_conf = facial_data.get("confidence_score", 0)
        eye_conf = eye_data.get("confidence_score", 0)
        speech_conf = speech_data.get("metrics", {}).get("confidence_score", 0)
        
        return (facial_conf * 0.3 + eye_conf * 0.3 + speech_conf * 0.4)
    
    def _calculate_engagement_score(self,
                                  facial_data: Dict,
                                  eye_data: Dict,
                                  speech_data: Dict) -> float:
        """Calculate overall engagement score."""
        # Combine various engagement indicators
        facial_engagement = 1.0 if facial_data.get("emotion") in ["happy", "neutral"] else 0.5
        eye_engagement = 1.0 if eye_data.get("attention_state") == "Focused" else 0.5
        speech_engagement = speech_data.get("metrics", {}).get("clarity_score", 0)
        
        return (facial_engagement * 0.3 + eye_engagement * 0.3 + speech_engagement * 0.4)
    
    def _generate_improvement_suggestions(self,
                                       facial_data: Dict,
                                       eye_data: Dict,
                                       speech_data: Dict) -> List[str]:
        """Generate improvement suggestions based on metrics."""
        suggestions = []
        
        # Facial expression suggestions
        if facial_data.get("confidence_score", 0) < 0.5:
            suggestions.append("Try to maintain more consistent facial expressions")
        
        # Eye tracking suggestions
        if eye_data.get("attention_state") == "Losing Attention":
            suggestions.append("Maintain better eye contact and focus")
        elif eye_data.get("attention_state") == "Stuck":
            suggestions.append("Try to be more dynamic with your eye movements")
        
        # Speech suggestions
        speech_metrics = speech_data.get("metrics", {})
        if speech_metrics.get("filler_count", 0) > 5:
            suggestions.append("Reduce the use of filler words")
        if speech_metrics.get("pause_count", 0) > 5:
            suggestions.append("Work on reducing unnecessary pauses")
        if speech_metrics.get("clarity_score", 0) < 0.5:
            suggestions.append("Focus on speaking more clearly")
        
        return suggestions
    
    def _generate_summary(self) -> Dict:
        """Generate a summary of the interview for Gemini model."""
        if not self.current_data:
            return {}
            
        # Calculate averages
        avg_confidence = sum(m["overall_confidence"] for m in self.current_data) / len(self.current_data)
        avg_engagement = sum(m["engagement_score"] for m in self.current_data) / len(self.current_data)
        
        # Get most common states
        facial_states = [m["facial_expression"]["emotion"] for m in self.current_data]
        eye_states = [m["eye_tracking"]["attention_state"] for m in self.current_data]
        
        return {
            "duration_seconds": len(self.current_data),
            "average_confidence": round(avg_confidence, 2),
            "average_engagement": round(avg_engagement, 2),
            "most_common_facial_state": max(set(facial_states), key=facial_states.count),
            "most_common_eye_state": max(set(eye_states), key=eye_states.count),
            "total_improvement_suggestions": len(set(
                suggestion 
                for m in self.current_data 
                for suggestion in m["improvement_suggestions"]
            ))
        }
    
    def _generate_key_insights(self) -> List[str]:
        """Generate key insights from the interview data."""
        if not self.current_data:
            return []
            
        insights = []
        
        # Analyze trends
        confidence_trend = self._analyze_trend("overall_confidence")
        engagement_trend = self._analyze_trend("engagement_score")
        
        if confidence_trend == "increasing":
            insights.append("Confidence improved throughout the interview")
        elif confidence_trend == "decreasing":
            insights.append("Confidence decreased as the interview progressed")
            
        if engagement_trend == "increasing":
            insights.append("Engagement level increased over time")
        elif engagement_trend == "decreasing":
            insights.append("Engagement level decreased over time")
        
        return insights
    
    def _generate_improvement_areas(self) -> Dict:
        """Generate structured improvement areas."""
        if not self.current_data:
            return {
                "facial_expressions": [],
                "eye_contact": [],
                "speech": []
            }
            
        try:
            areas = {
                "facial_expressions": [],
                "eye_contact": [],
                "speech": []
            }
            
            # Analyze facial expressions
            facial_states = [m["facial_expression"]["emotion"] for m in self.current_data]
            if "unknown" in facial_states:
                areas["facial_expressions"].append("Improve facial expression detection")
            if "angry" in facial_states or "sad" in facial_states:
                areas["facial_expressions"].append("Maintain more positive expressions")
            
            # Analyze eye contact
            eye_states = [m["eye_tracking"]["attention_state"] for m in self.current_data]
            if "Losing Attention" in eye_states:
                areas["eye_contact"].append("Maintain better eye contact")
            if "Stuck" in eye_states:
                areas["eye_contact"].append("Be more dynamic with eye movements")
            
            # Analyze speech
            speech_metrics = [m["speech_analysis"] for m in self.current_data]
            avg_wpm = sum(m.get("wpm", 0) for m in speech_metrics) / len(speech_metrics)
            if avg_wpm < 100:
                areas["speech"].append("Speak at a more natural pace")
            elif avg_wpm > 200:
                areas["speech"].append("Slow down speech rate")
            
            return areas
        except Exception:
            return {
                "facial_expressions": [],
                "eye_contact": [],
                "speech": []
            }
    
    def _analyze_trend(self, metric: str) -> str:
        """Analyze trend of a metric over time."""
        if not self.current_data or len(self.current_data) < 2:
            return "stable"
        
        try:
            values = [m[metric] for m in self.current_data]
            if not values:
                return "stable"
            
            # Calculate simple linear regression
            x = np.arange(len(values))
            y = np.array(values)
            slope = np.polyfit(x, y, 1)[0]
            
            if slope > 0.1:
                return "increasing"
            elif slope < -0.1:
                return "decreasing"
            else:
                return "stable"
        except Exception:
            return "stable"
