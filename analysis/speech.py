import torch
import whisper
import librosa
import numpy as np
from transformers import pipeline
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import soundfile as sf
import io
from collections import Counter
import re
import wave

@dataclass
class SpeechMetrics:
    wpm: float  # Words per minute
    avg_pitch: float  # Average pitch in Hz
    pitch_variation: float  # Standard deviation of pitch
    filler_count: int  # Number of filler words
    confidence_score: float  # Confidence in speech (0-1)
    sentiment: str  # Positive, Neutral, or Negative
    transcript: str  # Full transcription
    pause_count: int  # Number of significant pauses
    avg_pause_duration: float  # Average duration of pauses
    clarity_score: float  # Clarity of speech (0-1)
    emotion: str  # Detected emotion
    complexity_score: float  # Vocabulary complexity (0-1)

class SpeechAnalyzer:
    def __init__(self):
        # Initialize Whisper model for transcription
        self.model = whisper.load_model("base")
        
        # Initialize sentiment and emotion analyzers
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="roberta-base",
            tokenizer="roberta-base"
        )
        
        self.emotion_analyzer = pipeline(
            "text-classification",
            model="finiteautomata/bertweet-base-emotion-analysis",
            tokenizer="finiteautomata/bertweet-base-emotion-analysis"
        )
        
        # Define filler words to track (expanded list)
        self.filler_words = ["um", "uh", "like", "you know", "so", "well", "actually", "basically", "literally", "honestly", "I think", "essentially", "kinda", "more or less"]
        
        # Define pause threshold (in seconds)
        self.pause_threshold = 0.3
        
    def analyze_audio(self, audio_data: bytes) -> Dict[str, Any]:
        try:
            # Convert bytes to numpy array
            audio_array, sr = self._bytes_to_audio(audio_data)
            
            if len(audio_array) == 0:
                return self._get_default_metrics()
            
            # Get transcript
            transcript = self._transcribe(audio_array)
            
            # Calculate metrics
            wpm = self._calculate_wpm(transcript)
            avg_pitch = self._calculate_pitch(audio_array, sr)
            filler_count = self._count_fillers(transcript)
            confidence_score = self._calculate_confidence(transcript)
            sentiment = self._analyze_sentiment(transcript)
            pitch_variation = self._calculate_pitch_variation(audio_array, sr)
            pause_count, avg_pause_duration = self._analyze_pauses(audio_array, sr)
            clarity_score = self._calculate_clarity(audio_array, sr)
            emotion = self._analyze_emotion(transcript)
            complexity_score = self._calculate_complexity(transcript)
            
            metrics = SpeechMetrics(
                wpm=wpm,
                avg_pitch=avg_pitch,
                pitch_variation=pitch_variation,
                filler_count=filler_count,
                confidence_score=confidence_score,
                sentiment=sentiment,
                transcript=transcript,
                pause_count=pause_count,
                avg_pause_duration=avg_pause_duration,
                clarity_score=clarity_score,
                emotion=emotion,
                complexity_score=complexity_score
            )
            
            return self._format_results(metrics)
            
        except Exception as e:
            print(f"Error analyzing audio: {str(e)}")
            return self._get_default_metrics()
    
    def _bytes_to_audio(self, audio_data: bytes) -> tuple[np.ndarray, int]:
        """Convert raw audio bytes to numpy array."""
        try:
            # Create a BytesIO object
            audio_io = io.BytesIO(audio_data)
            
            # Read the WAV data
            with wave.open(audio_io, 'rb') as wav_file:
                # Get audio parameters
                n_channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frame_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                
                # Read audio data
                audio_bytes = wav_file.readframes(n_frames)
                
                # Convert to numpy array
                dtype = np.int16 if sample_width == 2 else np.int8
                audio_array = np.frombuffer(audio_bytes, dtype=dtype)
                
                # Reshape if stereo
                if n_channels > 1:
                    audio_array = audio_array.reshape(-1, n_channels)
                    # Convert to mono by averaging channels
                    audio_array = np.mean(audio_array, axis=1)
                
                return audio_array, frame_rate
                
        except Exception as e:
            print(f"Error converting bytes to audio: {str(e)}")
            # Return empty array with default sample rate
            return np.array([], dtype=np.float32), 16000
    
    def _transcribe(self, audio_array: np.ndarray) -> str:
        """Transcribe audio using Whisper."""
        result = self.model.transcribe(audio_array)
        return result["text"]
    
    def _calculate_wpm(self, transcript: str) -> float:
        """Calculate words per minute."""
        if not transcript or len(transcript.strip()) == 0:
            return 0.0
        
        words = len(transcript.split())
        # Assuming average word length of 5 characters and average speaking rate
        # This is a more stable calculation than using transcript length
        return min(words * 2, 200)  # Cap at 200 WPM
    
    def _calculate_pitch(self, audio_array: np.ndarray, sr: int) -> float:
        """Calculate average pitch using librosa."""
        pitches, magnitudes = librosa.piptrack(y=audio_array, sr=sr)
        valid_pitches = pitches[magnitudes > np.max(magnitudes) * 0.1]
        
        if len(valid_pitches) == 0:
            return 0.0
            
        return float(np.mean(valid_pitches))
    
    def _calculate_pitch_variation(self, audio_array: np.ndarray, sr: int) -> float:
        """Calculate pitch variation using librosa."""
        pitches, magnitudes = librosa.piptrack(y=audio_array, sr=sr)
        valid_pitches = pitches[magnitudes > np.max(magnitudes) * 0.1]
        
        if len(valid_pitches) == 0:
            return 0.0
            
        return float(np.std(valid_pitches))
    
    def _analyze_pauses(self, audio_array: np.ndarray, sr: int) -> tuple[int, float]:
        """Analyze pauses in speech."""
        # Calculate energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        energy = librosa.feature.rms(y=audio_array, 
                                   frame_length=frame_length,
                                   hop_length=hop_length)[0]
        
        # Find pauses
        threshold = np.mean(energy) * 0.1
        pauses = energy < threshold
        
        # Calculate pause metrics
        pause_durations = []
        current_pause = 0
        
        for is_pause in pauses:
            if is_pause:
                current_pause += 1
            elif current_pause > 0:
                pause_duration = current_pause * hop_length / sr
                if pause_duration >= self.pause_threshold:
                    pause_durations.append(pause_duration)
                current_pause = 0
                
        return (len(pause_durations), np.mean(pause_durations) if pause_durations else 0.0)
    
    def _count_fillers(self, transcript: str) -> int:
        """Count filler words in transcript."""
        transcript_lower = transcript.lower()
        return sum(transcript_lower.count(filler) for filler in self.filler_words)
    
    def _calculate_confidence(self, transcript: str) -> float:
        """Calculate confidence score based on various factors."""
        words = transcript.split()
        if not words:
            return 0.0
            
        # Calculate base confidence from filler words
        filler_count = self._count_fillers(transcript)
        base_confidence = 1.0 - (filler_count / len(words))
        
        # Adjust confidence based on sentence structure
        sentences = re.split(r'[.!?]+', transcript)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        structure_score = min(1.0, avg_sentence_length / 15)  # Normalize to max 15 words per sentence
        
        # Combine scores
        confidence = (base_confidence * 0.7) + (structure_score * 0.3)
        return max(0.0, min(1.0, confidence))
    
    def _analyze_sentiment(self, transcript: str) -> str:
        """Analyze sentiment using RoBERTa."""
        if not transcript or len(transcript.strip()) == 0:
            return "neutral"
        
        try:
            result = self.sentiment_analyzer(transcript)[0]
            return result["label"]
        except Exception:
            return "neutral"
    
    def _analyze_emotion(self, transcript: str) -> str:
        """Analyze emotion using BERTweet."""
        if not transcript or len(transcript.strip()) == 0:
            return "neutral"
        
        try:
            result = self.emotion_analyzer(transcript)[0]
            return result["label"]
        except Exception:
            return "neutral"
    
    def _calculate_clarity(self, audio_array: np.ndarray, sr: int) -> float:
        """Calculate speech clarity score."""
        try:
            transcript = self._transcribe(audio_array)
            words = transcript.split()
            if not words:
                return 0.0
            
            # Count unique words
            unique_words = len(set(words))
            
            # Calculate word frequency
            word_freq = Counter(words)
            common_words = sum(1 for word, count in word_freq.items() if count > 2)
            
            # Calculate clarity score
            clarity = (unique_words / len(words)) * (1 - (common_words / len(words)))
            return max(0.0, min(1.0, clarity))
        except Exception:
            return 0.0
    
    def _calculate_complexity(self, transcript: str) -> float:
        """Calculate vocabulary complexity score."""
        if not transcript or len(transcript.strip()) == 0:
            return 0.0
        
        try:
            words = transcript.split()
            if not words:
                return 0.0
            
            # Count syllables
            syllable_count = sum(self._count_syllables(word) for word in words)
            
            # Calculate average word length and syllables per word
            avg_word_length = sum(len(word) for word in words) / len(words)
            avg_syllables = syllable_count / len(words)
            
            # Calculate complexity score
            complexity = (avg_word_length / 10) * 0.4 + (avg_syllables / 3) * 0.6
            return max(0.0, min(1.0, complexity))
        except Exception:
            return 0.0
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word."""
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index-1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count

    def _get_default_metrics(self) -> Dict[str, Any]:
        """Return default metrics when analysis fails."""
        return {
            "metrics": {
                "wpm": 0.0,
                "avg_pitch": 0.0,
                "filler_count": 0,
                "confidence_score": 0.0,
                "pitch_variation": 0.0,
                "pause_count": 0,
                "avg_pause_duration": 0.0,
                "clarity_score": 0.0,
                "complexity_score": 0.0
            },
            "analysis": {
                "speech_quality": "Poor",
                "pace": "Unknown",
                "filler_usage": "Unknown",
                "sentiment": "Neutral",
                "emotion": "Neutral"
            },
            "transcript": ""
        }

    def _format_results(self, metrics: SpeechMetrics) -> Dict[str, Any]:
        """Format the results as a dictionary."""
        return {
            "transcript": metrics.transcript,
            "metrics": {
                "wpm": round(metrics.wpm, 2),
                "avg_pitch": round(metrics.avg_pitch, 2),
                "pitch_variation": round(metrics.pitch_variation, 2),
                "filler_count": metrics.filler_count,
                "pause_count": metrics.pause_count,
                "avg_pause_duration": round(metrics.avg_pause_duration, 2),
                "confidence_score": round(metrics.confidence_score, 2),
                "clarity_score": round(metrics.clarity_score, 2),
                "complexity_score": round(metrics.complexity_score, 2)
            },
            "analysis": {
                "speech_quality": "Confident" if metrics.confidence_score > 0.7 else 
                                "Moderate" if metrics.confidence_score > 0.4 else 
                                "Needs Improvement",
                "pace": "Fast" if metrics.wpm > 180 else 
                       "Moderate" if metrics.wpm > 120 else 
                       "Slow",
                "filler_usage": "Low" if metrics.filler_count < 3 else 
                              "Moderate" if metrics.filler_count < 7 else 
                              "High",
                "pausing": "Natural" if 2 <= metrics.pause_count <= 5 else 
                          "Too Many" if metrics.pause_count > 5 else 
                          "Too Few",
                "clarity": "Clear" if metrics.clarity_score > 0.7 else 
                          "Moderate" if metrics.clarity_score > 0.4 else 
                          "Unclear",
                "complexity": "Complex" if metrics.complexity_score > 0.7 else 
                             "Moderate" if metrics.complexity_score > 0.4 else 
                             "Simple"
            },
            "sentiment": metrics.sentiment,
            "emotion": metrics.emotion
        }

def analyze_speech(audio_data: bytes) -> Dict:
    """Main function to analyze speech from audio data."""
    analyzer = SpeechAnalyzer()
    results = analyzer.analyze_audio(audio_data)
    
    return results

if __name__ == "__main__":
    # Example usage
    with open("test_audio.wav", "rb") as f:
        audio_data = f.read()
    
    results = analyze_speech(audio_data)
    print("Speech Analysis Results:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"{key}: {value}")
