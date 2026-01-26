"""
Pydantic schemas for API
"""
from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class TranscriptionResponse(BaseModel):
    """Response model for transcription endpoint"""
    text: str
    filler_words: List[Dict[str, Any]]
    filler_count: int
    cleaned_text: str
    duration_seconds: float
    word_count: int
    wpm: float
    # Pause and hesitation analysis
    total_pauses: int
    total_hesitations: int
    pause_durations: List[float]
    average_pause_duration: float
    total_pause_time: float
    hesitation_words: List[str]
    # Fluency metrics
    fluency_score: float
    pause_ratio: float
    hesitation_rate: float
    # Confidence metrics
    confidence_score: float
    wpm_score: float
    filler_score: float
    pause_score: float
    hesitation_score: float
    overall_rating: str
    recommendations: List[str]

