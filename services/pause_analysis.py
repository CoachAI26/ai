"""
Pause and Hesitation analysis service
"""
from typing import List, Dict, Any, Optional
import re


def analyze_pauses_and_hesitations(
    text: str,
    segments: Optional[List[Dict[str, Any]]] = None,
    pause_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Analyze pauses and hesitations in transcribed speech
    
    Args:
        text: Transcribed text
        segments: List of segments from Whisper with timestamps (optional)
        pause_threshold: Minimum duration in seconds to consider as a pause (default: 0.5s)
        
    Returns:
        Dictionary with:
        - total_pauses: Total number of pauses detected
        - total_hesitations: Total number of hesitation sounds (um, uh, etc.)
        - pause_durations: List of pause durations in seconds
        - average_pause_duration: Average pause duration in seconds
        - total_pause_time: Total time spent in pauses
        - hesitation_words: List of detected hesitation words
    """
    # Detect hesitation sounds in text
    hesitation_pattern = re.compile(r'\b(um+|uh+|er+|erm+|ah+|hmm+)\b', re.IGNORECASE)
    hesitation_matches = list(hesitation_pattern.finditer(text))
    hesitation_words = [match.group(0) for match in hesitation_matches]
    
    # Analyze pauses from segments if available
    pause_durations = []
    total_pause_time = 0.0
    
    if segments and len(segments) > 1:
        # Calculate pauses between segments
        for i in range(len(segments) - 1):
            current_segment = segments[i]
            next_segment = segments[i + 1]
            
            # Get end time of current segment and start time of next segment
            current_end = _get_segment_time(current_segment, 'end')
            next_start = _get_segment_time(next_segment, 'start')
            
            if current_end is not None and next_start is not None:
                pause_duration = next_start - current_end
                if pause_duration >= pause_threshold:
                    pause_durations.append(pause_duration)
                    total_pause_time += pause_duration
    
    # Calculate statistics
    total_pauses = len(pause_durations)
    total_hesitations = len(hesitation_words)
    average_pause_duration = (
        sum(pause_durations) / len(pause_durations) 
        if pause_durations else 0.0
    )
    
    return {
        "total_pauses": total_pauses,
        "total_hesitations": total_hesitations,
        "pause_durations": [round(d, 2) for d in pause_durations],
        "average_pause_duration": round(average_pause_duration, 2),
        "total_pause_time": round(total_pause_time, 2),
        "hesitation_words": hesitation_words,
        "pause_threshold_used": pause_threshold
    }


def _get_segment_time(segment: Any, time_type: str) -> Optional[float]:
    """
    Extract start or end time from a segment (handles both dict and object)
    
    Args:
        segment: Segment from Whisper (can be dict or object)
        time_type: 'start' or 'end'
        
    Returns:
        Time in seconds or None if not available
    """
    if isinstance(segment, dict):
        return segment.get(time_type)
    else:
        return getattr(segment, time_type, None)


def calculate_fluency_score(
    total_duration: float,
    total_pause_time: float,
    hesitation_count: int,
    word_count: int
) -> Dict[str, Any]:
    """
    Calculate fluency score based on pauses and hesitations
    
    Args:
        total_duration: Total speaking duration in seconds
        total_pause_time: Total time spent in pauses
        hesitation_count: Number of hesitation sounds
        word_count: Total number of words
        
    Returns:
        Dictionary with fluency metrics:
        - fluency_score: Score from 0-100 (higher is better)
        - pause_ratio: Ratio of pause time to total time
        - hesitation_rate: Hesitations per 100 words
    """
    if total_duration == 0:
        return {
            "fluency_score": 0.0,
            "pause_ratio": 0.0,
            "hesitation_rate": 0.0
        }
    
    # Calculate pause ratio
    pause_ratio = total_pause_time / total_duration if total_duration > 0 else 0.0
    
    # Calculate hesitation rate (per 100 words)
    hesitation_rate = (hesitation_count / word_count * 100) if word_count > 0 else 0.0
    
    # Calculate fluency score (0-100, higher is better)
    # Penalize for pauses and hesitations
    pause_penalty = min(pause_ratio * 50, 50)  # Max 50 points penalty for pauses
    hesitation_penalty = min(hesitation_rate * 0.5, 30)  # Max 30 points penalty for hesitations
    
    fluency_score = max(0, 100 - pause_penalty - hesitation_penalty)
    
    return {
        "fluency_score": round(fluency_score, 2),
        "pause_ratio": round(pause_ratio, 3),
        "hesitation_rate": round(hesitation_rate, 2)
    }

