"""
Pause and Hesitation analysis service.
Hesitation/filler counts come from GPT only (filler_words); no regex.
"""
from typing import List, Dict, Any, Optional

try:
    from scoring_config import PAUSE_THRESHOLD_SEC
except ImportError:
    PAUSE_THRESHOLD_SEC = 0.5


def analyze_pauses_and_hesitations(
    text: str,
    segments: Optional[List[Dict[str, Any]]] = None,
    pause_threshold: Optional[float] = None,
    filler_words: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Analyze pauses (from Whisper segments) and hesitations (from GPT filler_words only).
    
    Args:
        text: Transcribed text (unused when filler_words provided; kept for API compat)
        segments: List of segments from Whisper with timestamps (optional)
        pause_threshold: Minimum duration in seconds to consider as a pause (default: from scoring_config)
        filler_words: List of filler items from GPT (each with "word"); used for total_hesitations and hesitation_words. If None, hesitations are 0.
        
    Returns:
        Dictionary with total_pauses, total_hesitations, pause_durations, etc.
    """
    if filler_words:
        total_hesitations = len(filler_words)
        hesitation_words = [f.get("word", "") for f in filler_words if isinstance(f, dict)]
    else:
        total_hesitations = 0
        hesitation_words = []
    
    threshold = pause_threshold if pause_threshold is not None else PAUSE_THRESHOLD_SEC
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
                if pause_duration >= threshold:
                    pause_durations.append(pause_duration)
                    total_pause_time += pause_duration
    
    # Calculate statistics (total_hesitations already set from filler_words above)
    total_pauses = len(pause_durations)
    average_pause_duration = (
        sum(pause_durations) / len(pause_durations) 
        if pause_durations else 0.0
    )
    
    # Report pause durations in tenths of a second (0.1) for finer detail
    return {
        "total_pauses": total_pauses,
        "total_hesitations": total_hesitations,
        "pause_durations": [round(d, 1) for d in pause_durations],
        "average_pause_duration": round(average_pause_duration, 1),
        "total_pause_time": round(total_pause_time, 1),
        "hesitation_words": hesitation_words,
        "pause_threshold_used": threshold,
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
    word_count: int,
    filler_count: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Calculate fluency score based on pauses and hesitations.
    Hesitation count comes from GPT (filler_words). Uses scoring_config for penalties.
    """
    try:
        from scoring_config import (
            FLUENCY_PAUSE_PENALTY_PER_RATIO,
            FLUENCY_PAUSE_PENALTY_CAP,
            FLUENCY_HESITATION_PENALTY_PER_RATE,
            FLUENCY_HESITATION_PENALTY_CAP,
            USE_FILLER_COUNT_IN_HESITATION,
            FILLER_WEIGHT_IN_HESITATION,
        )
    except ImportError:
        FLUENCY_PAUSE_PENALTY_PER_RATIO = 80
        FLUENCY_PAUSE_PENALTY_CAP = 55
        FLUENCY_HESITATION_PENALTY_PER_RATE = 1.2
        FLUENCY_HESITATION_PENALTY_CAP = 35
        USE_FILLER_COUNT_IN_HESITATION = False
        FILLER_WEIGHT_IN_HESITATION = 0.6

    if total_duration == 0:
        return {
            "fluency_score": 0.0,
            "pause_ratio": 0.0,
            "hesitation_rate": 0.0
        }
    
    pause_ratio = total_pause_time / total_duration if total_duration > 0 else 0.0
    if USE_FILLER_COUNT_IN_HESITATION and filler_count is not None and word_count > 0:
        effective_hesitations = hesitation_count + FILLER_WEIGHT_IN_HESITATION * filler_count
        hesitation_rate = effective_hesitations / word_count * 100
    else:
        hesitation_rate = (hesitation_count / word_count * 100) if word_count > 0 else 0.0
    
    pause_penalty = min(pause_ratio * FLUENCY_PAUSE_PENALTY_PER_RATIO, FLUENCY_PAUSE_PENALTY_CAP)
    hesitation_penalty = min(hesitation_rate * FLUENCY_HESITATION_PENALTY_PER_RATE, FLUENCY_HESITATION_PENALTY_CAP)
    fluency_score = max(0, 100 - pause_penalty - hesitation_penalty)
    
    return {
        "fluency_score": round(fluency_score, 2),
        "pause_ratio": round(pause_ratio, 3),
        "hesitation_rate": round(hesitation_rate, 2)
    }

