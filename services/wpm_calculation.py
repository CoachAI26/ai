"""
WPM (Words Per Minute) calculation service
"""
import re
from typing import Dict, Any


def count_words(text: str) -> int:
    """
    Count words in text. Uses whitespace tokenization so contractions (I'd, wasn't)
    count as one word and the result matches typical "word count" expectations.
    """
    if not text or not isinstance(text, str):
        return 0
    # Normalize: collapse all whitespace (space, newline, tab, Unicode) to single space
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return 0
    # Split by space; count non-empty tokens (each token = one "word")
    tokens = [t for t in normalized.split(" ") if t]
    return len(tokens)


def calculate_wpm(text: str, duration_seconds: float) -> Dict[str, Any]:
    """
    Calculate Words Per Minute (WPM) based on text and duration
    
    Args:
        text: Transcribed text
        duration_seconds: Duration of speech in seconds
        
    Returns:
        Dictionary with:
        - word_count: Number of words in text
        - duration_seconds: Duration in seconds
        - wpm: Words per minute (0 if duration is 0)
    """
    word_count = count_words(text)
    
    # Calculate WPM: (words / seconds) * 60
    if duration_seconds > 0:
        wpm = (word_count / duration_seconds) * 60
    else:
        wpm = 0.0
    
    return {
        "word_count": word_count,
        "duration_seconds": round(duration_seconds, 2),
        "wpm": round(wpm, 2)
    }

