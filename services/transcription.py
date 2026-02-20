"""
Audio transcription service using OpenAI Whisper
"""
from typing import Dict, Any
from config import (
    get_openai_client,
    WHISPER_MODEL,
    WHISPER_PROMPT,
)


async def transcribe_audio_file(audio_file_path: str) -> Dict[str, Any]:
    """
    Transcribe audio file to text using OpenAI Whisper
    Uses prompt to preserve filler words like "um", "uh", "ah", etc.
    
    Args:
        audio_file_path: Path to the audio file
        
    Returns:
        Dictionary with:
        - text: Transcribed text as string
        - duration_seconds: Duration of the audio in seconds
    """
    client = get_openai_client()
    
    # Use prompt to guide Whisper to preserve filler words and disfluencies
    # Also use temperature=0.2 to make it more literal and preserve all sounds
    # Use verbose_json to get duration information
    # Do not pass language so Whisper detects it; we then enforce English-only
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
            prompt=WHISPER_PROMPT,  # Guide Whisper to preserve filler words
            temperature=0.2,  # Lower temperature for more literal transcription
            response_format="verbose_json"  # Get duration and segments (includes language)
        )
    
    # Extract segments first (needed for both duration and pause analysis)
    segments = getattr(transcription, "segments", None)
    if segments is None and isinstance(transcription, dict):
        segments = transcription.get("segments")

    # Duration for WPM: use actual speaking time from segments (sum of segment lengths)
    # so WPM = words / (speaking_seconds/60). Fall back to file duration if no segments.
    duration_seconds = 0.0
    if isinstance(segments, list) and segments:
        try:
            speaking_total = 0.0
            for s in segments:
                if isinstance(s, dict):
                    start = float(s.get("start") or 0)
                    end = float(s.get("end") or 0)
                else:
                    start = float(getattr(s, "start", 0) or 0)
                    end = float(getattr(s, "end", 0) or 0)
                speaking_total += max(0, end - start)
            if speaking_total > 0:
                duration_seconds = speaking_total
        except Exception:
            pass
    if duration_seconds <= 0:
        duration = getattr(transcription, "duration", None) or (isinstance(transcription, dict) and transcription.get("duration"))
        if isinstance(duration, (int, float)) and duration > 0:
            duration_seconds = float(duration)
        elif isinstance(segments, list) and segments:
            try:
                ends = []
                for s in segments:
                    if isinstance(s, dict) and "end" in s:
                        ends.append(float(s["end"]))
                    elif hasattr(s, "end"):
                        ends.append(float(s.end))
                if ends:
                    duration_seconds = max(ends)
            except Exception:
                pass

    # Detected language (e.g. "english", "french") for English-only enforcement
    detected_language = getattr(transcription, "language", None) or (isinstance(transcription, dict) and transcription.get("language")) or None
    
    return {
        "text": transcription.text,
        "duration_seconds": duration_seconds,
        "segments": segments,  # Include segments for pause analysis
        "language": detected_language,
    }

