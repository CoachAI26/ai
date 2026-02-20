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
    
    # Extract duration from response (support both object-like and dict-like)
    duration_seconds = 0.0

    # Try duration field directly
    duration = getattr(transcription, "duration", None)
    if duration is None and isinstance(transcription, dict):
        duration = transcription.get("duration")

    # Try segments if duration is missing or zero
    segments = getattr(transcription, "segments", None)
    if segments is None and isinstance(transcription, dict):
        segments = transcription.get("segments")

    if isinstance(duration, (int, float)) and duration > 0:
        duration_seconds = float(duration)
    elif isinstance(segments, list) and segments:
        try:
            # segments may be dicts or objects; use 'end' field
            ends = []
            for s in segments:
                if isinstance(s, dict) and "end" in s:
                    ends.append(float(s["end"]))
                elif hasattr(s, "end"):
                    ends.append(float(s.end))
            if ends:
                duration_seconds = max(ends)
        except Exception:
            duration_seconds = 0.0

    # Extract segments for pause analysis
    segments = None
    if hasattr(transcription, 'segments') and transcription.segments:
        segments = transcription.segments
    elif isinstance(transcription, dict) and 'segments' in transcription:
        segments = transcription['segments']

    # Detected language (e.g. "english", "french") for English-only enforcement
    detected_language = getattr(transcription, "language", None) or (isinstance(transcription, dict) and transcription.get("language")) or None
    
    return {
        "text": transcription.text,
        "duration_seconds": duration_seconds,
        "segments": segments,  # Include segments for pause analysis
        "language": detected_language,
    }

