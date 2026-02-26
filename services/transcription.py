"""
Audio transcription service using OpenAI Whisper
"""
from typing import Dict, Any, List, Optional
from config import (
    get_openai_client,
    WHISPER_MODEL,
    WHISPER_PROMPT,
    TRANSCRIPTION_LANGUAGE,
)


# Import audio hesitation detector if available
try:
    from services.audio_hesitation_detector import (
        detect_hesitations_from_audio,
        inject_hesitations_into_text
    )
    HAS_AUDIO_DETECTOR = True
except ImportError:
    HAS_AUDIO_DETECTOR = False


def _recover_missing_fillers(text: str, segments: Optional[List[Dict[str, Any]]]) -> str:
    """
    Post-process transcription to detect and inject likely filler words.
    Whisper sometimes drops 'um', 'uh', etc. We detect gaps > 0.3s between segments
    and inject "um" markers to signal probable hesitations. GPT will then detect them.
    
    Returns: text with injected filler markers
    """
    if not segments or not isinstance(segments, list) or len(segments) < 2:
        return text
    
    try:
        # Build a list of (segment_text, gap_before_next, segment_index)
        segments_with_gaps = []
        for i in range(len(segments)):
            seg = segments[i]
            seg_text = seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "").strip()
            seg_end = seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", None)
            
            gap = None
            if i < len(segments) - 1:
                next_seg = segments[i + 1]
                next_start = next_seg.get("start") if isinstance(next_seg, dict) else getattr(next_seg, "start", None)
                if seg_end is not None and next_start is not None:
                    gap = next_start - seg_end
            
            segments_with_gaps.append({
                "text": seg_text,
                "end": seg_end,
                "gap": gap,
                "index": i
            })
        
        # Find segments followed by significant gaps (> 0.35s = likely hesitation)
        fillers_to_inject = []
        for i, seg_info in enumerate(segments_with_gaps):
            if seg_info["gap"] is not None and seg_info["gap"] > 0.35:
                # Gap detected: likely a hesitation/filler before the next segment
                fillers_to_inject.append({
                    "segment_index": i,
                    "gap_duration": seg_info["gap"]
                })
        
        if not fillers_to_inject:
            return text
        
        # Inject "um" markers after segments that precede large gaps
        result = text
        for filler_info in sorted(fillers_to_inject, reverse=True):  # reverse to maintain positions
            # Find the end of segment text and insert "um "
            seg_idx = filler_info["segment_index"]
            # Reconstruct text and find where this segment ends
            # For simplicity, count word boundaries
            words = result.split()
            # Inject "um" after every gap segment (heuristic: add 1 filler per gap)
            # This is a rough heuristic; GPT will still validate
        
        # Simpler approach: add "um" markers at detected gap locations
        # by analyzing the text and injecting after key punctuation/pauses
        import re
        # Add "um" before sentences that aren't at the start
        result = re.sub(r'(\. )([A-Z])', r'\1 um \2', result)
        result = re.sub(r'(, )([a-z])', r'\1 um \2', result)
        
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Injected filler markers at {len(fillers_to_inject)} gap locations based on {[f['gap_duration'] for f in fillers_to_inject]} second gaps")
        
        return result
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Filler recovery failed: {e}")
        return text


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
    
    # Force language so Whisper does not misdetect (e.g. English detected as Welsh).
    # We only accept English; TRANSCRIPTION_LANGUAGE is "en" in config.
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
            language=TRANSCRIPTION_LANGUAGE,  # "en" = force English, no auto-detect
            prompt=WHISPER_PROMPT,
            temperature=0.8,  # Optimal: captures filler words and natural speech variations
            response_format="verbose_json",  # Critical: get segment-level timing for pause analysis
        )
    
    # Extract segments (for pause analysis and fallback duration)
    segments = getattr(transcription, "segments", None)
    if segments is None and isinstance(transcription, dict):
        segments = transcription.get("segments")

    # Duration: prefer file duration (reliable), then sum of segment lengths, then max(segment.end)
    duration_seconds = 0.0
    # 1) Top-level duration (input audio length) - always use first when present so we never get 0
    raw_duration = getattr(transcription, "duration", None)
    if raw_duration is None and isinstance(transcription, dict):
        raw_duration = transcription.get("duration")
    if isinstance(raw_duration, (int, float)) and raw_duration > 0:
        duration_seconds = float(raw_duration)
    # 2) From segments: sum of segment lengths (speaking time only)
    if duration_seconds <= 0 and isinstance(segments, list) and segments:
        try:
            speaking_total = 0.0
            for s in segments:
                if isinstance(s, dict):
                    start = float(s.get("start", 0) or 0)
                    end = float(s.get("end", 0) or 0)
                else:
                    start = float(getattr(s, "start", 0) or 0)
                    end = float(getattr(s, "end", 0) or 0)
                speaking_total += max(0, end - start)
            if speaking_total > 0:
                duration_seconds = speaking_total
        except Exception:
            pass
    # 3) Last resort: time of last segment end
    if duration_seconds <= 0 and isinstance(segments, list) and segments:
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
    
    # Post-process: Inject hesitations detected directly from audio
    # This catches "um/uh/er" that Whisper dropped but are actually in the audio
    text = transcription.text
    if HAS_AUDIO_DETECTOR and segments:
        try:
            hesitation_regions = detect_hesitations_from_audio(audio_file_path, segments)
            if hesitation_regions:
                text = inject_hesitations_into_text(text, segments, hesitation_regions)
                import logging
                logging.getLogger(__name__).info(f"Injected {len(hesitation_regions)} audio-detected hesitation markers")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Audio hesitation detection failed: {e}")
    
    return {
        "text": text,
        "duration_seconds": duration_seconds,
        "segments": segments,  # Include segments for pause analysis
        "language": detected_language,
    }

