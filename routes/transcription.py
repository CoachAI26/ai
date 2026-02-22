"""
Transcription API endpoints
"""
import os
import re
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Form

logger = logging.getLogger(__name__)
from models.schemas import TranscriptionResponse
from services.transcription import transcribe_audio_file
from services.filler_detection import (
    detect_filler_words_with_gpt,
    remove_filler_words,
    generate_improved_text,
    check_answer_relevance_to_title,
    OFF_TOPIC_MESSAGE,
)
from services.wpm_calculation import calculate_wpm
from services.pause_analysis import analyze_pauses_and_hesitations, calculate_fluency_score
from services.confidence_analysis import calculate_confidence_score
# from services.tts import text_to_speech  # TTS disabled: do not return AI voice

# Supported audio formats by OpenAI Whisper
SUPPORTED_AUDIO_FORMATS = {
    'audio/mpeg',  # MP3
    'audio/wav',  # WAV
    'audio/x-m4a',  # M4A
    'audio/mp4',  # M4A alternative
    'audio/ogg',  # OGG
    'audio/flac',  # FLAC
    'audio/webm',  # WebM
    'audio/x-aac',  # AAC
}

router = APIRouter(prefix="/api/v1", tags=["transcription"])


@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Voice Transcription & Filler Word Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/api/v1/transcribe": "POST - Upload audio file for transcription",
            "/api/v1/health": "GET - Health check"
        }
    }


@router.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "transcription-api"}


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    level: str | None = Form(None),
    category: str | None = Form(None),
    title: str | None = Form(None),
):
    """
    Transcribe audio file to text and analyze speech patterns
    
    - **file**: Audio file (MP3, WAV, M4A, OGG, FLAC, WebM, etc.)
    
    Supported formats:
    - MP3 (audio/mpeg)
    - WAV (audio/wav)
    - M4A (audio/x-m4a, audio/mp4)
    - OGG (audio/ogg)
    - FLAC (audio/flac)
    - WebM (audio/webm)
    - AAC (audio/x-aac)
    
    Returns comprehensive speech analysis including:
    - **text**: Full transcribed text
    - **filler_words**: List of detected filler words with positions
    - **filler_count**: Number of filler words found
    - **cleaned_text**: Text with filler words removed
    - **duration_seconds**: Speaking duration
    - **word_count**: Total word count
    - **wpm**: Words per minute
    - **total_pauses**: Number of pauses detected
    - **total_hesitations**: Number of hesitation sounds
    - **pause_durations**: List of pause durations
    - **average_pause_duration**: Average pause duration
    - **total_pause_time**: Total time spent in pauses
    - **hesitation_words**: List of hesitation words found
    - **fluency_score**: Fluency score (0-100)
    - **pause_ratio**: Ratio of pause time to total time
    - **hesitation_rate**: Hesitations per 100 words
    - **confidence_score**: Overall confidence score (0-100)
    - **wpm_score**: WPM component score
    - **filler_score**: Filler words component score
    - **pause_score**: Pause component score
    - **hesitation_score**: Hesitation component score
    - **overall_rating**: Overall rating (Excellent/Good/Moderate/Low/Very Low)
    - **recommendations**: Personalized improvement recommendations
    """
    logger.info("---------- POST /transcribe ----------")
    logger.info(
        "Request | file=%s level=%s category=%s title=%s",
        file.filename or "(no name)",
        level or "-",
        category or "-",
        (title[:50] + "..." if title and len(title) > 50 else title) or "-",
    )
    # Validate file type
    if not file.content_type:
        # Try to validate by file extension if content_type is missing
        if file.filename:
            ext = file.filename.lower().split('.')[-1]
            supported_extensions = {'mp3', 'wav', 'm4a', 'ogg', 'flac', 'webm', 'aac', 'mp4'}
            if ext not in supported_extensions:
                logger.warning("Rejected: unsupported file extension ext=%s", ext)
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format. Supported formats: MP3, WAV, M4A, OGG, FLAC, WebM, AAC"
                )
        else:
            logger.warning("Rejected: file type could not be determined")
            raise HTTPException(
                status_code=400,
                detail="File type could not be determined. Please ensure the file is an audio file."
            )
    elif file.content_type not in SUPPORTED_AUDIO_FORMATS and not file.content_type.startswith('audio/'):
        logger.warning("Rejected: unsupported content_type=%s", file.content_type)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: {file.content_type}. Supported formats: MP3, WAV, M4A, OGG, FLAC, WebM, AAC"
        )
    
    temp_file_path = None
    
    try:
        # Read audio file
        audio_content = await file.read()
        
        # Save temporarily
        temp_file_path = f"/tmp/{file.filename}"
        with open(temp_file_path, "wb") as f:
            f.write(audio_content)
        logger.info("Saved temp file size=%d bytes path=%s", len(audio_content), temp_file_path)

        # Transcribe using OpenAI Whisper (language detected for enforcement)
        transcription_result = await transcribe_audio_file(temp_file_path)
        text = transcription_result["text"]
        duration_seconds = transcription_result["duration_seconds"]
        segments = transcription_result.get("segments")
        detected_language = (transcription_result.get("language") or "").strip().lower()
        text_preview = (text[:80] + "...") if text and len(text) > 80 else (text or "")
        logger.info(
            "Transcribed | language=%s duration=%.2fs len=%d | text_preview=%s",
            detected_language or "(none)",
            duration_seconds,
            len(text or ""),
            text_preview,
        )
        # Full text for debug
        _log_text = (text or "").strip()
        if _log_text:
            logger.info("DEBUG transcribed text [%d chars]: %s", len(_log_text), _log_text[:800] + ("..." if len(_log_text) > 800 else ""))

        # English-only: reject if user did not speak English
        if detected_language and detected_language not in ("en", "english"):
            logger.warning("Rejected: non-English language=%s", detected_language)
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(
                status_code=400,
                detail="Please speak in English. Other languages are not accepted.",
            )

        # No speech or empty transcription -> don't return fake scores
        if not (text and text.strip()):
            logger.warning("Rejected: no speech detected (empty transcription)")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(
                status_code=400,
                detail="No speech detected in the audio. Please try again with a clear recording.",
            )
        if duration_seconds <= 0:
            logger.warning("Rejected: duration=0 (cannot compute WPM)")
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(
                status_code=400,
                detail="Could not determine audio duration. Please try again with a valid recording.",
            )

        # Detect filler words using GPT
        filler_words = await detect_filler_words_with_gpt(text)
        logger.info("Fillers | count=%d words=%s", len(filler_words), [f.get("word") for f in filler_words[:15]])
        # Remove filler words to get cleaned text
        cleaned_text = remove_filler_words(text, filler_words)
        _cleaned_preview = (cleaned_text[:150] + "...") if cleaned_text and len(cleaned_text) > 150 else (cleaned_text or "")
        logger.info("  -> cleaned_text (preview): %s", _cleaned_preview)
        logger.info("  -> cleaned_text (full): %s", cleaned_text or "")
        # Calculate WPM
        wpm_data = calculate_wpm(text, duration_seconds)
        logger.info(
            "WPM | duration=%.2fs word_count=%d wpm=%.2f",
            wpm_data["duration_seconds"],
            wpm_data["word_count"],
            wpm_data["wpm"],
        )
        
        # Analyze pauses and hesitations
        pause_data = analyze_pauses_and_hesitations(
            text=text,
            segments=segments,
            pause_threshold=0.5
        )
        
        # Calculate fluency score
        fluency_data = calculate_fluency_score(
            total_duration=duration_seconds,
            total_pause_time=pause_data["total_pause_time"],
            hesitation_count=pause_data["total_hesitations"],
            word_count=wpm_data["word_count"]
        )
        logger.info(
            "Pause/Fluency | pauses=%d hesitations=%d pause_ratio=%.3f hesitation_rate=%.2f fluency=%.2f",
            pause_data["total_pauses"],
            pause_data["total_hesitations"],
            fluency_data["pause_ratio"],
            fluency_data["hesitation_rate"],
            fluency_data["fluency_score"],
        )
        # Calculate confidence score (async now)
        confidence_data = await calculate_confidence_score(
            wpm=wpm_data["wpm"],
            filler_count=len(filler_words),
            word_count=wpm_data["word_count"],
            total_pauses=pause_data["total_pauses"],
            total_hesitations=pause_data["total_hesitations"],
            pause_ratio=fluency_data["pause_ratio"],
            hesitation_rate=fluency_data["hesitation_rate"],
            fluency_score=fluency_data["fluency_score"],
            level=level,
            category=category,
            title=title,
        )

        # Check if answer is relevant to the challenge title (when title is provided)
        off_topic = False
        if title and title.strip():
            is_relevant = await check_answer_relevance_to_title(title.strip(), text)
            if not is_relevant:
                off_topic = True
            logger.info("Relevance check | title=%s relevant=%s off_topic=%s", title[:40], is_relevant, off_topic)

        if off_topic:
            improved_text = OFF_TOPIC_MESSAGE
            logger.info("Off-topic -> improved_text: [fixed message]")
            # Lower scores when off-topic
            penalty = 0.5
            confidence_data = {
                **confidence_data,
                "confidence_score": round(min(confidence_data["confidence_score"] * penalty, 40.0), 2),
                "wpm_score": round(min(confidence_data["wpm_score"] * penalty, 50.0), 2),
                "filler_score": round(min(confidence_data["filler_score"] * penalty, 50.0), 2),
                "pause_score": round(min(confidence_data["pause_score"] * penalty, 50.0), 2),
                "hesitation_score": round(min(confidence_data["hesitation_score"] * penalty, 50.0), 2),
                "overall_rating": "Low",
                "recommendations": [
                    f"Try to address the challenge topic: \"{title}\". Speak about the question or key points related to it instead of going off-topic.",
                    "Your response was not related to the given challenge. Next time, stay on topic to get a proper score and feedback.",
                ],
            }
        else:
            # Generate improved text only when on-topic
            improved_text = await generate_improved_text(
                cleaned_text,
                level=level,
                category=category,
                title=title,
            )
            _imp_preview = (improved_text[:200] + "...") if improved_text and len(improved_text) > 200 else (improved_text or "")
            logger.info("Improved text (preview): %s", _imp_preview)

        # TTS disabled: do not return AI voice
        # tts_result = await text_to_speech(improved_text)
        tts_result = None

        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        logger.info("---------- RESPONSE (returned to client) ----------")
        logger.info(
            "Response | confidence=%.2f rating=%s wpm=%.2f words=%d filler_count=%d "
            "wpm_score=%.1f filler_score=%.1f pause_score=%.1f hesitation_score=%.1f",
            confidence_data["confidence_score"],
            confidence_data["overall_rating"],
            wpm_data["wpm"],
            wpm_data["word_count"],
            len(filler_words),
            confidence_data["wpm_score"],
            confidence_data["filler_score"],
            confidence_data["pause_score"],
            confidence_data["hesitation_score"],
        )
        logger.info("  -> text (returned, full): %s", text)
        logger.info("  -> improved_text (returned, full): %s", improved_text)
        if confidence_data.get("recommendations"):
            for i, rec in enumerate(confidence_data["recommendations"][:5], 1):
                logger.info("  -> recommendation[%d]: %s", i, rec[:120] + ("..." if len(rec) > 120 else ""))
        return TranscriptionResponse(
            text=text,
            improved_text=improved_text,  # Add improved text to response
            tts_speech=tts_result,        # Add TTS audio data
            level=level,
            category=category,
            title=title,
            filler_words=filler_words,
            filler_count=len(filler_words),
            cleaned_text=cleaned_text,
            duration_seconds=wpm_data["duration_seconds"],
            word_count=wpm_data["word_count"],
            wpm=wpm_data["wpm"],
            total_pauses=pause_data["total_pauses"],
            total_hesitations=pause_data["total_hesitations"],
            pause_durations=pause_data["pause_durations"],
            average_pause_duration=pause_data["average_pause_duration"],
            total_pause_time=pause_data["total_pause_time"],
            hesitation_words=pause_data["hesitation_words"],
            fluency_score=fluency_data["fluency_score"],
            pause_ratio=fluency_data["pause_ratio"],
            hesitation_rate=fluency_data["hesitation_rate"],
            confidence_score=confidence_data["confidence_score"],
            wpm_score=confidence_data["wpm_score"],
            filler_score=confidence_data["filler_score"],
            pause_score=confidence_data["pause_score"],
            hesitation_score=confidence_data["hesitation_score"],
            overall_rating=confidence_data["overall_rating"],
            recommendations=confidence_data["recommendations"]
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.exception("Transcribe failed: %s", e)
        # Clean up temp file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )

