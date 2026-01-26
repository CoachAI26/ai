"""
Transcription API endpoints
"""
import os
import re
from fastapi import APIRouter, File, UploadFile, HTTPException
from models.schemas import TranscriptionResponse
from services.transcription import transcribe_audio_file
from services.filler_detection import detect_filler_words_with_gpt, remove_filler_words
from services.wpm_calculation import calculate_wpm
from services.pause_analysis import analyze_pauses_and_hesitations, calculate_fluency_score
from services.confidence_analysis import calculate_confidence_score

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
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Transcribe audio file to text and detect filler words
    
    - **file**: Audio file (MP3, WAV, M4A, OGG, FLAC, WebM, etc.)
    
    Supported formats:
    - MP3 (audio/mpeg)
    - WAV (audio/wav)
    - M4A (audio/x-m4a, audio/mp4)
    - OGG (audio/ogg)
    - FLAC (audio/flac)
    - WebM (audio/webm)
    - AAC (audio/x-aac)
    
    Returns:
    - **text**: Full transcribed text
    - **filler_words**: List of detected filler words with positions
    - **filler_count**: Number of filler words found
    - **cleaned_text**: Text with filler words removed
    """
    # Validate file type
    if not file.content_type:
        # Try to validate by file extension if content_type is missing
        if file.filename:
            ext = file.filename.lower().split('.')[-1]
            supported_extensions = {'mp3', 'wav', 'm4a', 'ogg', 'flac', 'webm', 'aac', 'mp4'}
            if ext not in supported_extensions:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format. Supported formats: MP3, WAV, M4A, OGG, FLAC, WebM, AAC"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="File type could not be determined. Please ensure the file is an audio file."
            )
    elif file.content_type not in SUPPORTED_AUDIO_FORMATS and not file.content_type.startswith('audio/'):
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
        
        # Transcribe using OpenAI Whisper (English only)
        transcription_result = await transcribe_audio_file(temp_file_path)
        text = transcription_result["text"]
        duration_seconds = transcription_result["duration_seconds"]
        segments = transcription_result.get("segments")
        
        # Detect filler words using GPT
        filler_words = await detect_filler_words_with_gpt(text)
        
        # Remove filler words to get cleaned text
        cleaned_text = remove_filler_words(text, filler_words)
        
        # Calculate WPM
        wpm_data = calculate_wpm(text, duration_seconds)
        
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
        
        # Calculate confidence score
        confidence_data = calculate_confidence_score(
            wpm=wpm_data["wpm"],
            filler_count=len(filler_words),
            word_count=wpm_data["word_count"],
            total_pauses=pause_data["total_pauses"],
            total_hesitations=pause_data["total_hesitations"],
            pause_ratio=fluency_data["pause_ratio"],
            hesitation_rate=fluency_data["hesitation_rate"],
            fluency_score=fluency_data["fluency_score"]
        )
        
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        return TranscriptionResponse(
            text=text,
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
        # Clean up temp file if it exists
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )

