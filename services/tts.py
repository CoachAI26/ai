"""
Text-to-Speech service using OpenAI's TTS model
"""
import os
import base64
from pathlib import Path
from typing import Optional, Dict, Any
from config import get_openai_client

# Default voice for TTS
DEFAULT_VOICE = "alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer


async def text_to_speech(text: str, voice: str = DEFAULT_VOICE) -> Dict[str, Any]:
    """
    Convert text to speech using OpenAI's TTS model
    
    Args:
        text: Text to convert to speech
        voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        
    Returns:
        Dictionary containing:
        - audio_content: Base64 encoded audio data
        - audio_format: Format of the audio (mp3)
        - voice: Voice used for TTS
    """
    client = get_openai_client()
    
    try:
        # Call OpenAI TTS API
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="mp3"
        )
        
        # Convert audio response to base64
        audio_data = response.content
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        return {
            "audio_content": audio_base64,
            "audio_format": "mp3",
            "voice": voice
        }
        
    except Exception as e:
        print(f"Error in text-to-speech conversion: {str(e)}")
        raise ValueError(f"Failed to convert text to speech: {str(e)}")
