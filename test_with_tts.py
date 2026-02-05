#!/usr/bin/env python3
"""
Test script to verify the full pipeline including TTS functionality
"""
import os
import base64
import requests
from pydub import AudioSegment
from pydub.playback import play

# Configuration
BASE_URL = "http://localhost:8000"
API_ENDPOINT = f"{BASE_URL}/api/v1/transcribe"
AUDIO_FILE = "test_fatemeh.m4a"
OUTPUT_DIR = "test_output"
LEVEL = "medium"
CATEGORY = "interview"
TITLE = "Tell me about yourself."

def convert_to_wav_if_needed(audio_path: str) -> str:
    """Convert audio to WAV format if needed"""
    if audio_path.lower().endswith('.m4a'):
        print("Converting M4A to WAV for better compatibility...")
        wav_path = os.path.splitext(audio_path)[0] + '.wav'
        audio = AudioSegment.from_file(audio_path, format='m4a')
        audio.export(wav_path, format='wav')
        return wav_path
    return audio_path

def test_full_pipeline(audio_file_path: str, output_dir: str = OUTPUT_DIR):
    """
    Test the full pipeline including TTS functionality
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Testing with file: {audio_file_path}")
    
    # Convert to WAV if needed
    converted_audio = convert_to_wav_if_needed(audio_file_path)
    use_temp_file = converted_audio != audio_file_path
    
    try:
        # Prepare the request with explicit MIME type
        with open(converted_audio, 'rb') as f:
            files = {
                'file': (
                    os.path.basename(converted_audio),
                    f,
                    'audio/wav' if converted_audio.endswith('.wav') else 'audio/m4a'
                )
            }
            data = {
                "level": LEVEL,
                "category": CATEGORY,
                "title": TITLE,
            }
            
            print("Sending request to API...")
            response = requests.post(API_ENDPOINT, files=files, data=data)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return
        
        # Get the response data
        data = response.json()
        
        # Save the original and improved text
        with open(f"{output_dir}/original.txt", "w", encoding='utf-8') as f:
            f.write(data["text"])
        
        with open(f"{output_dir}/improved.txt", "w", encoding='utf-8') as f:
            f.write(data["improved_text"])
        
        print("\n" + "="*50)
        print("ORIGINAL TEXT:")
        print("="*50)
        print(data["text"])
        
        print("\n" + "="*50)
        print("IMPROVED TEXT:")
        print("="*50)
        print(data["improved_text"])
        
        # Save and play TTS audio if available
        if data.get("tts_speech"):
            tts_data = data["tts_speech"]
            audio_data = base64.b64decode(tts_data["audio_content"])
            output_file = f"{output_dir}/improved_speech.mp3"
            
            with open(output_file, "wb") as f:
                f.write(audio_data)
            
            print(f"\nSaved TTS audio to: {output_file}")
            
            # Try to play the audio
            try:
                print("Playing TTS audio...")
                audio = AudioSegment.from_mp3(output_file)
                play(audio)
            except Exception as e:
                print(f"Could not play audio: {str(e)}")
                print("You can find the audio file at:", output_file)
        else:
            print("\nNo TTS audio data in response")
            
    finally:
        # Clean up temporary converted file if it was created
        if use_temp_file and os.path.exists(converted_audio):
            os.remove(converted_audio)

if __name__ == "__main__":
    # Test with the specified audio file
    test_full_pipeline(AUDIO_FILE)