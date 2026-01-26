"""
Test script for AI services without using API endpoints
"""
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from services.transcription import transcribe_audio_file
from services.filler_detection import detect_filler_words_with_gpt, remove_filler_words


async def test_transcription(audio_file_path: str):
    """
    Test audio transcription
    """
    print("\n" + "="*60)
    print("🎤 TESTING AUDIO TRANSCRIPTION")
    print("="*60)
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Error: Audio file not found: {audio_file_path}")
        return None
    
    print(f"📁 Audio file: {audio_file_path}")
    print("🔄 Transcribing...")
    
    try:
        result = await transcribe_audio_file(audio_file_path)
        text = result["text"]
        duration_seconds = result["duration_seconds"]
        print("\n✅ Transcription successful!")
        if isinstance(duration_seconds, (int, float)):
            print(f"⏱️  Duration: {duration_seconds:.2f}s")
        print(f"\n📝 Transcribed text:\n{text}\n")
        return text, duration_seconds
    except Exception as e:
        print(f"❌ Error during transcription: {str(e)}")
        return None, None


async def test_filler_word_detection(text: str):
    """
    Test filler word detection with GPT
    """
    print("\n" + "="*60)
    print("🔍 TESTING FILLER WORD DETECTION")
    print("="*60)
    
    print(f"📝 Analyzing text:\n{text}\n")
    print("🔄 Detecting filler words with GPT-4o...")
    
    try:
        filler_words = await detect_filler_words_with_gpt(text)
        
        print(f"\n✅ Detection complete!")
        print(f"📊 Found {len(filler_words)} filler word(s)\n")
        
        if filler_words:
            print("🎯 Detected filler words:")
            for i, filler in enumerate(filler_words, 1):
                word = filler['word']
                position = filler['position']
                length = filler['length']
                # Show context around the filler word
                start = max(0, position - 20)
                end = min(len(text), position + length + 20)
                context = text[start:end]
                highlighted = context.replace(
                    word, 
                    f"👉{word}👈"
                )
                print(f"\n  {i}. '{word}'")
                print(f"     Position: {position}, Length: {length}")
                print(f"     Context: ...{highlighted}...")
        else:
            print("✨ No filler words detected!")
        
        return filler_words
    except Exception as e:
        print(f"❌ Error during filler word detection: {str(e)}")
        return []


def test_text_cleaning(text: str, filler_words: list):
    """
    Test text cleaning (removing filler words)
    """
    print("\n" + "="*60)
    print("🧹 TESTING TEXT CLEANING")
    print("="*60)
    
    print("🔄 Removing filler words...")
    
    try:
        cleaned_text = remove_filler_words(text, filler_words)
        
        print("\n✅ Cleaning complete!")
        print(f"\n📝 Original text:\n{text}\n")
        print(f"✨ Cleaned text:\n{cleaned_text}\n")
        
        return cleaned_text
    except Exception as e:
        print(f"❌ Error during text cleaning: {str(e)}")
        return text


async def test_with_audio_file(audio_file_path: str):
    """
    Complete test with audio file
    """
    print("\n" + "🚀"*30)
    print("STARTING COMPLETE TEST WITH AUDIO FILE")
    print("🚀"*30)
    
    # Step 1: Transcribe
    text, duration_seconds = await test_transcription(audio_file_path)
    if not text:
        return
    
    # Step 2: Detect filler words
    filler_words = await test_filler_word_detection(text)
    
    # Step 3: Clean text
    cleaned_text = test_text_cleaning(text, filler_words)
    
    # WPM summary (based on audio duration)
    word_count = len(text.split())
    wpm = None
    if isinstance(duration_seconds, (int, float)) and duration_seconds > 0:
        wpm = (word_count / duration_seconds) * 60.0

    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Original text length: {len(text)} characters")
    if isinstance(duration_seconds, (int, float)):
        print(f"Duration seconds: {duration_seconds:.2f}")
    print(f"Word count: {word_count}")
    if wpm is not None:
        print(f"WPM: {wpm:.2f}")
    print(f"Filler words found: {len(filler_words)}")
    print(f"Cleaned text length: {len(cleaned_text)} characters")
    print(f"Characters removed: {len(text) - len(cleaned_text)}")
    print("="*60 + "\n")


async def test_with_text_only(text: str):
    """
    Test with text only (no audio file)
    """
    print("\n" + "🚀"*30)
    print("STARTING TEST WITH TEXT ONLY")
    print("🚀"*30)
    
    # Step 1: Detect filler words
    filler_words = await test_filler_word_detection(text)
    
    # Step 2: Clean text
    cleaned_text = test_text_cleaning(text, filler_words)
    
    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)
    print(f"Original text length: {len(text)} characters")
    print(f"Filler words found: {len(filler_words)}")
    print(f"Cleaned text length: {len(cleaned_text)} characters")
    print(f"Characters removed: {len(text) - len(cleaned_text)}")
    print("="*60 + "\n")


def main():
    """
    Main test function
    """
    print("\n" + "🧪"*30)
    print("AI SERVICES TEST SCRIPT")
    print("🧪"*30)
    
    # Check if audio file path provided
    if len(sys.argv) > 1:
        audio_file_path = sys.argv[1]
        # Test with audio file
        asyncio.run(test_with_audio_file(audio_file_path))
    else:
        # Test with sample text
        sample_text = """So, um, I think that, you know, the project is basically ready. 
        Well, actually, I mean, it's sort of working, but, like, we need to test it more, 
        right? I guess, um, we should probably, you know, run some more tests."""
        
        print("\n📝 Using sample text for testing...")
        print("💡 Tip: Provide an audio file path as argument to test transcription")
        print("   Example: python test_services.py audio.mp3\n")
        
        asyncio.run(test_with_text_only(sample_text))


if __name__ == "__main__":
    main()

