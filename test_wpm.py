"""
Test script for WPM calculation service
"""
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from services.transcription import transcribe_audio_file
from services.wpm_calculation import calculate_wpm, count_words


async def test_wpm_calculation(audio_file_path: str):
    """
    Test WPM calculation with audio file
    """
    print("\n" + "="*60)
    print("📊 TESTING WPM CALCULATION")
    print("="*60)
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Error: Audio file not found: {audio_file_path}")
        return None
    
    print(f"📁 Audio file: {audio_file_path}")
    print("🔄 Transcribing and calculating WPM...")
    
    try:
        # Transcribe audio
        result = await transcribe_audio_file(audio_file_path)
        text = result["text"]
        duration_seconds = result["duration_seconds"]
        
        print("\n✅ Transcription successful!")
        print(f"\n📝 Transcribed text:\n{text}\n")
        
        # Calculate WPM
        wpm_data = calculate_wpm(text, duration_seconds)
        
        print("="*60)
        print("📊 WPM CALCULATION RESULTS")
        print("="*60)
        print(f"⏱️  Duration: {wpm_data['duration_seconds']} seconds")
        print(f"📝 Word count: {wpm_data['word_count']} words")
        print(f"🚀 WPM (Words Per Minute): {wpm_data['wpm']}")
        print("="*60)
        
        # Additional analysis
        if duration_seconds > 0:
            words_per_second = wpm_data['word_count'] / duration_seconds
            print(f"\n📈 Additional metrics:")
            print(f"   Words per second: {words_per_second:.2f}")
            print(f"   Average time per word: {duration_seconds / wpm_data['word_count']:.2f} seconds")
        
        # WPM interpretation
        print(f"\n💡 WPM Interpretation:")
        if wpm_data['wpm'] < 100:
            print("   Slow speech (typical: 100-150 WPM)")
        elif wpm_data['wpm'] <= 150:
            print("   Normal speech rate (typical: 100-150 WPM)")
        elif wpm_data['wpm'] <= 200:
            print("   Fast speech (typical: 150-200 WPM)")
        else:
            print("   Very fast speech (over 200 WPM)")
        
        return wpm_data
    
    except Exception as e:
        print(f"❌ Error during WPM calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_wpm_with_text_only(text: str, duration_seconds: float):
    """
    Test WPM calculation with text only (no audio file)
    """
    print("\n" + "="*60)
    print("📊 TESTING WPM CALCULATION (TEXT ONLY)")
    print("="*60)
    
    print(f"📝 Text: {text}")
    print(f"⏱️  Duration: {duration_seconds} seconds\n")
    
    try:
        wpm_data = calculate_wpm(text, duration_seconds)
        
        print("="*60)
        print("📊 WPM CALCULATION RESULTS")
        print("="*60)
        print(f"📝 Word count: {wpm_data['word_count']} words")
        print(f"⏱️  Duration: {wpm_data['duration_seconds']} seconds")
        print(f"🚀 WPM (Words Per Minute): {wpm_data['wpm']}")
        print("="*60 + "\n")
        
        return wpm_data
    
    except Exception as e:
        print(f"❌ Error during WPM calculation: {str(e)}")
        return None


def main():
    """
    Main test function
    """
    print("\n" + "🧪"*30)
    print("WPM CALCULATION TEST SCRIPT")
    print("🧪"*30)
    
    # Check if audio file path provided
    if len(sys.argv) > 1:
        audio_file_path = sys.argv[1]
        # Test with audio file
        asyncio.run(test_wpm_calculation(audio_file_path))
    else:
        # Test with sample text
        sample_text = "So, um, I think that, you know, the project is basically ready. Well, actually, I mean, it's sort of working, but, like, we need to test it more, right?"
        sample_duration = 15.5  # 15.5 seconds
        
        print("\n📝 Using sample text for testing...")
        print("💡 Tip: Provide an audio file path as argument to test with real audio")
        print("   Example: python test_wpm.py audio.m4a\n")
        
        test_wpm_with_text_only(sample_text, sample_duration)


if __name__ == "__main__":
    main()

