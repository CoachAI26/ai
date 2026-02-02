"""
Test script for Confidence analysis service
"""
import asyncio
import os
import sys
from pathlib import Path
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from services.transcription import transcribe_audio_file
from services.filler_detection import detect_filler_words_with_gpt
from services.wpm_calculation import calculate_wpm
from services.pause_analysis import analyze_pauses_and_hesitations, calculate_fluency_score
from services.confidence_analysis import calculate_confidence_score


async def test_confidence_analysis(audio_file_path: str):
    """
    Test confidence analysis with audio file
    """
    print("\n" + "="*60)
    print("💪 TESTING CONFIDENCE ANALYSIS")
    print("="*60)
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Error: Audio file not found: {audio_file_path}")
        return None
    
    print(f"📁 Audio file: {audio_file_path}")
    print("🔄 Analyzing speech confidence...")
    
    try:
        # Step 1: Transcribe
        result = await transcribe_audio_file(audio_file_path)
        text = result["text"]
        duration_seconds = result["duration_seconds"]
        segments = result.get("segments")
        
        print("\n✅ Transcription successful!")
        print(f"\n📝 Transcribed text:\n{text}\n")
        
        # Step 2: Calculate WPM
        wpm_data = calculate_wpm(text, duration_seconds)
        
        # Step 3: Detect filler words
        filler_words = await detect_filler_words_with_gpt(text)
        
        # Step 4: Analyze pauses
        pause_data = analyze_pauses_and_hesitations(
            text=text,
            segments=segments,
            pause_threshold=0.5
        )
        
        # Step 5: Calculate fluency
        fluency_data = calculate_fluency_score(
            total_duration=duration_seconds,
            total_pause_time=pause_data["total_pause_time"],
            hesitation_count=pause_data["total_hesitations"],
            word_count=wpm_data["word_count"]
        )
        
        # Step 6: Calculate confidence (now async)
        confidence_data = await calculate_confidence_score(
            wpm=wpm_data["wpm"],
            filler_count=len(filler_words),
            word_count=wpm_data["word_count"],
            total_pauses=pause_data["total_pauses"],
            total_hesitations=pause_data["total_hesitations"],
            pause_ratio=fluency_data["pause_ratio"],
            hesitation_rate=fluency_data["hesitation_rate"],
            fluency_score=fluency_data["fluency_score"]
        )
        
        # Display results
        print("="*60)
        print("📊 SPEECH METRICS")
        print("="*60)
        print(f"⏱️  Duration: {duration_seconds:.2f} seconds")
        print(f"📝 Word count: {wpm_data['word_count']} words")
        print(f"🚀 WPM: {wpm_data['wpm']:.2f}")
        print(f"🤔 Filler words: {len(filler_words)}")
        print(f"⏸️  Pauses: {pause_data['total_pauses']}")
        print(f"🤔 Hesitations: {pause_data['total_hesitations']}")
        print(f"💬 Fluency score: {fluency_data['fluency_score']:.2f}/100")
        
        print("\n" + "="*60)
        print("💪 CONFIDENCE ANALYSIS")
        print("="*60)
        print(f"⭐ Overall Confidence Score: {confidence_data['confidence_score']:.2f}/100")
        print(f"📊 Overall Rating: {confidence_data['overall_rating']}")
        
        print(f"\n📈 Component Scores:")
        print(f"   🚀 WPM Score: {confidence_data['wpm_score']:.2f}/100")
        print(f"   🤔 Filler Score: {confidence_data['filler_score']:.2f}/100")
        print(f"   ⏸️  Pause Score: {confidence_data['pause_score']:.2f}/100")
        print(f"   🤔 Hesitation Score: {confidence_data['hesitation_score']:.2f}/100")
        
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(confidence_data['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print("="*60 + "\n")
        
        return confidence_data
    
    except Exception as e:
        print(f"❌ Error during confidence analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """
    Main test function
    """
    print("\n" + "🧪"*30)
    print("CONFIDENCE ANALYSIS TEST SCRIPT")
    print("🧪"*30)
    
    # Check if audio file path provided
    if len(sys.argv) > 1:
        audio_file_path = sys.argv[1]
        # Test with audio file
        asyncio.run(test_confidence_analysis(audio_file_path))
    else:
        print("\n📝 Please provide an audio file path as argument")
        print("   Example: python test_confidence.py audio.m4a\n")


if __name__ == "__main__":
    main()

