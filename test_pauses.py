"""
Test script for Pause and Hesitation analysis service
"""
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from services.transcription import transcribe_audio_file
from services.pause_analysis import analyze_pauses_and_hesitations, calculate_fluency_score
from services.wpm_calculation import count_words


async def test_pause_analysis(audio_file_path: str):
    """
    Test pause and hesitation analysis with audio file
    """
    print("\n" + "="*60)
    print("⏸️  TESTING PAUSE AND HESITATION ANALYSIS")
    print("="*60)
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Error: Audio file not found: {audio_file_path}")
        return None
    
    print(f"📁 Audio file: {audio_file_path}")
    print("🔄 Transcribing and analyzing pauses...")
    
    try:
        # Transcribe audio
        result = await transcribe_audio_file(audio_file_path)
        text = result["text"]
        duration_seconds = result["duration_seconds"]
        segments = result.get("segments")
        
        print("\n✅ Transcription successful!")
        print(f"\n📝 Transcribed text:\n{text}\n")
        
        # Analyze pauses and hesitations
        pause_data = analyze_pauses_and_hesitations(
            text=text,
            segments=segments,
            pause_threshold=0.5  # 0.5 seconds minimum for a pause
        )
        
        # Count words
        word_count = count_words(text)
        
        # Calculate fluency score
        fluency_data = calculate_fluency_score(
            total_duration=duration_seconds,
            total_pause_time=pause_data["total_pause_time"],
            hesitation_count=pause_data["total_hesitations"],
            word_count=word_count
        )
        
        # Display results
        print("="*60)
        print("⏸️  PAUSE ANALYSIS RESULTS")
        print("="*60)
        print(f"⏱️  Total duration: {duration_seconds:.2f} seconds")
        print(f"📊 Total pauses: {pause_data['total_pauses']}")
        print(f"🤔 Total hesitations: {pause_data['total_hesitations']}")
        
        if pause_data['pause_durations']:
            print(f"\n📈 Pause durations: {pause_data['pause_durations']} seconds")
            print(f"📊 Average pause duration: {pause_data['average_pause_duration']:.2f} seconds")
            print(f"⏱️  Total pause time: {pause_data['total_pause_time']:.2f} seconds")
        else:
            print("\n📈 No significant pauses detected (threshold: 0.5s)")
        
        if pause_data['hesitation_words']:
            print(f"\n🤔 Hesitation words found: {pause_data['hesitation_words']}")
        
        print("\n" + "="*60)
        print("💬 FLUENCY ANALYSIS")
        print("="*60)
        print(f"⭐ Fluency score: {fluency_data['fluency_score']:.2f}/100")
        print(f"📊 Pause ratio: {fluency_data['pause_ratio']:.3f} ({fluency_data['pause_ratio']*100:.1f}% of total time)")
        print(f"🤔 Hesitation rate: {fluency_data['hesitation_rate']:.2f} per 100 words")
        
        # Interpretation
        print("\n💡 Interpretation:")
        if fluency_data['fluency_score'] >= 80:
            print("   ✅ Excellent fluency - very smooth speech")
        elif fluency_data['fluency_score'] >= 60:
            print("   ✅ Good fluency - generally smooth speech")
        elif fluency_data['fluency_score'] >= 40:
            print("   ⚠️  Moderate fluency - some pauses and hesitations")
        else:
            print("   ⚠️  Low fluency - frequent pauses and hesitations")
        
        print("="*60 + "\n")
        
        return {
            **pause_data,
            **fluency_data
        }
    
    except Exception as e:
        print(f"❌ Error during pause analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_pauses_with_text_only(text: str, duration_seconds: float):
    """
    Test pause analysis with text only (no audio file)
    Note: Without segments, only hesitations can be detected
    """
    print("\n" + "="*60)
    print("⏸️  TESTING PAUSE ANALYSIS (TEXT ONLY)")
    print("="*60)
    
    print(f"📝 Text: {text}")
    print(f"⏱️  Duration: {duration_seconds} seconds\n")
    
    try:
        # Analyze pauses and hesitations (without segments, only hesitations will be detected)
        pause_data = analyze_pauses_and_hesitations(
            text=text,
            segments=None,  # No segments available
            pause_threshold=0.5
        )
        
        word_count = count_words(text)
        
        # Calculate fluency score
        fluency_data = calculate_fluency_score(
            total_duration=duration_seconds,
            total_pause_time=pause_data["total_pause_time"],
            hesitation_count=pause_data["total_hesitations"],
            word_count=word_count
        )
        
        print("="*60)
        print("⏸️  PAUSE ANALYSIS RESULTS")
        print("="*60)
        print(f"🤔 Total hesitations: {pause_data['total_hesitations']}")
        if pause_data['hesitation_words']:
            print(f"🤔 Hesitation words: {pause_data['hesitation_words']}")
        print(f"📊 Note: Pause analysis requires audio segments (not available with text only)")
        
        print("\n" + "="*60)
        print("💬 FLUENCY ANALYSIS")
        print("="*60)
        print(f"⭐ Fluency score: {fluency_data['fluency_score']:.2f}/100")
        print(f"🤔 Hesitation rate: {fluency_data['hesitation_rate']:.2f} per 100 words")
        print("="*60 + "\n")
        
        return {
            **pause_data,
            **fluency_data
        }
    
    except Exception as e:
        print(f"❌ Error during pause analysis: {str(e)}")
        return None


def main():
    """
    Main test function
    """
    print("\n" + "🧪"*30)
    print("PAUSE AND HESITATION ANALYSIS TEST SCRIPT")
    print("🧪"*30)
    
    # Check if audio file path provided
    if len(sys.argv) > 1:
        audio_file_path = sys.argv[1]
        # Test with audio file
        asyncio.run(test_pause_analysis(audio_file_path))
    else:
        # Test with sample text
        sample_text = "So, um, I think that, you know, the project is basically ready. Well, actually, I mean, it's sort of working, but, like, we need to test it more, right?"
        sample_duration = 15.5  # 15.5 seconds
        
        print("\n📝 Using sample text for testing...")
        print("💡 Tip: Provide an audio file path as argument to test with real audio and pause detection")
        print("   Example: python test_pauses.py audio.m4a\n")
        
        test_pauses_with_text_only(sample_text, sample_duration)


if __name__ == "__main__":
    main()

