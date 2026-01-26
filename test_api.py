"""
Complete API test script
Tests all endpoints with actual HTTP requests
"""
import requests
import os
import sys
import time
from pathlib import Path

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"


def test_health_endpoint():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("🏥 TESTING HEALTH ENDPOINT")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Health endpoint working!")
            return True
        else:
            print("❌ Health endpoint failed!")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print("⚠️  Make sure the server is running: python main.py")
        return False


def test_root_endpoints():
    """Test root endpoints"""
    print("\n" + "="*60)
    print("🏠 TESTING ROOT ENDPOINTS")
    print("="*60)
    
    try:
        # Test main root
        response = requests.get(f"{BASE_URL}/")
        print(f"GET / - Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test API root
        response = requests.get(f"{API_BASE}/")
        print(f"\nGET /api/v1/ - Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("\n✅ Root endpoints working!")
            return True
        else:
            print("\n❌ Root endpoints failed!")
            return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def test_transcribe_endpoint(audio_file_path: str):
    """Test transcription endpoint with audio file"""
    print("\n" + "="*60)
    print("🎤 TESTING TRANSCRIBE ENDPOINT")
    print("="*60)
    
    if not os.path.exists(audio_file_path):
        print(f"❌ Error: Audio file not found: {audio_file_path}")
        return False
    
    print(f"📁 Audio file: {audio_file_path}")
    print("🔄 Uploading and processing...")
    
    try:
        with open(audio_file_path, "rb") as f:
            files = {"file": (os.path.basename(audio_file_path), f, "audio/m4a")}
            response = requests.post(
                f"{API_BASE}/transcribe",
                files=files,
                timeout=120  # 2 minutes timeout for processing
            )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Error: {response.text}")
            return False
        
        data = response.json()
        
        # Check all required fields
        required_fields = [
            "text",
            "filler_words",
            "filler_count",
            "cleaned_text",
            "duration_seconds",
            "word_count",
            "wpm",
            "total_pauses",
            "total_hesitations",
            "pause_durations",
            "average_pause_duration",
            "total_pause_time",
            "hesitation_words",
            "fluency_score",
            "pause_ratio",
            "hesitation_rate",
            "confidence_score",
            "wpm_score",
            "filler_score",
            "pause_score",
            "hesitation_score",
            "overall_rating",
            "recommendations"
        ]
        
        print("\n" + "="*60)
        print("📊 RESPONSE VALIDATION")
        print("="*60)
        
        missing_fields = []
        for field in required_fields:
            if field in data:
                print(f"✅ {field}: {type(data[field]).__name__}")
            else:
                print(f"❌ Missing: {field}")
                missing_fields.append(field)
        
        if missing_fields:
            print(f"\n❌ Missing {len(missing_fields)} required fields!")
            return False
        
        # Display results
        print("\n" + "="*60)
        print("📝 TRANSCRIPTION RESULTS")
        print("="*60)
        print(f"Text: {data['text'][:100]}..." if len(data['text']) > 100 else f"Text: {data['text']}")
        print(f"Word Count: {data['word_count']}")
        print(f"Duration: {data['duration_seconds']:.2f} seconds")
        print(f"WPM: {data['wpm']:.2f}")
        
        print("\n" + "="*60)
        print("🤔 FILLER WORDS ANALYSIS")
        print("="*60)
        print(f"Filler Count: {data['filler_count']}")
        if data['filler_words']:
            print("Filler Words:")
            for fw in data['filler_words'][:5]:  # Show first 5
                print(f"  - '{fw['word']}' at position {fw['position']}")
        
        print("\n" + "="*60)
        print("⏸️  PAUSE & HESITATION ANALYSIS")
        print("="*60)
        print(f"Total Pauses: {data['total_pauses']}")
        print(f"Total Hesitations: {data['total_hesitations']}")
        print(f"Total Pause Time: {data['total_pause_time']:.2f} seconds")
        print(f"Average Pause Duration: {data['average_pause_duration']:.2f} seconds")
        
        print("\n" + "="*60)
        print("💬 FLUENCY & CONFIDENCE")
        print("="*60)
        print(f"Fluency Score: {data['fluency_score']:.2f}/100")
        print(f"Confidence Score: {data['confidence_score']:.2f}/100")
        print(f"Overall Rating: {data['overall_rating']}")
        print(f"\nComponent Scores:")
        print(f"  WPM Score: {data['wpm_score']:.2f}/100")
        print(f"  Filler Score: {data['filler_score']:.2f}/100")
        print(f"  Pause Score: {data['pause_score']:.2f}/100")
        print(f"  Hesitation Score: {data['hesitation_score']:.2f}/100")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(data['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n✅ All fields present and endpoint working correctly!")
        print("="*60 + "\n")
        
        return True
        
    except requests.exceptions.Timeout:
        print("❌ Request timeout - processing took too long")
        return False
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def check_server_running():
    """Check if server is running"""
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def main():
    """Main test function"""
    print("\n" + "🧪"*30)
    print("COMPLETE API TEST")
    print("🧪"*30)
    
    # Check if server is running
    print("\n🔍 Checking if server is running...")
    if not check_server_running():
        print("❌ Server is not running!")
        print("⚠️  Please start the server first:")
        print("   python main.py")
        print("\n   Or in another terminal:")
        print("   uvicorn main:app --reload")
        return
    
    print("✅ Server is running!")
    
    # Test health endpoint
    if not test_health_endpoint():
        return
    
    # Test root endpoints
    if not test_root_endpoints():
        return
    
    # Test transcribe endpoint
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        test_transcribe_endpoint(audio_file)
    else:
        print("\n" + "="*60)
        print("📝 TRANSCRIBE ENDPOINT TEST")
        print("="*60)
        print("⚠️  No audio file provided")
        print("💡 Usage: python test_api.py audio.m4a")
        print("   Example: python test_api.py test2.m4a")


if __name__ == "__main__":
    main()

