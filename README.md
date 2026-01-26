# Voice Transcription & Speech Analysis API

A comprehensive API for transcribing audio to text and analyzing speech patterns including filler words, WPM, pauses, hesitations, and confidence scoring.

## Features

- **Audio Transcription**: Convert audio files to text using OpenAI Whisper API
- **Filler Word Detection**: Intelligent detection of filler words using GPT-4o with high accuracy
- **WPM Calculation**: Calculate Words Per Minute based on speaking duration
- **Pause & Hesitation Analysis**: Analyze pauses and hesitations in speech
- **Fluency Score**: Calculate fluency score based on speech patterns
- **Confidence Score**: Comprehensive confidence scoring based on all metrics
- **English Only**: Currently supports English language only
- **Text Cleaning**: Remove filler words to get cleaned text
- **RESTful API**: FastAPI endpoints for backend integration

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

**Important Note**: If you encounter a `proxies` error, you may need to reinstall dependencies:
```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

2. Create `.env` file and add your API key:
```
OPENAI_API_KEY=your_openai_api_key_here
```

## Project Structure

```
seon_ai/
├── main.py                 # FastAPI app entry point
├── config.py              # Configuration and OpenAI client
├── test_services.py       # Direct test for AI services (transcription + filler detection)
├── test_wpm.py            # Separate test for WPM calculation
├── test_pauses.py         # Separate test for Pause analysis
├── test_confidence.py     # Separate test for Confidence analysis
├── models/                # Pydantic models
│   ├── __init__.py
│   └── schemas.py
├── services/              # AI and business logic (each feature in separate file)
│   ├── __init__.py
│   ├── transcription.py   # Audio to text transcription
│   ├── filler_detection.py # Filler word detection
│   ├── wpm_calculation.py # WPM calculation
│   ├── pause_analysis.py  # Pause and hesitation analysis
│   └── confidence_analysis.py # Confidence score calculation
├── routes/                # API endpoints
│   ├── __init__.py
│   └── transcription.py
├── requirements.txt
└── README.md
```

## Running the Application

```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## API Usage

### Documentation
After running, you can view the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Health Check
```bash
GET http://localhost:8000/api/v1/health
```

### Transcribe Audio
```bash
POST http://localhost:8000/api/v1/transcribe
Content-Type: multipart/form-data

file: [audio_file]
```

### Example with curl:
```bash
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.mp3"
```

### API Response:
```json
{
  "text": "So, um, I think that, you know, the project is basically ready",
  "filler_words": [
    {
      "word": "um",
      "position": 4,
      "length": 2
    },
    {
      "word": "you know",
      "position": 25,
      "length": 8
    },
    {
      "word": "basically",
      "position": 45,
      "length": 9
    }
  ],
  "filler_count": 3,
  "cleaned_text": "So, I think that the project is ready",
  "duration_seconds": 12.5,
  "word_count": 31,
  "wpm": 148.8,
  "total_pauses": 2,
  "total_hesitations": 3,
  "fluency_score": 75.5,
  "confidence_score": 82.3,
  "overall_rating": "Good",
  "recommendations": [
    "You're doing well! Try to reduce filler words even further for more confident speech.",
    "Overall, your speech is good. Focus on the area mentioned above."
  ]
}
```

## Important Notes

- **Language**: English only
- **Filler Word Detection**: Uses GPT-4o for high accuracy
- **Context-Aware**: GPT makes decisions based on context to determine if a word is a filler

## Supported Audio Formats

OpenAI Whisper API supports the following formats:

- **MP3** (audio/mpeg)
- **WAV** (audio/wav)
- **M4A** (audio/x-m4a, audio/mp4) ✅
- **OGG** (audio/ogg)
- **FLAC** (audio/flac)
- **WebM** (audio/webm)
- **AAC** (audio/x-aac)

> **Note**: M4A format is fully supported and you can directly upload M4A files.

## Testing Services

You can test AI services directly without using the API:

### Test with Sample Text:
```bash
python test_services.py
```

This command tests with a sample text and displays results.

### Test with Audio File:
```bash
python test_services.py audio.mp3
```

This command:
1. Transcribes the audio file to text
2. Detects filler words
3. Displays cleaned text
4. Shows summary of results

### Test Output:
- Transcribed text
- List of filler words with positions
- Cleaned text
- Statistics and summary

## Testing WPM

For separate WPM calculation testing:

### Test with Sample Text:
```bash
python test_wpm.py
```

This command tests with a sample text and assumed duration.

### Test with Audio File:
```bash
python test_wpm.py audio.m4a
```

This command:
1. Transcribes the audio file to text
2. Extracts speaking duration from audio file
3. Counts words
4. Calculates WPM (Words Per Minute)
5. Displays WPM interpretation (slow/normal/fast)

### WPM Test Output:
- Speaking duration (duration_seconds)
- Word count (word_count)
- WPM (Words Per Minute)
- Speaking rate interpretation

## Testing Confidence Analysis

For confidence analysis testing:

### Test with Audio File:
```bash
python test_confidence.py audio.m4a
```

This command:
1. Calculates all metrics (WPM, Filler, Pause, Hesitation)
2. Calculates overall Confidence Score (0-100)
3. Displays Component Scores
4. Shows Overall Rating (Excellent/Good/Moderate/Low/Very Low)
5. Provides personalized improvement recommendations

### Confidence Test Output:
- Confidence Score (0-100)
- Component Scores (WPM, Filler, Pause, Hesitation)
- Overall Rating
- Recommendations (improvement suggestions)

## How Confidence Score is Calculated

Confidence Score is calculated based on weighted combination of these factors:

- **WPM Score (25%)**: Speaking rate (optimal: 120-160 WPM)
- **Filler Score (25%)**: Number of filler words (optimal: < 2 per 100 words)
- **Pause Score (20%)**: Pause time ratio (optimal: < 10%)
- **Hesitation Score (15%)**: Number of hesitation sounds (optimal: < 3 per 100 words)
- **Fluency Score (15%)**: Overall fluency score

The final score ranges from 0 to 100 and is divided into 5 categories:
- **85-100**: Excellent
- **70-84**: Good
- **55-69**: Moderate
- **40-54**: Low
- **0-39**: Very Low

## License

MIT License

## Repository

https://github.com/CoachAI26/ai.git
