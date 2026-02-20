# API Endpoints Documentation

## Base URL
```
http://localhost:8000
```

## Endpoints

### 1. Root Endpoint
**GET** `/`

Returns API information and available endpoints.

**Response:**
```json
{
  "message": "Voice Transcription & Filler Word Detection API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/api/v1/health"
}
```

### 2. API Root Endpoint
**GET** `/api/v1/`

Returns API version and available endpoints.

**Response:**
```json
{
  "message": "Voice Transcription & Filler Word Detection API",
  "version": "1.0.0",
  "endpoints": {
    "/api/v1/transcribe": "POST - Upload audio file for transcription",
    "/api/v1/health": "GET - Health check"
  }
}
```

### 3. Health Check
**GET** `/api/v1/health`

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "service": "transcription-api"
}
```

### 4. Transcribe Audio (Main Endpoint)
**POST** `/api/v1/transcribe`

Upload an audio file and get comprehensive speech analysis.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (audio file)

**Supported Formats:**
- MP3, WAV, M4A, OGG, FLAC, WebM, AAC

**Request (optional form fields):**
- `level` (optional): Challenge level
- `category` (optional): Category
- `title` (optional): Title

**Response:**
```json
{
  "text": "Full transcribed text",
  "improved_text": "Improved version with better flow and clarity",
  "tts_speech": { "audio_content": "<base64>", "audio_format": "mp3", "voice": "alloy" },
  "level": "B1",
  "category": "Speaking",
  "title": "My presentation",
  "filler_words": [
    {
      "word": "um",
      "position": 4,
      "length": 2
    }
  ],
  "filler_count": 3,
  "cleaned_text": "Text without filler words",
  "duration_seconds": 12.5,
  "word_count": 31,
  "wpm": 148.8,
  "total_pauses": 2,
  "total_hesitations": 3,
  "pause_durations": [0.8, 1.2],
  "average_pause_duration": 1.0,
  "total_pause_time": 2.0,
  "hesitation_words": ["um", "uh"],
  "fluency_score": 75.5,
  "pause_ratio": 0.16,
  "hesitation_rate": 9.68,
  "confidence_score": 82.3,
  "wpm_score": 95.0,
  "filler_score": 80.0,
  "pause_score": 75.0,
  "hesitation_score": 70.0,
  "overall_rating": "Good",
  "recommendations": [
    "You're doing well! Try to reduce filler words even further for more confident speech."
  ]
}
```

**Response Fields:**
- `text`: Transcribed text (original)
- `improved_text`: Improved version of the text (better flow, no fillers)
- `tts_speech`: TTS audio of improved text (base64)
- `level`, `category`, `title`: Echo of request context (if sent)
- `filler_words`: List of filler words with positions
- `filler_count`: Number of filler words
- `cleaned_text`: Text without filler words
- `duration_seconds`: Speaking duration
- `word_count`: Total word count
- `wpm`: Words per minute
- `total_pauses`: Number of pauses
- `total_hesitations`: Number of hesitation sounds
- `pause_durations`: List of pause durations
- `average_pause_duration`: Average pause duration
- `total_pause_time`: Total pause time
- `hesitation_words`: List of hesitation words
- `fluency_score`: Fluency score (0-100)
- `pause_ratio`: Pause time ratio
- `hesitation_rate`: Hesitations per 100 words
- `confidence_score`: Overall confidence (0-100)
- `wpm_score`: WPM component score
- `filler_score`: Filler component score
- `pause_score`: Pause component score
- `hesitation_score`: Hesitation component score
- `overall_rating`: Rating (Excellent/Good/Moderate/Low/Very Low)
- `recommendations`: Improvement recommendations

## Testing

### Using curl:
```bash
curl -X POST "http://localhost:8000/api/v1/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.m4a"
```

### Using Python test script:
```bash
# Start server first
python main.py

# In another terminal
python test_api.py audio.m4a
```

### Using Swagger UI:
Visit `http://localhost:8000/docs` for interactive API documentation.

