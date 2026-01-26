"""
Configuration and OpenAI client setup
"""
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API Key
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in .env file")

# Model configurations
WHISPER_MODEL = "whisper-1"
GPT_MODEL = "gpt-4o"
GPT_TEMPERATURE = 0.1

# Language settings
TRANSCRIPTION_LANGUAGE = "en"  # English only

# Whisper prompt to preserve filler words
# Using a more explicit prompt with examples to guide Whisper
WHISPER_PROMPT = (
    "um um um This is a spoken transcription. um uh er Please transcribe EVERYTHING exactly as spoken. "
    "um uh er erm ah hmm Include ALL filler words like: um, uh, er, erm, ah, hmm, mmm, umm, uhh. "
    "Do NOT remove or skip any hesitation sounds. Transcribe um every single um word and sound. "
    "Example: 'um I was um in the mall um today' should be transcribed exactly as 'um I was um in the mall um today'. "
    "Preserve all um uh er sounds exactly as they are spoken."
)

# Lazy initialization of OpenAI client
_openai_client = None

def get_openai_client() -> OpenAI:
    """
    Get or create OpenAI client (lazy initialization)
    This prevents initialization errors during import
    Handles httpx compatibility issues
    """
    global _openai_client
    if _openai_client is None:
        try:
            # Try to create client with default settings
            _openai_client = OpenAI(api_key=API_KEY)
        except (TypeError, ValueError) as e:
            if 'proxies' in str(e) or 'unexpected keyword' in str(e).lower():
                # Fallback: create client with custom httpx client to avoid proxies issue
                import httpx
                # Create a custom httpx client
                http_client = httpx.Client(
                    timeout=httpx.Timeout(60.0, connect=10.0)
                )
                _openai_client = OpenAI(
                    api_key=API_KEY,
                    http_client=http_client
                )
            else:
                raise
    return _openai_client

