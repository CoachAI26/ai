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
GPT_MODEL = "gpt-5.2"  
GPT_TEMPERATURE = 0.0 

# Language settings
TRANSCRIPTION_LANGUAGE = "en"  # English only

# Whisper prompt to preserve filler words and hesitations with maximum fidelity
# CRITICAL: This is the most important setting for accurate spoken English transcription
WHISPER_PROMPT = (
    "This is a natural English conversation with hesitations, thinking pauses, and filler words. "
    "CRITICAL TRANSCRIPTION RULES:\n"
    "1. PRESERVE EVERY HESITATION: um, uh, er, erm, ah, hmm, mm-hmm, mhm, eh, uh-huh ARE WORDS and must be transcribed.\n"
    "2. EACH FILLER IS DISTINCT: If speaker says 'um' 3 times, write 'um um um' not 'um...um'.\n"
    "3. NATURAL SPEECH MARKERS to preserve: 'like', 'you know', 'I mean', 'sort of', 'kind of', 'actually', 'basically'.\n"
    "4. EXAMPLE correct transcriptions:\n"
    "   Speaker audio: [hesitant voice] 'Um, I think, um, the answer is, uh, probably yes'\n"
    "   CORRECT: 'Um, I think, um, the answer is, uh, probably yes'\n"
    "   WRONG: 'I think the answer is probably yes'\n\n"
    "   Speaker audio: [with multiple pauses] 'uh I was uh thinking uh maybe uh we could'\n"
    "   CORRECT: 'uh I was uh thinking uh maybe uh we could'\n"
    "   WRONG: 'I was thinking maybe we could'\n\n"
    "5. PAUSE INDICATOR: When speaker takes a physical pause/breath (0.3+ seconds), transcribe as-is; don't skip the surrounding words/fillers.\n"
    "6. ACCURACY over cleanliness: Include every 'um', 'uh', 'er', 'hmm' you hear, even if it sounds repetitive.\n"
    "7. CONTRACTED FILLERS: Preserve 'mm-hmm', 'uh-huh', 'um-um' exactly as heard.\n"
    "DO NOT remove, skip, or clean up filler words. Transcribe EVERYTHING spoken, exactly as spoken."
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

