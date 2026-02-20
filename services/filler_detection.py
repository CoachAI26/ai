"""
Filler word detection service using GPT
"""
import re
import json
from typing import List, Dict, Any, Optional
from config import (
    get_openai_client,
    GPT_MODEL,
    GPT_TEMPERATURE
)

# Filler word detection prompt for GPT
FILLER_WORD_DETECTION_PROMPT = """You are an expert at identifying filler words and disfluencies in spoken English. Your task is to analyze the transcribed text and identify ALL filler words, hesitations, and unnecessary words that should be removed for clarity.

Filler words include:
- Hesitation sounds: "um", "uh", "er", "erm", "ah", "hmm" - ALWAYS mark these as fillers
- Filler phrases used as pauses: "like", "you know", "I mean", "sort of", "kind of" (only when used as fillers)
- Unnecessary qualifiers when used as fillers: "basically", "actually", "literally" (when not adding meaning)
- Repetitive confirmations: "right", "okay", "yeah" (when used as fillers, not as actual responses)
- Thinking pauses: "well", "so" (when used to stall, not to transition meaningfully)

CRITICAL RULES:
1. ALWAYS mark ALL hesitation sounds (um, uh, er, erm, ah, hmm) - these are ALWAYS fillers regardless of context
2. Find EVERY occurrence of hesitation sounds in the text - do not miss any!
3. DO NOT mark words that have actual meaning in context:
   - "you know" when used to check understanding or emphasize a point (e.g., "but, you know, I try my best")
   - "like" when comparing or giving examples (e.g., "shirt like this")
   - "right" when confirming a fact or asking for agreement
   - "well" when starting a thoughtful response or transition
   - "actually" when correcting or providing accurate information
4. Be precise - context matters for phrases, but hesitation sounds are ALWAYS fillers
5. Include the exact word/phrase as it appears in the text (preserve case)
6. Find the character position (index) where EACH filler word starts in the original text
7. Count carefully - if there are multiple "um" or "uh" in the text, mark ALL of them
8. Be thorough - scan the entire text character by character to find all filler words

IMPORTANT: You MUST find ALL occurrences of hesitation sounds. If the text contains multiple "um" or "uh", you MUST mark ALL of them, not just one!

Example: If text is "um I was um in the mall um today", you should find THREE "um" words, not just one!

Return your response as a JSON object with this exact format:
{
  "fillers": [
    {
      "word": "um",
      "position": 15,
      "length": 2
    },
    {
      "word": "um",
      "position": 45,
      "length": 2
    },
    {
      "word": "uh",
      "position": 67,
      "length": 2
    }
  ]
}

If no filler words are found, return: {"fillers": []}

Text to analyze:
"""


def _regex_hesitation_fillers(text: str) -> List[Dict[str, Any]]:
    """
    Deterministic fallback: find *all* hesitation sounds in English text.
    This guarantees we never miss repeated 'um/uh/er/erm/ah/hmm' even if GPT does.
    """
    # word-boundaries + allow punctuation around tokens (e.g., "um," / "(uh)")
    pattern = re.compile(r"\b(um+|uh+|er+|erm+|ah+|hmm+)\b", re.IGNORECASE)
    out: List[Dict[str, Any]] = []
    for m in pattern.finditer(text):
        token = m.group(0)
        out.append({"word": token, "position": m.start(), "length": len(token)})
    return out


async def detect_filler_words_with_gpt(text: str) -> List[Dict[str, Any]]:
    """
    Detect filler words using GPT with high accuracy
    Only for English text
    
    Args:
        text: The transcribed text to analyze
        
    Returns:
        List of filler words with their positions and lengths
    """
    try:
        # Create prompt with the text
        prompt = FILLER_WORD_DETECTION_PROMPT + text
        
        # Call GPT-4o for better accuracy
        client = get_openai_client()
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at identifying filler words in spoken English. You MUST find ALL hesitation sounds (um, uh, er, etc.) in the text. Scan the entire text carefully and mark EVERY occurrence. Always return valid JSON only, no additional text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=GPT_TEMPERATURE,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        response_content = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        try:
            parsed = json.loads(response_content)
            
            # Handle both direct array and wrapped object
            if isinstance(parsed, list):
                filler_words = parsed
            elif isinstance(parsed, dict) and "fillers" in parsed:
                filler_words = parsed["fillers"]
            elif isinstance(parsed, dict) and "filler_words" in parsed:
                filler_words = parsed["filler_words"]
            else:
                # Try to find array in the response
                filler_words = []
                for key, value in parsed.items():
                    if isinstance(value, list):
                        filler_words = value
                        break
        except json.JSONDecodeError:
            # If JSON parsing fails, try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', response_content, re.DOTALL)
            if json_match:
                filler_words = json.loads(json_match.group(1))
            else:
                # Last resort: try to find array pattern
                array_match = re.search(r'\[.*?\]', response_content, re.DOTALL)
                if array_match:
                    filler_words = json.loads(array_match.group(0))
                else:
                    filler_words = []
        
        # Validate and clean the results
        validated_fillers = []
        for filler in filler_words:
            if isinstance(filler, dict) and "word" in filler and "position" in filler:
                word = filler["word"]
                position = int(filler["position"])
                length = filler.get("length", len(word))
                
                # Verify the position is valid
                if 0 <= position < len(text):
                    # Verify the word actually exists at that position
                    actual_word = text[position:position+length].strip()
                    if word.lower() in actual_word.lower() or actual_word.lower() in word.lower():
                        validated_fillers.append({
                            "word": word,
                            "position": position,
                            "length": length
                        })

        # Deterministic merge: always include all hesitation sounds found via regex
        # (GPT sometimes returns only the first occurrence.)
        regex_fillers = _regex_hesitation_fillers(text)
        existing_positions = {(f["position"], f["length"]) for f in validated_fillers}
        for f in regex_fillers:
            key = (f["position"], f["length"])
            if key not in existing_positions:
                validated_fillers.append(f)
                existing_positions.add(key)
        
        # Sort by position and remove overlaps
        validated_fillers = sorted(validated_fillers, key=lambda x: x['position'])
        non_overlapping = []
        last_end = -1
        
        for filler in validated_fillers:
            if filler['position'] >= last_end:
                non_overlapping.append(filler)
                last_end = filler['position'] + filler['length']
        
        return non_overlapping
    
    except Exception as e:
        # If GPT fails, return empty list (fallback)
        print(f"Error in GPT filler word detection: {str(e)}")
        return []


def remove_filler_words(text: str, filler_positions: List[Dict[str, Any]]) -> str:
    """
    Remove filler words from text
    
    Args:
        text: Original text
        filler_positions: List of filler words with positions and lengths
        
    Returns:
        Text with filler words removed
    """
    # Sort by position in reverse to remove from end to start
    filler_positions_sorted = sorted(filler_positions, key=lambda x: x['position'], reverse=True)
    
    result = text
    for filler in filler_positions_sorted:
        start = filler['position']
        end = start + filler['length']
        # Remove the filler word
        result = result[:start] + result[end:]
        # Clean up spaces around punctuation and collapse whitespace
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\s+([,.;:!?])', r'\1', result)   # space before punctuation
        result = re.sub(r'([,.;:!?])\s+', r'\1 ', result)  # normalize after punctuation
        result = result.strip()
    
    return result


async def generate_improved_text(
    text: str,
    level: Optional[str] = None,
    category: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """
    Generate an improved version of the text using GPT
    
    Args:
        text: Original text to improve
        
    Returns:
        Improved version of the text with better flow and clarity
    """
    from config import get_openai_client, GPT_MODEL
    
    client = get_openai_client()
    
    context_block = ""
    if level or category or title:
        context_block = "\n\nChallenge context:\n"
        if level:
            context_block += f"- Level: {level}\n"
        if category:
            context_block += f"- Category: {category}\n"
        if title:
            context_block += f"- Title: {title}\n"

    prompt = """
    You are a professional speech editor. Your task is to improve the following transcribed speech 
    by making it more concise, clear, and natural while preserving the original meaning and tone.
    
    Guidelines:
    1. Remove all filler words and hesitations (um, uh, like, you know, etc.)
    2. Fix any grammar or syntax errors
    3. Make the speech more concise by removing unnecessary repetition
    4. Improve sentence structure and flow
    5. Keep the original meaning and tone intact
    6. Maintain a conversational style
    7. Keep technical terms and proper nouns as-is
    
    Input text to improve:
    """
    
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a professional speech editor that improves transcribed speech."},
                {"role": "user", "content": f"{prompt}{context_block}\n\n{text}"}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        improved_text = response.choices[0].message.content.strip()
        
        # Remove any surrounding quotes if present
        improved_text = re.sub(r'^"|"$', '', improved_text)
        
        return improved_text
    except Exception as e:
        print(f"Error generating improved text: {str(e)}")
        # Return the original text if there's an error
        return text


# Fixed message when user's answer is not relevant to the challenge title
OFF_TOPIC_MESSAGE = (
    "Your response doesn't seem to address the challenge topic. "
    "Please try again and speak about the given question or topic."
)


async def check_answer_relevance_to_title(title: str, user_text: str) -> bool:
    """
    Check if the user's transcribed answer is relevant to the challenge title/question.
    Returns True if relevant or if we cannot determine, False if clearly off-topic.
    """
    if not title or not (user_text or "").strip():
        return True
    from config import get_openai_client, GPT_MODEL
    client = get_openai_client()
    prompt = f"""You are a strict judge. The challenge question/topic is:
"{title}"

The user's spoken answer (transcribed) is:
"{user_text.strip()}"

Is this answer clearly relevant to the question/topic? Does it address the same subject?
Answer with exactly one word: YES or NO.
- YES: the answer is about the same topic or directly responds to the question.
- NO: the answer is about something else, unrelated, or just filler/noise."""
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You answer only YES or NO. No explanation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "").strip().upper()
        return raw.startswith("YES")
    except Exception as e:
        print(f"Error checking relevance: {str(e)}")
        return True  # On error, do not penalize

