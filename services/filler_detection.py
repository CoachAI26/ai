"""
Filler word detection service using GPT
"""
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from config import (
    get_openai_client,
    GPT_MODEL,
    GPT_TEMPERATURE
)

# Regex pattern to catch hesitation sounds (um, uh, er, erm, ah, hmm) — used to strengthen GPT results
HESITATION_REGEX = re.compile(
    r"\b(um+|uh+|u+h+|er+|erm+|ah+|hmm+|mhm+|eh+|eh+m+)\b",
    re.IGNORECASE,
)

# Filler word detection prompt for GPT — high-accuracy; regex strengthens hesitation detection.
FILLER_WORD_DETECTION_PROMPT = """You are an expert at identifying filler words and disfluencies in spoken English. Your task is to analyze the transcribed text and identify ALL filler words, hesitations, and unnecessary words with MAXIMUM ACCURACY.

Filler words include:
- Hesitation sounds: "um", "uh", "er", "erm", "ah", "hmm", "umm", "uhh" — mark EVERY occurrence, no exceptions
- Filler phrases when used as fillers: "like", "you know", "I mean", "sort of", "kind of"
- Unnecessary qualifiers when used as fillers: "basically", "actually", "literally" (when they add no meaning)
- Repetitive fillers: "right", "okay", "yeah" when used to stall (not as real answers)
- Thinking stalls: "well", "so" when used only to buy time, not to transition meaningfully

CRITICAL RULES FOR MAXIMUM ACCURACY:
1. Scan the text character by character from start to end. Do not skip any part.
2. For hesitation sounds (um, uh, er, erm, ah, hmm, umm, uhh): mark EVERY single occurrence, including repeated letters (e.g. "umm", "uhh"). If "um" appears 5 times, output 5 entries with correct positions.
3. "position" must be the exact character index (0-based) where the filler STARTS in the original text. Count spaces and punctuation.
4. "length" must be the exact number of characters of that filler as it appears in the text.
5. Preserve the exact "word" as written (case, spelling). Use the substring from text[position:position+length].
6. Do NOT mark words that have real meaning in context: "like" in "something like that", "right" when confirming, "well" when starting an answer, "actually" when correcting.
7. When in doubt whether a word is a filler in context, prefer marking it if it is a hesitation sound (um/uh/er/erm/ah/hmm); for phrases (like, you know) only mark when clearly used as filler.
8. Output every filler once; no duplicates for the same span; no overlapping spans.

Example: For "um I was um in the mall um today" you must return THREE entries for "um" at positions 0, 9, and 25 (or whatever the exact indices are in the string).

Also count the total number of words in the text (standard word count: split by whitespace, count each token; e.g. "I'd" is one word, "wasn't" is one word). Include this as "word_count" in your JSON.

Return your response as a JSON object with this exact format:
{
  "word_count": 42,
  "fillers": [
    {
      "word": "um",
      "position": 15,
      "length": 2
    },
    {
      "word": "uh",
      "position": 67,
      "length": 2
    }
  ]
}

If no filler words are found, return: {"word_count": <number>, "fillers": []}
Always include "word_count" as an integer.

Text to analyze:
"""


async def detect_filler_words_with_gpt(text: str) -> Tuple[List[Dict[str, Any]], int]:
    """
    Detect filler words and get word count using GPT.
    Only for English text.

    Returns:
        (list of filler words with positions/lengths, word_count from GPT)
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
                    "content": "You are the only source of filler detection and word count. You MUST find every filler with exact character positions and include word_count (total words in the text, split by whitespace). Return only valid JSON with word_count and fillers, no extra text."
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
        word_count_from_gpt: Optional[int] = None
        try:
            parsed = json.loads(response_content)
            if isinstance(parsed, dict):
                word_count_from_gpt = parsed.get("word_count")
                if word_count_from_gpt is not None:
                    try:
                        word_count_from_gpt = int(word_count_from_gpt)
                    except (TypeError, ValueError):
                        word_count_from_gpt = None
            # Handle both direct array and wrapped object
            if isinstance(parsed, list):
                filler_words = parsed
            elif isinstance(parsed, dict) and "fillers" in parsed:
                filler_words = parsed["fillers"]
            elif isinstance(parsed, dict) and "filler_words" in parsed:
                filler_words = parsed["filler_words"]
            else:
                filler_words = []
                if isinstance(parsed, dict):
                    for key, value in parsed.items():
                        if key != "word_count" and isinstance(value, list):
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

        # Strengthen: add regex-detected hesitation sounds (um/uh/er/erm/ah/hmm) that GPT may have missed
        def _overlaps(span_start: int, span_end: int, existing: List[Dict[str, Any]]) -> bool:
            for f in existing:
                e_start, e_end = f["position"], f["position"] + f.get("length", 0)
                if not (span_end <= e_start or span_start >= e_end):
                    return True
            return False

        for m in HESITATION_REGEX.finditer(text):
            pos, length = m.start(), len(m.group(0))
            if not _overlaps(pos, pos + length, validated_fillers):
                validated_fillers.append({"word": m.group(0), "position": pos, "length": length})

        # Sort by position and remove overlaps (keep first occurrence when overlapping)
        validated_fillers = sorted(validated_fillers, key=lambda x: x["position"])
        non_overlapping = []
        last_end = -1
        for filler in validated_fillers:
            start = filler["position"]
            end = start + filler.get("length", len(filler.get("word", "")))
            if start >= last_end:
                non_overlapping.append(filler)
                last_end = end

        # Use GPT word_count if valid; otherwise fallback to local count
        if word_count_from_gpt is not None and word_count_from_gpt >= 0:
            word_count = word_count_from_gpt
        else:
            from services.wpm_calculation import count_words
            word_count = count_words(text)

        return (non_overlapping, word_count)

    except Exception as e:
        print(f"Error in GPT filler word detection: {str(e)}")
        from services.wpm_calculation import count_words
        return ([], count_words(text) if text else 0)


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
    Lenient: give benefit of the doubt; only NO when clearly off-topic.
    Returns True if relevant or unclear, False only when clearly unrelated.
    """
    if not title or not (user_text or "").strip():
        return True
    # Very short answers: don't penalize as off-topic (might be partial or misheard)
    if len((user_text or "").strip().split()) < 3:
        return True
    from config import get_openai_client, GPT_MODEL
    client = get_openai_client()
    prompt = f"""You are a fair judge. Give the speaker the benefit of the doubt.

Challenge question/topic:
"{title}"

User's spoken answer (transcribed, may have filler words or small errors):
"{user_text.strip()}"

Is this answer related to the question/topic?
Answer with exactly one word: YES or NO.

- YES if: the answer is about the same topic, or touches on it, or is a reasonable attempt, or you are unsure. Prefer YES when in doubt.
- NO only if: the answer is clearly about a completely different subject, or is only noise/filler with no relation to the topic."""
    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You answer only YES or NO. When in doubt, answer YES. No explanation."},
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

