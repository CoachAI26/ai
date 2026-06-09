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


# Cache which param name the OpenAI client accepts (old client = max_tokens, new = max_completion_tokens)
_max_tokens_param: Optional[str] = None


def _max_tokens_kwargs(n: int) -> dict:
    """Return kwargs for token limit: works with both old (max_tokens) and new (max_completion_tokens) OpenAI client."""
    global _max_tokens_param
    if _max_tokens_param is not None:
        return {_max_tokens_param: n}
    if GPT_MODEL.startswith(("gpt-5", "o1", "o3", "o4")):
        return {}
    import inspect
    client = get_openai_client()
    sig = inspect.signature(client.chat.completions.create)
    if "max_completion_tokens" in sig.parameters:
        _max_tokens_param = "max_completion_tokens"
    else:
        _max_tokens_param = "max_tokens"
    return {_max_tokens_param: n}


# Regex pattern to catch hesitation sounds — comprehensive list with multiple variations
HESITATION_REGEX = re.compile(
    r"\b("
    r"mm-hmm|uh-huh|um-um|uh-uh|ah-ha|eh-eh|"
    r"u+h+|um+|uh+|erm+|err+|er+|ahh+|ah+|hmm+|hm+|mhm+|mmm+|mm+|"
    r"huh+|eh+m+|eh+|em+|euh+|ew+|ur+"
    r")\b",
    re.IGNORECASE,
)


def _span_overlaps(span_start: int, span_end: int, existing: List[Dict[str, Any]]) -> bool:
    for f in existing:
        e_start = int(f.get("position", 0))
        e_end = e_start + int(f.get("length", 0))
        if not (span_end <= e_start or span_start >= e_end):
            return True
    return False


def _detect_hesitation_sounds_locally(text: str) -> List[Dict[str, Any]]:
    """
    Deterministic detection for explicit hesitation sounds in the transcript.
    GPT can still add contextual fillers later, but these should never depend on GPT.
    """
    if not text:
        return []
    fillers: List[Dict[str, Any]] = []
    for match in HESITATION_REGEX.finditer(text):
        word = match.group(0)
        fillers.append({"word": word, "position": match.start(), "length": len(word)})
    return fillers


GPT_FILLER_BLOCKLIST = {
    "i think",
    "i guess",
}


def _should_accept_gpt_filler(word: str) -> bool:
    normalized = re.sub(r"\s+", " ", word.strip().lower())
    return normalized not in GPT_FILLER_BLOCKLIST

CONTEXTUAL_FILLER_PATTERNS = [
    r"\byou\s+know\b",
    r"\bi\s+mean\b",
    r"\bsort\s+of\b",
    r"\bkind\s+of\b",
    r"\blet\s+me\s+see\b",
    r"\byou\s+see\b",
    r"\bfor\s+sure\b",
    r"\blike\b",
    r"\bactually\b",
    r"\bbasically\b",
    r"\bliterally\b",
    r"\bwell\b",
    r"\bright\b",
    r"\byeah\b",
    r"\bokay\b",
    r"\bjust\b",
    r"\bso\b",
    r"\banyway\b",
]


def _candidate_context(text: str, start: int, end: int, radius: int = 45) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return text[left:right]


def _detect_contextual_candidates(text: str, existing: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Find exact candidate spans in the transcript. GPT may classify these spans,
    but it is not allowed to invent new filler positions.
    """
    candidates: List[Dict[str, Any]] = []
    if not text:
        return candidates

    for pattern in CONTEXTUAL_FILLER_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start, end = match.span()
            if _span_overlaps(start, end, existing) or _span_overlaps(start, end, candidates):
                continue
            word = text[start:end]
            if not _should_accept_gpt_filler(word):
                continue
            candidates.append({
                "id": len(candidates),
                "word": word,
                "position": start,
                "length": end - start,
                "context": _candidate_context(text, start, end),
            })

    # Repeated adjacent words caused by stutter: "I I", "the the".
    repeated_word_pattern = re.compile(r"\b([A-Za-z][A-Za-z']*)\b([\s,.;:!?]+)\1\b", re.IGNORECASE)
    for match in repeated_word_pattern.finditer(text):
        start, end = match.start(1), match.end(1)
        if _span_overlaps(start, end, existing) or _span_overlaps(start, end, candidates):
            continue
        candidates.append({
            "id": len(candidates),
            "word": text[start:end],
            "position": start,
            "length": end - start,
            "context": _candidate_context(text, start, end),
        })

    return sorted(candidates, key=lambda item: item["position"])


FILLER_CLASSIFICATION_PROMPT = """Classify only the provided candidate spans as filler or not filler.

Rules:
- Accept a candidate only if the exact candidate words are used to stall, hesitate, restart, or buy thinking time.
- Reject normal meaning/content uses, discourse structure, emphasis, comparison, agreement, or grammar.
- Do not add new fillers. Do not change positions. Only return candidate IDs from the provided list.
- When unsure, reject the candidate. Precision is more important than guessing.

Return pure JSON:
{"accepted_ids": [0, 2]}
"""


async def detect_filler_words_with_gpt(text: str) -> Tuple[List[Dict[str, Any]], int]:
    """
    Detect filler words and get word count.

    Explicit hesitation sounds are detected deterministically from the transcript.
    GPT is only allowed to classify exact candidate spans that already exist in
    the transcript, so it cannot invent filler positions.

    Returns:
        (list of filler words with positions/lengths, deterministic word_count)
    """
    from services.wpm_calculation import count_words

    local_hesitations = _detect_hesitation_sounds_locally(text)
    word_count = count_words(text)

    try:
        candidates = _detect_contextual_candidates(text, local_hesitations)

        validated_fillers = list(local_hesitations)
        if candidates:
            client = get_openai_client()
            response = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You classify candidate transcript spans as speech fillers. Return JSON only. Never add candidates.",
                    },
                    {
                        "role": "user",
                        "content": (
                            FILLER_CLASSIFICATION_PROMPT
                            + "\nTranscript:\n"
                            + text
                            + "\n\nCandidates:\n"
                            + json.dumps(candidates, ensure_ascii=False)
                        ),
                    },
                ],
                temperature=GPT_TEMPERATURE,
                response_format={"type": "json_object"},
            )
            response_content = response.choices[0].message.content.strip()
            parsed = json.loads(response_content)
            accepted_ids = parsed.get("accepted_ids", []) if isinstance(parsed, dict) else []
            accepted_ids = {int(item) for item in accepted_ids if isinstance(item, (int, str)) and str(item).isdigit()}

            for candidate in candidates:
                if candidate["id"] not in accepted_ids:
                    continue
                position = int(candidate["position"])
                length = int(candidate["length"])
                actual_word = text[position:position + length]
                if (
                    actual_word == candidate["word"]
                    and _should_accept_gpt_filler(actual_word)
                    and not _span_overlaps(position, position + length, validated_fillers)
                ):
                    validated_fillers.append({
                        "word": actual_word,
                        "position": position,
                        "length": length,
                    })

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

        return (non_overlapping, word_count)

    except Exception as e:
        print(f"Error in GPT filler word detection: {str(e)}")
        return (local_hesitations, word_count)


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

        # Remove punctuation attached only to the filler, e.g. "Um," or "Mm.".
        while end < len(result) and result[end] in ",.;:!?":
            end += 1

        result = result[:start] + result[end:]
        # Clean up spaces around punctuation and collapse whitespace
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'^\s*[,.!?;:]\s*', '', result)
        result = re.sub(r'\s+([,.;:!?])', r'\1', result)   # space before punctuation
        result = re.sub(r'([,.;:!?])\s+', r'\1 ', result)  # normalize after punctuation
        result = re.sub(r'([,.;:!?]){2,}', r'\1', result)
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
            **_max_tokens_kwargs(2000),
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
OFF_TOPIC_ALERT_CODE = "OFF_TOPIC_CHALLENGE"


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
            **_max_tokens_kwargs(10),
        )
        raw = (response.choices[0].message.content or "").strip().upper()
        return raw.startswith("YES")
    except Exception as e:
        print(f"Error checking relevance: {str(e)}")
        return True  # On error, do not penalize
