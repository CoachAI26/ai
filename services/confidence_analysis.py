"""
Confidence analysis service based on speech metrics
"""
import json
import re
from typing import Dict, Any, Optional
from config import get_openai_client, GPT_MODEL, GPT_TEMPERATURE
from scoring_config import (
    WPM_OPTIMAL_MIN,
    WPM_OPTIMAL_MAX,
    FILLERS_PER_100_FOR_FULL_SCORE,
    PAUSE_RATIO_FOR_FULL_SCORE,
    HESITATION_RATE_FOR_FULL_SCORE,
    USE_FILLER_COUNT_IN_HESITATION,
    FILLER_WEIGHT_IN_HESITATION,
    RATING_EXCELLENT_MIN,
    RATING_GOOD_MIN,
    RATING_MODERATE_MIN,
    RATING_LOW_MIN,
    WEIGHT_WPM,
    WEIGHT_FILLER,
    WEIGHT_PAUSE,
    WEIGHT_HESITATION,
    WEIGHT_FLUENCY,
)


async def calculate_confidence_score(
    wpm: float,
    filler_count: int,
    word_count: int,
    total_pauses: int,
    total_hesitations: int,
    pause_ratio: float,
    hesitation_rate: float,
    fluency_score: float,
    level: Optional[str] = None,
    category: Optional[str] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate confidence score based on multiple speech metrics
    
    Args:
        wpm: Words per minute
        filler_count: Number of filler words
        word_count: Total word count
        total_pauses: Number of pauses
        total_hesitations: Number of hesitation sounds
        pause_ratio: Ratio of pause time to total time
        hesitation_rate: Hesitations per 100 words
        fluency_score: Fluency score (0-100)
        
    Returns:
        Dictionary with confidence metrics:
        - confidence_score: Overall confidence score (0-100)
        - wpm_score: WPM component score (0-100)
        - filler_score: Filler words component score (0-100)
        - pause_score: Pause component score (0-100)
        - hesitation_score: Hesitation component score (0-100)
        - overall_rating: Text rating (Very Low, Low, Moderate, Good, Excellent)
        - recommendations: List of improvement recommendations
    """
    # 1. WPM Score — configurable optimal band (narrower = stricter)
    w_min, w_max = WPM_OPTIMAL_MIN, WPM_OPTIMAL_MAX
    band = max(1, (w_max - w_min) / 2)  # falloff band below/above optimal
    if w_min <= wpm <= w_max:
        wpm_score = 100.0
    elif w_min - band <= wpm < w_min:
        wpm_score = 70.0 + ((wpm - (w_min - band)) / band) * 30
    elif w_max < wpm <= w_max + band:
        wpm_score = 100.0 - ((wpm - w_max) / band) * 30
    elif wpm < w_min - band:
        if wpm >= 100 and (w_min - band) > 100:
            wpm_score = 50.0 + (wpm - 100) / ((w_min - band) - 100) * 20
        elif wpm < 100:
            wpm_score = max(0, 50.0 - ((100 - wpm) / 40) * 50)
        else:
            wpm_score = 50.0
    else:
        wpm_score = max(0, 30.0 - ((wpm - (w_max + band)) / 30) * 30)

    # 2. Filler Score — full score only when fillers per 100 <= config threshold
    fillers_per_100 = (filler_count / word_count * 100) if word_count > 0 else 0
    fillers_per_100_value = fillers_per_100
    t = FILLERS_PER_100_FOR_FULL_SCORE
    if fillers_per_100 <= t:
        filler_score = 100.0
    elif fillers_per_100 <= t + 1.5:
        filler_score = 75.0 - ((fillers_per_100 - t) / 1.5) * 25
    elif fillers_per_100 <= t + 4:
        filler_score = 50.0 - ((fillers_per_100 - t - 1.5) / 2.5) * 25
    elif fillers_per_100 <= t + 7:
        filler_score = 25.0 - ((fillers_per_100 - t - 4) / 3) * 20
    else:
        filler_score = max(0, 5.0 - (fillers_per_100 - t - 7) * 0.5)

    # 3. Pause Score — full score only when pause_ratio <= config threshold
    p_full = PAUSE_RATIO_FOR_FULL_SCORE
    if pause_ratio <= p_full:
        pause_score = 100.0
    elif pause_ratio <= p_full + 0.05:
        pause_score = 80.0 - ((pause_ratio - p_full) / 0.05) * 30
    elif pause_ratio <= p_full + 0.13:
        pause_score = 50.0 - ((pause_ratio - p_full - 0.05) / 0.08) * 35
    elif pause_ratio <= p_full + 0.23:
        pause_score = 15.0 - ((pause_ratio - p_full - 0.13) / 0.10) * 15
    else:
        pause_score = 0.0

    # 4. Hesitation Score — use effective rate (regex + filler_count) when config enabled
    if USE_FILLER_COUNT_IN_HESITATION and word_count > 0:
        effective_hesitations = total_hesitations + FILLER_WEIGHT_IN_HESITATION * filler_count
        hesitation_rate_used = effective_hesitations / word_count * 100
    else:
        hesitation_rate_used = hesitation_rate
    h_full = HESITATION_RATE_FOR_FULL_SCORE
    if hesitation_rate_used <= h_full:
        hesitation_score = 100.0
    elif hesitation_rate_used <= h_full + 1.5:
        hesitation_score = 80.0 - ((hesitation_rate_used - h_full) / 1.5) * 30
    elif hesitation_rate_used <= h_full + 4.5:
        hesitation_score = 50.0 - ((hesitation_rate_used - h_full - 1.5) / 3) * 30
    elif hesitation_rate_used <= h_full + 8.5:
        hesitation_score = 20.0 - ((hesitation_rate_used - h_full - 4.5) / 4) * 20
    else:
        hesitation_score = 0.0

    # 5. Overall Confidence Score (weighted average from config)
    confidence_score = (
        wpm_score * WEIGHT_WPM +
        filler_score * WEIGHT_FILLER +
        pause_score * WEIGHT_PAUSE +
        hesitation_score * WEIGHT_HESITATION +
        fluency_score * WEIGHT_FLUENCY
    )

    # Overall rating — configurable bands
    if confidence_score >= RATING_EXCELLENT_MIN:
        overall_rating = "Excellent"
    elif confidence_score >= RATING_GOOD_MIN:
        overall_rating = "Good"
    elif confidence_score >= RATING_MODERATE_MIN:
        overall_rating = "Moderate"
    elif confidence_score >= RATING_LOW_MIN:
        overall_rating = "Low"
    else:
        overall_rating = "Very Low"
    
    # Generate recommendations using GPT
    recommendations = await _generate_recommendations_with_gpt(
        wpm=wpm,
        wpm_score=wpm_score,
        filler_count=filler_count,
        fillers_per_100=fillers_per_100_value,
        pause_ratio=pause_ratio,
        hesitation_rate=hesitation_rate,
        total_pauses=total_pauses,
        total_hesitations=total_hesitations,
        confidence_score=confidence_score,
        overall_rating=overall_rating,
        level=level,
        category=category,
        title=title,
    )
    
    return {
        "confidence_score": round(confidence_score, 2),
        "wpm_score": round(wpm_score, 2),
        "filler_score": round(filler_score, 2),
        "pause_score": round(pause_score, 2),
        "hesitation_score": round(hesitation_score, 2),
        "overall_rating": overall_rating,
        "recommendations": recommendations
    }


async def _generate_recommendations_with_gpt(
    wpm: float,
    wpm_score: float,
    filler_count: int,
    fillers_per_100: float,
    pause_ratio: float,
    hesitation_rate: float,
    total_pauses: int,
    total_hesitations: int,
    confidence_score: float,
    overall_rating: str,
    level: Optional[str] = None,
    category: Optional[str] = None,
    title: Optional[str] = None,
) -> list:
    """
    Generate personalized recommendations using GPT based on speech metrics
    """
    challenge_context = ""
    if level or category or title:
        challenge_context = "\nChallenge context:\n"
        if level:
            challenge_context += f"- Level: {level}\n"
        if category:
            challenge_context += f"- Category: {category}\n"
        if title:
            challenge_context += f"- Title: {title}\n"

    prompt = f"""You are an expert speech coach analyzing a person's speaking performance. Based on the following metrics, provide personalized, actionable recommendations to improve their speech confidence and fluency.{challenge_context}

Speech Metrics:
- Words Per Minute (WPM): {wpm:.2f} (Score: {wpm_score:.2f}/100)
- Filler Words: {filler_count} total ({fillers_per_100:.2f} per 100 words)
- Pauses: {total_pauses} pauses ({pause_ratio*100:.1f}% of speaking time)
- Hesitations: {total_hesitations} hesitation sounds ({hesitation_rate:.2f} per 100 words)
- Overall Confidence Score: {confidence_score:.2f}/100
- Overall Rating: {overall_rating}

Provide 2-4 specific, actionable recommendations that:
1. Are personalized based on the actual metrics
2. Are encouraging and constructive
3. Focus on the areas that need the most improvement
4. Provide concrete steps or techniques
5. Are written in a friendly, supportive tone

Return your response as a JSON object with this exact format:
{{
  "recommendations": [
    "First recommendation here",
    "Second recommendation here",
    "Third recommendation here"
  ]
}}

If the performance is excellent (confidence score >= 90), provide 1-2 positive reinforcement messages.
If there are multiple areas to improve, prioritize the most impactful ones.

Recommendations:"""

    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert speech coach. Provide personalized, actionable recommendations for improving speech confidence. Always return valid JSON only, no additional text."
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
        
        try:
            parsed = json.loads(response_content)
            if isinstance(parsed, dict) and "recommendations" in parsed:
                recommendations = parsed["recommendations"]
                if isinstance(recommendations, list):
                    # Validate and clean recommendations
                    cleaned_recommendations = []
                    for rec in recommendations:
                        if isinstance(rec, str) and len(rec.strip()) > 0:
                            cleaned_recommendations.append(rec.strip())
                    return cleaned_recommendations if cleaned_recommendations else ["Keep practicing to improve your speech confidence!"]
        except json.JSONDecodeError:
            # Fallback: try to extract recommendations from text
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
                if "recommendations" in parsed:
                    return parsed["recommendations"]
        
        # Last resort fallback
        return ["Keep practicing to improve your speech confidence!"]
    
    except Exception as e:
        print(f"Error in GPT recommendations generation: {str(e)}")
        # Fallback to basic recommendation
        return ["Keep practicing to improve your speech confidence!"]

