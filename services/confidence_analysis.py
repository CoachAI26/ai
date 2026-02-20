"""
Confidence analysis service based on speech metrics
"""
import json
import re
from typing import Dict, Any, Optional
from config import get_openai_client, GPT_MODEL, GPT_TEMPERATURE


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
    # 1. WPM Score — STRICTER: narrow optimal range (130–150), harsher penalties outside
    # Optimal: 130-150 WPM only gets full score
    if 130 <= wpm <= 150:
        wpm_score = 100.0
    elif 115 <= wpm < 130:
        wpm_score = 70.0 + ((wpm - 115) / 15) * 30  # 70-100
    elif 150 < wpm <= 165:
        wpm_score = 100.0 - ((wpm - 150) / 15) * 30  # 100-70
    elif 100 <= wpm < 115:
        wpm_score = 50.0 + ((wpm - 100) / 15) * 20  # 50-70
    elif 165 < wpm <= 185:
        wpm_score = 70.0 - ((wpm - 165) / 20) * 40  # 70-30
    elif wpm < 100:
        wpm_score = max(0, 50.0 - ((100 - wpm) / 40) * 50)  # 0-50
    else:  # wpm > 185
        wpm_score = max(0, 30.0 - ((wpm - 185) / 30) * 30)  # 0-30

    # 2. Filler Score — STRICTER: only 0–1 per 100 words gets full score
    fillers_per_100 = (filler_count / word_count * 100) if word_count > 0 else 0
    fillers_per_100_value = fillers_per_100
    if fillers_per_100 <= 1:
        filler_score = 100.0
    elif fillers_per_100 <= 2.5:
        filler_score = 75.0 - ((fillers_per_100 - 1) / 1.5) * 25  # 75-50
    elif fillers_per_100 <= 5:
        filler_score = 50.0 - ((fillers_per_100 - 2.5) / 2.5) * 25  # 50-25
    elif fillers_per_100 <= 8:
        filler_score = 25.0 - ((fillers_per_100 - 5) / 3) * 20  # 25-5
    else:
        filler_score = max(0, 5.0 - (fillers_per_100 - 8) * 0.5)  # 5-0

    # 3. Pause Score — STRICTER: only < 5% pause time gets full score
    if pause_ratio <= 0.05:
        pause_score = 100.0
    elif pause_ratio <= 0.10:
        pause_score = 80.0 - ((pause_ratio - 0.05) / 0.05) * 30  # 80-50
    elif pause_ratio <= 0.18:
        pause_score = 50.0 - ((pause_ratio - 0.10) / 0.08) * 35  # 50-15
    elif pause_ratio <= 0.28:
        pause_score = 15.0 - ((pause_ratio - 0.18) / 0.10) * 15  # 15-0
    else:
        pause_score = 0.0

    # 4. Hesitation Score — STRICTER: only 0–1.5 per 100 words gets full score
    if hesitation_rate <= 1.5:
        hesitation_score = 100.0
    elif hesitation_rate <= 3:
        hesitation_score = 80.0 - ((hesitation_rate - 1.5) / 1.5) * 30  # 80-50
    elif hesitation_rate <= 6:
        hesitation_score = 50.0 - ((hesitation_rate - 3) / 3) * 30  # 50-20
    elif hesitation_rate <= 10:
        hesitation_score = 20.0 - ((hesitation_rate - 6) / 4) * 20  # 20-0
    else:
        hesitation_score = 0.0

    # 5. Overall Confidence Score (weighted average)
    confidence_score = (
        wpm_score * 0.25 +
        filler_score * 0.25 +
        pause_score * 0.20 +
        hesitation_score * 0.15 +
        fluency_score * 0.15
    )

    # Overall rating — STRICTER thresholds (harder to get Excellent/Good)
    if confidence_score >= 90:
        overall_rating = "Excellent"
    elif confidence_score >= 75:
        overall_rating = "Good"
    elif confidence_score >= 58:
        overall_rating = "Moderate"
    elif confidence_score >= 42:
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

