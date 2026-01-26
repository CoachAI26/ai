"""
Confidence analysis service based on speech metrics
"""
from typing import Dict, Any


def calculate_confidence_score(
    wpm: float,
    filler_count: int,
    word_count: int,
    total_pauses: int,
    total_hesitations: int,
    pause_ratio: float,
    hesitation_rate: float,
    fluency_score: float
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
    # 1. WPM Score (Optimal range: 120-160 WPM)
    # Too slow (< 100) or too fast (> 200) reduces confidence
    if 120 <= wpm <= 160:
        wpm_score = 100.0
    elif 100 <= wpm < 120:
        wpm_score = 80.0 + ((wpm - 100) / 20) * 20  # 80-100
    elif 160 < wpm <= 200:
        wpm_score = 100.0 - ((wpm - 160) / 40) * 20  # 100-80
    elif wpm < 100:
        wpm_score = max(0, 80.0 - ((100 - wpm) / 50) * 40)  # 40-80
    else:  # wpm > 200
        wpm_score = max(0, 80.0 - ((wpm - 200) / 50) * 40)  # 40-80
    
    # 2. Filler Score (Lower is better)
    # Optimal: 0-2 fillers per 100 words
    fillers_per_100 = (filler_count / word_count * 100) if word_count > 0 else 0
    if fillers_per_100 <= 2:
        filler_score = 100.0
    elif fillers_per_100 <= 5:
        filler_score = 80.0 - ((fillers_per_100 - 2) / 3) * 20  # 80-60
    elif fillers_per_100 <= 10:
        filler_score = 60.0 - ((fillers_per_100 - 5) / 5) * 30  # 60-30
    else:
        filler_score = max(0, 30.0 - ((fillers_per_100 - 10) / 10) * 30)  # 30-0
    
    # 3. Pause Score (Lower pause ratio is better)
    # Optimal: < 10% pause time
    if pause_ratio <= 0.10:
        pause_score = 100.0
    elif pause_ratio <= 0.20:
        pause_score = 80.0 - ((pause_ratio - 0.10) / 0.10) * 30  # 80-50
    elif pause_ratio <= 0.30:
        pause_score = 50.0 - ((pause_ratio - 0.20) / 0.10) * 30  # 50-20
    else:
        pause_score = max(0, 20.0 - ((pause_ratio - 0.30) / 0.20) * 20)  # 20-0
    
    # 4. Hesitation Score (Lower hesitation rate is better)
    # Optimal: < 3 hesitations per 100 words
    if hesitation_rate <= 3:
        hesitation_score = 100.0
    elif hesitation_rate <= 6:
        hesitation_score = 80.0 - ((hesitation_rate - 3) / 3) * 20  # 80-60
    elif hesitation_rate <= 10:
        hesitation_score = 60.0 - ((hesitation_rate - 6) / 4) * 30  # 60-30
    else:
        hesitation_score = max(0, 30.0 - ((hesitation_rate - 10) / 10) * 30)  # 30-0
    
    # 5. Overall Confidence Score (weighted average)
    # Weights: WPM (25%), Filler (25%), Pause (20%), Hesitation (15%), Fluency (15%)
    confidence_score = (
        wpm_score * 0.25 +
        filler_score * 0.25 +
        pause_score * 0.20 +
        hesitation_score * 0.15 +
        fluency_score * 0.15
    )
    
    # Determine overall rating
    if confidence_score >= 85:
        overall_rating = "Excellent"
    elif confidence_score >= 70:
        overall_rating = "Good"
    elif confidence_score >= 55:
        overall_rating = "Moderate"
    elif confidence_score >= 40:
        overall_rating = "Low"
    else:
        overall_rating = "Very Low"
    
    # Generate recommendations
    recommendations = _generate_recommendations(
        wpm=wpm,
        wpm_score=wpm_score,
        filler_count=filler_count,
        fillers_per_100=fillers_per_100,
        pause_ratio=pause_ratio,
        hesitation_rate=hesitation_rate,
        total_pauses=total_pauses
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


def _generate_recommendations(
    wpm: float,
    wpm_score: float,
    filler_count: int,
    fillers_per_100: float,
    pause_ratio: float,
    hesitation_rate: float,
    total_pauses: int
) -> list:
    """
    Generate personalized recommendations based on metrics
    """
    recommendations = []
    
    # WPM recommendations
    if wpm < 100:
        recommendations.append("Try to speak slightly faster. Optimal speaking rate is 120-160 WPM.")
    elif wpm > 200:
        recommendations.append("Consider slowing down your speech for better clarity and comprehension.")
    elif wpm_score < 80:
        recommendations.append("Aim for a speaking rate between 120-160 WPM for optimal communication.")
    
    # Filler word recommendations
    if fillers_per_100 > 5:
        recommendations.append(f"Reduce filler words (currently {fillers_per_100:.1f} per 100 words). Practice pausing silently instead of using 'um' or 'uh'.")
    elif fillers_per_100 > 2:
        recommendations.append("You're doing well! Try to reduce filler words even further for more confident speech.")
    
    # Pause recommendations
    if pause_ratio > 0.20:
        recommendations.append(f"Reduce pauses (currently {pause_ratio*100:.1f}% of speaking time). Plan your thoughts before speaking.")
    elif pause_ratio > 0.10:
        recommendations.append("Consider reducing pause time slightly for more fluid speech.")
    
    # Hesitation recommendations
    if hesitation_rate > 6:
        recommendations.append(f"Work on reducing hesitations (currently {hesitation_rate:.1f} per 100 words). Practice speaking more smoothly.")
    elif hesitation_rate > 3:
        recommendations.append("Good progress! Continue working on reducing hesitation sounds.")
    
    # General recommendations
    if not recommendations:
        recommendations.append("Excellent! Your speech shows high confidence. Keep up the great work!")
    elif len(recommendations) == 1:
        recommendations.append("Overall, your speech is good. Focus on the area mentioned above.")
    
    return recommendations

