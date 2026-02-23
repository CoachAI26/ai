"""
Central scoring parameters for speech analysis.
Tune these to make scores stricter so "bad" speech does not get 100 everywhere.
"""

# -----------------------------------------------------------------------------
# Pause detection (pause_analysis.py)
# -----------------------------------------------------------------------------
# Minimum gap between Whisper segments (seconds) to count as a pause.
# Lower = more short hesitations count as pauses (stricter).
PAUSE_THRESHOLD_SEC = 0.35

# -----------------------------------------------------------------------------
# WPM score (confidence_analysis.py)
# -----------------------------------------------------------------------------
# Only this range gets full 100. Narrower = harder to get full WPM score.
WPM_OPTIMAL_MIN = 135
WPM_OPTIMAL_MAX = 145

# -----------------------------------------------------------------------------
# Filler score (confidence_analysis.py)
# -----------------------------------------------------------------------------
# Full score only when fillers per 100 words <= this. Lower = stricter.
FILLERS_PER_100_FOR_FULL_SCORE = 0.5

# -----------------------------------------------------------------------------
# Pause ratio score (confidence_analysis.py)
# -----------------------------------------------------------------------------
# Full score only when pause_ratio <= this. Lower = stricter (e.g. 0.03 = 3%).
PAUSE_RATIO_FOR_FULL_SCORE = 0.03

# -----------------------------------------------------------------------------
# Hesitation score (confidence_analysis.py)
# -----------------------------------------------------------------------------
# Hesitation = regex (um/uh/er...) + optionally filler_count for "like/you know".
# Full score only when effective rate per 100 words <= this.
HESITATION_RATE_FOR_FULL_SCORE = 0.8

# Hesitation = from GPT only (total_hesitations already = len(filler_words)). No extra filler weighting.
USE_FILLER_COUNT_IN_HESITATION = False
FILLER_WEIGHT_IN_HESITATION = 0.6

# -----------------------------------------------------------------------------
# Fluency score (pause_analysis.py)
# -----------------------------------------------------------------------------
# Penalty = ratio * PER_RATIO (e.g. 5% pause -> 0.05 * 80 = 4 pt). Higher = stricter.
FLUENCY_PAUSE_PENALTY_PER_RATIO = 80.0
FLUENCY_PAUSE_PENALTY_CAP = 55.0
FLUENCY_HESITATION_PENALTY_PER_RATE = 1.2
FLUENCY_HESITATION_PENALTY_CAP = 35.0

# -----------------------------------------------------------------------------
# Overall confidence rating bands (confidence_analysis.py)
# -----------------------------------------------------------------------------
RATING_EXCELLENT_MIN = 90
RATING_GOOD_MIN = 75
RATING_MODERATE_MIN = 58
RATING_LOW_MIN = 42
# Below RATING_LOW_MIN = "Very Low"

# -----------------------------------------------------------------------------
# Confidence score weights (must sum to 1.0)
# -----------------------------------------------------------------------------
WEIGHT_WPM = 0.25
WEIGHT_FILLER = 0.25
WEIGHT_PAUSE = 0.20
WEIGHT_HESITATION = 0.15
WEIGHT_FLUENCY = 0.15
