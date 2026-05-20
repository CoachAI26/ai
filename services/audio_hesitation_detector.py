"""
Audio-based hesitation detection service.
Analyzes raw audio to detect "um", "uh", "er" patterns without relying on transcription.

Uses energy levels, silence patterns, and pitch analysis to identify likely hesitation sounds.
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def detect_hesitations_from_audio(audio_file_path: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect hesitation sounds directly from audio file using energy/silence analysis.
    Returns list of likely hesitation markers that can be injected into transcription.
    
    Args:
        audio_file_path: Path to audio file
        segments: Whisper segments with timing info (from verbose_json)
        
    Returns:
        List of likely hesitation locations with timing and confidence
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        logger.warning("librosa not installed; skipping audio-based hesitation detection")
        return []

    try:
        # Load audio file
        y, sr = librosa.load(audio_file_path, sr=None)
        
        # Compute short-time energy (indicates voiced vs unvoiced sounds)
        energies = librosa.feature.melspectrogram(y=y, sr=sr)
        energy_db = librosa.power_to_db(energies, ref=np.max)
        energy_mean = np.mean(energy_db, axis=0)
        
        # Find low-energy segments (potential hesitations: um/uh are often quieter)
        energy_threshold = np.percentile(energy_mean, 25)  # Bottom 25% = likely hesitations
        
        # Get hop length and frame times
        hop_length = 512  # default librosa hop
        frame_times = librosa.frames_to_time(np.arange(len(energy_mean)), sr=sr, hop_length=hop_length)
        
        # Find frames with low energy (hesitation zones)
        hesitation_frames = np.where(energy_mean < energy_threshold)[0]
        
        if len(hesitation_frames) == 0:
            return []
        
        # Convert frame indices to time ranges
        hesitation_times = frame_times[hesitation_frames]
        
        # Group consecutive hesitation frames into regions
        hesitation_regions = []
        if len(hesitation_times) > 0:
            current_region_start = hesitation_times[0]
            for i in range(1, len(hesitation_times)):
                time_gap = hesitation_times[i] - hesitation_times[i-1]
                if time_gap > 0.1:  # New region if gap > 0.1s
                    if hesitation_times[i-1] - current_region_start > 0.08:  # Only if region > 80ms
                        hesitation_regions.append({
                            "start": current_region_start,
                            "end": hesitation_times[i-1],
                            "duration": hesitation_times[i-1] - current_region_start,
                            "type": "low_energy"
                        })
                    current_region_start = hesitation_times[i]
            # Add last region
            if hesitation_times[-1] - current_region_start > 0.08:
                hesitation_regions.append({
                    "start": current_region_start,
                    "end": hesitation_times[-1],
                    "duration": hesitation_times[-1] - current_region_start,
                    "type": "low_energy"
                })
        
        # Also detect silence/pauses between speech segments (gaps > 0.2s often have hesitations)
        silence_regions = []
        if segments and len(segments) > 1:
            for i in range(len(segments) - 1):
                current_end = segments[i].get("end") if isinstance(segments[i], dict) else getattr(segments[i], "end", None)
                next_start = segments[i+1].get("start") if isinstance(segments[i+1], dict) else getattr(segments[i+1], "start", None)
                
                if current_end is not None and next_start is not None:
                    gap = next_start - current_end
                    if 0.2 < gap < 1.0:  # Meaningful gap that might contain hesitation
                        silence_regions.append({
                            "start": current_end,
                            "end": next_start,
                            "duration": gap,
                            "type": "inter_segment_gap",
                            "segment_before": i
                        })
        
        logger.info(f"Audio hesitation analysis: found {len(hesitation_regions)} low-energy regions, {len(silence_regions)} inter-segment gaps")
        
        return hesitation_regions + silence_regions
        
    except Exception as e:
        logger.warning(f"Audio hesitation detection failed: {e}")
        return []


def inject_hesitations_into_text(
    text: str,
    segments: List[Dict[str, Any]],
    hesitation_regions: List[Dict[str, Any]]
) -> str:
    """
    Inject hesitation markers ("um ") into transcribed text based on detected audio regions.
    
    Strategy:
    1. For each detected hesitation region, find the corresponding segment boundary
    2. Inject "um " after the word that ends near the hesitation time
    3. Or inject before the segment that starts after a detected gap
    
    Args:
        text: Original transcribed text
        segments: Whisper segments with timing
        hesitation_regions: List of detected hesitation zones from audio
        
    Returns:
        Text with injected hesitation markers
    """
    if not hesitation_regions or not segments:
        return text
    
    try:
        def _um_marker_at(pos: int) -> str:
            prefix = "" if pos <= 0 or text[pos - 1].isspace() else " "
            suffix = "" if pos >= len(text) or text[pos].isspace() else " "
            return f"{prefix}um{suffix}"

        # Build a map of segment boundaries to text positions
        segment_map = []
        text_pos = 0
        for seg in segments:
            seg_start = seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", None)
            seg_end = seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", None)
            seg_text = seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")
            
            if seg_text:
                text_pos_start = text.find(seg_text, text_pos)
                if text_pos_start >= 0:
                    text_pos_end = text_pos_start + len(seg_text)
                    segment_map.append({
                        "text": seg_text,
                        "start": seg_start,
                        "end": seg_end,
                        "text_start": text_pos_start,
                        "text_end": text_pos_end
                    })
                    text_pos = text_pos_end
        
        # For each hesitation, find where to inject it
        injections = []
        if segment_map:
            first_seg = segment_map[0]
            first_start = first_seg.get("start")
            first_text_start = first_seg.get("text_start", 0)
        else:
            first_start = None
            first_text_start = 0

        for hes in hesitation_regions:
            hes_time = hes["start"]
            hes_type = hes.get("type")

            # If a hesitation happens before the first recognized speech segment,
            # inject it at the beginning. This catches cases where Whisper drops an
            # opening "um/uh" before the first real words.
            if (
                first_start is not None
                and hes.get("end") is not None
                and hes["end"] <= first_start + 0.15
                and hes.get("duration", 0) >= 0.12
            ):
                injections.append((first_text_start, _um_marker_at(first_text_start)))
                continue

            # Low-energy regions inside speech are too noisy to treat as fillers.
            # Use them only for the opening case above; normal injections should
            # come from actual gaps between Whisper segments.
            if hes_type != "inter_segment_gap":
                continue
            
            # Find the segment that contains or precedes this hesitation time
            relevant_seg = None
            for seg in segment_map:
                if seg["end"] is not None and hes_time >= seg["end"] - 0.1:
                    relevant_seg = seg
            
            if relevant_seg:
                # Inject at a segment boundary while preserving word spacing.
                inject_pos = relevant_seg["text_end"]
                injections.append((inject_pos, _um_marker_at(inject_pos)))
        
        # Apply injections in reverse order to maintain position consistency
        result = text
        deduped_injections = []
        for pos, marker in sorted(injections, key=lambda item: item[0]):
            if deduped_injections and abs(pos - deduped_injections[-1][0]) < 3:
                continue
            deduped_injections.append((pos, marker))

        for pos, marker in sorted(deduped_injections, reverse=True):
            if 0 <= pos <= len(result):
                result = result[:pos] + marker + result[pos:]
        
        logger.info(f"Injected {len(deduped_injections)} hesitation markers into text")
        return result
        
    except Exception as e:
        logger.warning(f"Hesitation injection failed: {e}")
        return text
