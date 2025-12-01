# app/audio/tts.py
"""Text-to-speech module for accessibility. Generates MP3 files from text."""
from gtts import gTTS
import tempfile
import os
import math


def speak_text_to_file(text: str, lang: str = "en", slow: bool = False):
    """
    Convert text to speech and save to temporary MP3 file.
    
    Args:
        text: Text to convert to speech
        lang: Language code (default: "en" for English)
        slow: If True, speak more slowly (default: False)
    
    Returns:
        str: Path to generated MP3 file, or None if failed
    
    Raises:
        Exception: If gTTS fails or file operations fail
    """
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return None
    
    try:
        tts = gTTS(text=text, lang=lang, slow=slow)
        fd, path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        tts.save(path)
        return path
    except Exception as e:
        print(f"TTS Error: {e}")
        return None


def speak_and_render(result, use_short: bool = True):
    """
    Generate audio from a prediction result or comparison dict.
    
    Handles two types of inputs:
    1. Single location result (dict with 'score' key)
    2. Comparison result (dict with 'narration_short' or 'narration_detailed' keys)
    
    Args:
        result: Dict from predict_point() or compare_two()
        use_short: If True and comparison, use short narration; else use detailed
    
    Returns:
        str: Path to generated MP3 file, or None if failed
    """
    if result is None:
        return None
    
    # Determine which text to convert
    text = None
    
    # Check if it's a comparison result
    if "narration_short" in result:
        if use_short:
            text = result.get("narration_short")
        else:
            text = result.get("narration_detailed")
    
    # Otherwise, assume it's a single location result
    elif isinstance(result, dict) and "score" in result:
        score = result.get("score")
        comps = result.get("components", {})
        
        # Build narration from single location result
        text = f"This location has a livability score of {score:.0f} out of 100. "
        
        if comps:
            comp_texts = []
            for key, val in comps.items():
                comp_name = key.replace("_score", "").replace("_", " ").title()
                if val is not None:
                    try:
                        val_float = float(val)
                        if not (math.isnan(val_float) or math.isinf(val_float)):
                            comp_texts.append(f"{comp_name} score is {val_float:.0f}")
                    except (TypeError, ValueError):
                        pass
            
            if comp_texts:
                text += " ".join(comp_texts) + "."
        
        # Add confidence if available
        confidence = result.get("confidence")
        if confidence is not None:
            try:
                conf_pct = float(confidence) * 100
                if 0 <= conf_pct <= 100:
                    text += f" Confidence level is {conf_pct:.0f} percent."
            except (TypeError, ValueError):
                pass
    
    if not text:
        return None
    
    # Convert to speech
    return speak_text_to_file(text)


def get_audio_bytes(file_path):
    """
    Read audio file and return as bytes for Streamlit playback.
    
    Args:
        file_path: Path to MP3 file
    
    Returns:
        bytes: File contents, or None if file doesn't exist
    """
    if not file_path or not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, "rb") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None


def cleanup_audio_files(*file_paths):
    """
    Delete temporary audio files (optional cleanup).
    
    Args:
        *file_paths: One or more file paths to delete
    """
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Could not delete {path}: {e}")