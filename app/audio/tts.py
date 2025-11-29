# app/audio/tts.py
from gtts import gTTS
import tempfile
import os

def speak_text_to_file(text: str, lang: str = "en"):
    tts = gTTS(text, lang=lang)
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    tts.save(path)
    return path

def speak_and_render(result):
    # result can be dict from compare or single predict result
    if isinstance(result, dict) and "score" in result:
        text = f"This location has a livability score of {result['score']:.0f} out of 100. "
        comps = result.get("components", {})
        text += " ".join([f"{k} score is {v}." for k, v in comps.items()])
    else:
        text = str(result)
    path = speak_text_to_file(text)
    return path  # streamlit can accept a file path for st.audio
