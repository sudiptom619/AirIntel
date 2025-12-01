# app/audio/__init__.py
"""Audio/TTS components for accessibility."""

from .tts import speak_text_to_file, speak_and_render, get_audio_bytes, cleanup_audio_files

__all__ = [
    'speak_text_to_file',
    'speak_and_render',
    'get_audio_bytes',
    'cleanup_audio_files',
]
