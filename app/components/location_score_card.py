"""Render a small location score card in Streamlit.

Provides a single function `render_score_card(result)` that takes the dict
returned by `models.predict.predict_point` and renders a compact UI: score,
component breakdown and a tiny bar chart. It will also call the TTS helper
from `app.audio.tts` if available and return any audio bytes path returned.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def render_score_card(result: dict):
    """Render score + breakdown and return any audio bytes/file path from TTS.

    result must be a mapping with keys: 'score', 'components', optional
    'confidence' and 'feature_snapshot'.
    """
    if result is None:
        st.info("No result to show")
        return None

    score = result.get("score")
    conf = result.get("confidence")
    comps = result.get("components", {})

    st.subheader(f"Livability Score: {score} / 100")

    if conf is None:
        st.info("Confidence: unknown")
    else:
        if conf < 0.7:
            st.warning(f"Confidence: {conf:.2f} (low — some inputs missing)")
        else:
            st.success(f"Confidence: {conf:.2f}")

    # Components table
    if comps:
        df = pd.DataFrame.from_dict(comps, orient="index", columns=["score"]) 
        st.table(df)

        # Small inline bar chart using matplotlib to avoid extra deps
        fig, ax = plt.subplots(figsize=(4, 2.6))
        try:
            df_sorted = df.sort_values("score")
            ax.barh(df_sorted.index.astype(str), df_sorted["score"].astype(float), color="#2b8cbe")
            ax.set_xlim(0, 100)
            ax.set_xlabel("component score")
            ax.set_title("Component breakdown")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception:
            # fallback: show nothing if plotting fails
            pass

    # Raw feature snapshot
    with st.expander("Raw feature snapshot"):
        raw = result.get("feature_snapshot", {})
        if raw:
            # show a short json-like preview
            st.json({k: raw.get(k) for k in list(raw)[:20]})
        else:
            st.write("No raw features available")

    # TTS: call the project's speak helper if present
    try:
        from app.audio.tts import speak_and_render

        audio_file = speak_and_render(result)
        if audio_file:
            st.audio(audio_file)
            return audio_file
    except Exception:
        # ignore missing tts or any tts errors
        return None

    return None
