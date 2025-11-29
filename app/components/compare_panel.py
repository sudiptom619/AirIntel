# app/components/compare_panel.py
import streamlit as st
from models.predict import predict_point
import pandas as pd

def compare_two(lat_a, lon_a, lat_b, lon_b):
    ra = predict_point(lat_a, lon_a)
    rb = predict_point(lat_b, lon_b)
    score_a, score_b = ra["score"], rb["score"]
    pct_diff = round((score_a - score_b) / max(score_b, 1) * 100, 1)
    # differences per component
    comps = {}
    for k in ra["components"].keys():
        a = ra["components"][k]
        b = rb["components"][k]
        comps[k] = {"A": a, "B": b, "diff": round(a - b, 2)}
    # UI: side-by-side
    st.markdown("### Comparison")
    col1, col2 = st.columns(2)
    col1.metric("Location A Score", f"{score_a}")
    col2.metric("Location B Score", f"{score_b}")
    st.table(pd.DataFrame(comps).T)
    # Build recommendation
    if score_a > score_b:
        recommend = f"Location A is better by {abs(pct_diff)}%."
    else:
        recommend = f"Location B is better by {abs(pct_diff)}%."
    st.success(recommend)
    # Build text for TTS
    text = f"Location A scored {score_a} and Location B scored {score_b}. {recommend}"
    return ra, rb, text
