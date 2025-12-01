# app/components/compare_panel.py
"""Enhanced comparison panel with structured output for TTS and visual display."""
import streamlit as st
from models.predict import predict_point
import pandas as pd
import math


def safe_float(val):
    """Convert value to float, return None if invalid."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def format_for_tts(text):
    """Format text for clear TTS narration (expand abbreviations, etc)."""
    # Replace common abbreviations
    replacements = {
        "pm25": "PM2.5",
        "pm10": "PM10",
        "ndvi": "vegetation index",
        "μg/m³": "micrograms per cubic meter",
        "%": "percent",
    }
    for abbr, full in replacements.items():
        text = text.replace(abbr, full)
    return text


def compare_two(lat_a, lon_a, lat_b, lon_b):
    """
    Compare two locations and return structured result.
    
    Returns:
        dict with keys:
        - location_a, location_b: prediction dicts
        - comparison: {score_diff, score_pct_diff, better_location, component_diffs}
        - narration_short: brief TTS-friendly summary
        - narration_detailed: full breakdown
    """
    # Predict for both locations
    ra = predict_point(lat_a, lon_a)
    rb = predict_point(lat_b, lon_b)
    
    score_a = safe_float(ra.get("score"))
    score_b = safe_float(rb.get("score"))
    
    if score_a is None or score_b is None:
        return {
            "error": "Could not compute scores for one or both locations",
            "location_a": ra,
            "location_b": rb,
        }
    
    # Compute score difference
    score_diff = score_a - score_b
    pct_diff = round((score_a - score_b) / max(abs(score_b), 1) * 100, 1)
    better_location = "A" if score_a > score_b else ("B" if score_b > score_a else "tied")
    
    # Component-level differences
    comps_a = ra.get("components", {})
    comps_b = rb.get("components", {})
    
    component_diffs = {}
    for key in set(list(comps_a.keys()) + list(comps_b.keys())):
        val_a = safe_float(comps_a.get(key))
        val_b = safe_float(comps_b.get(key))
        
        if val_a is not None and val_b is not None:
            diff = val_a - val_b
            pct = round((diff / max(abs(val_b), 1)) * 100, 1) if val_b != 0 else 0
            component_diffs[key] = {
                "a": round(val_a, 2),
                "b": round(val_b, 2),
                "diff": round(diff, 2),
                "pct": pct,
                "better": "A" if diff > 0 else ("B" if diff < 0 else "tied"),
            }
    
    # Build narrations
    if better_location == "tied":
        rec_text = "Both locations have equal livability scores."
    else:
        rec_text = f"Location {better_location} is better by {abs(pct_diff)}%."
    
    narration_short = (
        f"Location A scored {score_a:.1f} out of 100. "
        f"Location B scored {score_b:.1f} out of 100. "
        f"{rec_text}"
    )
    
    # Build detailed narration
    narration_detailed = narration_short + "\n\n"
    narration_detailed += "Component breakdown:\n"
    for comp_name, comp_data in component_diffs.items():
        comp_label = comp_name.replace("_score", "").replace("_", " ").title()
        narration_detailed += (
            f"{comp_label}: Location A scored {comp_data['a']}, "
            f"Location B scored {comp_data['b']}. "
        )
        if comp_data["pct"] != 0:
            better = comp_data["better"]
            narration_detailed += f"Location {better} is {abs(comp_data['pct'])}% better.\n"
        else:
            narration_detailed += "Equal.\n"
    
    # Confidence summary
    conf_a = safe_float(ra.get("confidence"))
    conf_b = safe_float(rb.get("confidence"))
    if conf_a is not None or conf_b is not None:
        narration_detailed += f"\nConfidence: Location A {conf_a:.2%}, Location B {conf_b:.2%}."
    
    # Format for TTS
    narration_short_tts = format_for_tts(narration_short)
    narration_detailed_tts = format_for_tts(narration_detailed)
    
    return {
        "location_a": ra,
        "location_b": rb,
        "comparison": {
            "score_a": score_a,
            "score_b": score_b,
            "score_diff": score_diff,
            "score_pct_diff": pct_diff,
            "better_location": better_location,
            "component_diffs": component_diffs,
        },
        "recommendation": rec_text,
        "narration_short": narration_short_tts,
        "narration_detailed": narration_detailed_tts,
        "error": None,
    }


def render_comparison_ui(comparison_result):
    """
    Render the comparison UI in Streamlit.
    
    Args:
        comparison_result: dict returned by compare_two()
    """
    if comparison_result.get("error"):
        st.error(f"❌ {comparison_result['error']}")
        return
    
    comp = comparison_result["comparison"]
    score_a = comp["score_a"]
    score_b = comp["score_b"]
    better = comp["better_location"]
    
    # Side-by-side score cards
    st.markdown("### 📊 Overall Score Comparison")
    col1, col2, col3 = st.columns([1, 0.3, 1])
    
    with col1:
        if better == "A":
            st.metric(
                "📍 Location A",
                f"{score_a:.1f}/100",
                delta=f"+{comp['score_pct_diff']:.1f}%",
                delta_color="off",
            )
        else:
            st.metric("📍 Location A", f"{score_a:.1f}/100")
    
    with col2:
        st.markdown("#### vs")
    
    with col3:
        if better == "B":
            st.metric(
                "📍 Location B",
                f"{score_b:.1f}/100",
                delta=f"+{abs(comp['score_pct_diff']):.1f}%",
                delta_color="off",
            )
        else:
            st.metric("📍 Location B", f"{score_b:.1f}/100")
    
    # Recommendation
    if better == "tied":
        st.info("➖ Both locations have equal livability scores.")
    elif better == "A":
        st.success(
            f"✅ **Location A is {abs(comp['score_pct_diff']):.1f}% better** — {comparison_result['recommendation']}"
        )
    else:
        st.success(
            f"✅ **Location B is {abs(comp['score_pct_diff']):.1f}% better** — {comparison_result['recommendation']}"
        )
    
    # Component-level comparison table
    st.markdown("### 🔍 Component Breakdown")
    comp_diffs = comp["component_diffs"]
    
    if comp_diffs:
        df_data = []
        for comp_name, comp_data in comp_diffs.items():
            comp_label = comp_name.replace("_score", "").replace("_", " ").title()
            df_data.append({
                "Component": comp_label,
                "Location A": comp_data["a"],
                "Location B": comp_data["b"],
                "Difference": comp_data["diff"],
                "Winner": comp_data["better"],
            })
        
        df_compare = pd.DataFrame(df_data)
        
        # Color the winner column
        def highlight_winner(row):
            colors = []
            for val in row:
                if val == "A":
                    colors.append("background-color: #d4edda")  # green
                elif val == "B":
                    colors.append("background-color: #d4edda")  # green
                else:
                    colors.append("")
            return colors
        
        st.table(df_compare)
        
        # Show component details in expander
        with st.expander("📈 Detailed component analysis"):
            for comp_name, comp_data in comp_diffs.items():
                comp_label = comp_name.replace("_score", "").replace("_", " ").title()
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(
                        f"**{comp_label} — Location A:** {comp_data['a']} "
                        f"({'🏆' if comp_data['better'] == 'A' else ''})"
                    )
                with col_b:
                    st.write(
                        f"**{comp_label} — Location B:** {comp_data['b']} "
                        f"({'🏆' if comp_data['better'] == 'B' else ''})"
                    )
                if abs(comp_data["pct"]) > 0:
                    st.caption(f"Difference: {comp_data['pct']:+.1f}%")
    else:
        st.info("No component data available for comparison.")
    
    # Confidence scores
    conf_a = comparison_result["location_a"].get("confidence")
    conf_b = comparison_result["location_b"].get("confidence")
    
    if conf_a is not None or conf_b is not None:
        st.markdown("### 🎯 Prediction Confidence")
        col1, col2 = st.columns(2)
        with col1:
            if conf_a is not None:
                conf_pct = min(100, max(0, conf_a * 100))
                st.metric("Location A Confidence", f"{conf_pct:.1f}%")
        with col2:
            if conf_b is not None:
                conf_pct = min(100, max(0, conf_b * 100))
                st.metric("Location B Confidence", f"{conf_pct:.1f}%")