# app/components/legends.py
"""Map legends for the Streamlit UI."""
import streamlit as st


def render_livability_legend():
    """Render a legend explaining the livability score colors."""
    st.markdown("""
    <style>
    .legend-box {
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        margin: 5px 0;
    }
    .legend-item {
        display: flex;
        align-items: center;
        margin: 5px 0;
    }
    .legend-color {
        width: 20px;
        height: 20px;
        border-radius: 3px;
        margin-right: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="legend-box">
        <strong>🎨 Livability Score Legend</strong>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #006837;"></div>
            <span>80-100: Excellent</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #78c679;"></div>
            <span>60-80: Good</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #ffeda0;"></div>
            <span>40-60: Moderate</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #feb24c;"></div>
            <span>20-40: Poor</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background-color: #bd0026;"></div>
            <span>0-20: Very Poor</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_component_legend(component: str):
    """Render a legend for a specific component.
    
    Args:
        component: One of 'pollution', 'traffic', 'green', 'industry', 'population', 'flood'
    """
    legends = {
        "pollution": {
            "title": "🌫️ Air Quality",
            "description": "Based on PM2.5 concentration (μg/m³)",
            "items": [
                ("#006837", "0-12: Good"),
                ("#78c679", "12-35: Moderate"),
                ("#feb24c", "35-55: Unhealthy for Sensitive"),
                ("#bd0026", "55+: Unhealthy"),
            ]
        },
        "traffic": {
            "title": "🚗 Traffic Density",
            "description": "Road length per area (m/m²)",
            "items": [
                ("#ffffcc", "Low density"),
                ("#c2e699", "Moderate"),
                ("#78c679", "High"),
                ("#31a354", "Very High"),
            ]
        },
        "green": {
            "title": "🌳 Vegetation (NDVI)",
            "description": "Normalized Difference Vegetation Index",
            "items": [
                ("#8c510a", "-1 to 0: Barren/Water"),
                ("#d8b365", "0-0.2: Sparse"),
                ("#5ab4ac", "0.2-0.5: Moderate"),
                ("#01665e", "0.5-1: Dense"),
            ]
        },
        "industry": {
            "title": "🏭 Industrial Proximity",
            "description": "Distance to nearest industrial area",
            "items": [
                ("#d73027", "< 500m: Very Close"),
                ("#fc8d59", "500-1000m: Close"),
                ("#fee08b", "1-2km: Moderate"),
                ("#91cf60", "2-5km: Far"),
                ("#1a9850", "> 5km: Very Far"),
            ]
        },
        "population": {
            "title": "👥 Population Density",
            "description": "People per km²",
            "items": [
                ("#edf8fb", "< 1000: Low"),
                ("#b3cde3", "1000-5000: Moderate"),
                ("#8c96c6", "5000-10000: High"),
                ("#810f7c", "> 10000: Very High"),
            ]
        },
        "flood": {
            "title": "🌊 Flood Risk",
            "description": "Based on elevation and terrain",
            "items": [
                ("#deebf7", "Low Risk"),
                ("#9ecae1", "Moderate Risk"),
                ("#3182bd", "High Risk"),
            ]
        }
    }
    
    legend = legends.get(component)
    if not legend:
        return
    
    st.markdown(f"""
    <div class="legend-box">
        <strong>{legend['title']}</strong>
        <p style="font-size: 0.8em; color: #666;">{legend['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    for color, label in legend["items"]:
        st.markdown(f"""
        <div class="legend-item">
            <div class="legend-color" style="background-color: {color};"></div>
            <span>{label}</span>
        </div>
        """, unsafe_allow_html=True)


def render_compact_legend():
    """Render a compact single-line legend for the sidebar."""
    st.markdown("""
    <div style="font-size: 0.8em; padding: 5px; background: #f0f0f0; border-radius: 3px;">
        🟢 Good (80+) | 🟡 Moderate (40-80) | 🔴 Poor (<40)
    </div>
    """, unsafe_allow_html=True)
