# app/streamlit_app.py - WITH WORKING DYNAMIC HEATMAP
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
from models.predict import predict_point
from geopy.geocoders import Nominatim
from streamlit_folium import st_folium
import folium
import geopandas as gpd
import numpy as np
import math

from app.components.location_score_card import render_score_card
from app.components.compare_panel import compare_two, render_comparison_ui
from app.components.comparison_helper import (
    initialize_comparison_state,
    render_location_selector,
    clear_comparison_state,
    both_locations_selected,
)
from app.audio.tts import speak_and_render


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    try:
        rlat1, rlon1, rlat2, rlon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = rlat2 - rlat1
        dlon = rlon2 - rlon1
        a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(min(1, math.sqrt(a)))
        R = 6371.0
        return R * c
    except:
        return float('inf')


@st.cache_data(max_entries=1)
def load_heat_df(path: str = "data/models/predictions.geojson"):
    """Load ALL predictions without downsampling for dynamic filtering."""
    try:
        preds = gpd.read_file(path)
    except Exception as e:
        st.sidebar.error(f"Could not load heatmap data: {e}")
        return None

    if "centroid_lat" in preds.columns and "centroid_lon" in preds.columns:
        df = pd.DataFrame({
            "lat": preds["centroid_lat"].astype(float),
            "lon": preds["centroid_lon"].astype(float),
            "score": preds.get("predicted", preds.get("livability_score", 0)).astype(float),
        })
    else:
        c = preds.geometry.centroid
        df = pd.DataFrame({
            "lat": c.y,
            "lon": c.x,
            "score": preds.get("predicted", preds.get("livability_score", 0)).astype(float)
        })

    df = df.dropna(subset=["score"])
    return df


st.set_page_config(layout="wide", page_title="Pollution Livability")
st.title("🌍 Pollution & Livability Analysis")

# Initialize session state
initialize_comparison_state()
if "selected_location" not in st.session_state:
    st.session_state.selected_location = None

# Sidebar: Analysis mode
st.sidebar.title("📊 Analysis Mode")
analysis_mode = st.sidebar.radio(
    "Choose mode:",
    ["Single Location", "Compare Two Locations"],
    help="Single: Analyze one location. Compare: See side-by-side comparison."
)

# ============================================================================
# MODE 1: SINGLE LOCATION
# ============================================================================
if analysis_mode == "Single Location":
    st.sidebar.title("📍 Select Location")
    input_mode = st.sidebar.radio(
        "Input mode:",
        ["Map click", "Search name", "Manual coordinates"]
    )

    lat = None
    lon = None

    # Search input
    if input_mode == "Search name":
        q = st.sidebar.text_input("Enter location name")
        if st.sidebar.button("Search"):
            geocoder = Nominatim(user_agent="livability_app")
            loc = geocoder.geocode(q)
            if loc:
                lat, lon = loc.latitude, loc.longitude
                st.session_state.selected_location = {"lat": lat, "lon": lon}
                st.sidebar.success(f"✅ Found: {loc.address}")
            else:
                st.sidebar.error("❌ Location not found")

    # Manual input
    elif input_mode == "Manual coordinates":
        lat = st.sidebar.number_input("Latitude", value=22.5726, format="%.6f")
        lon = st.sidebar.number_input("Longitude", value=88.3639, format="%.6f")
    else:
        st.sidebar.info("👆 Click on the map to select a location")

    # Heatmap controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🗺️ Map Controls")
    show_heat = st.sidebar.checkbox("Show heat layer", value=True)
    heat_opacity = st.sidebar.slider("Heat opacity", 0.0, 1.0, 0.6)
    heat_radius = st.sidebar.slider("Heat point radius", 2, 12, 6)
    
    localize_heat = st.sidebar.checkbox("🎯 Center heatmap on selection", value=True, 
                                        help="When checked, heatmap will show data around your selected location")
    local_radius_km = st.sidebar.slider("Show data within (km)", 0.5, 20.0, 5.0, step=0.5)
    auto_zoom = st.sidebar.checkbox("Auto-zoom to selection", value=True)

    # Load heatmap data
    heat_df = load_heat_df()

    # Determine effective location
    effective_lat = None
    effective_lon = None
    if lat and lon:
        effective_lat, effective_lon = float(lat), float(lon)
    elif st.session_state.selected_location:
        try:
            effective_lat = float(st.session_state.selected_location["lat"])
            effective_lon = float(st.session_state.selected_location["lon"])
        except:
            pass

    # Set map center
    DEFAULT_CENTER = [22.5726, 88.3639]
    map_center = DEFAULT_CENTER
    zoom_start = 11
    
    if effective_lat and effective_lon:
        map_center = [effective_lat, effective_lon]
        if auto_zoom:
            zoom_start = 13

    # Filter heatmap based on selection
    display_df = heat_df
    if localize_heat and effective_lat and effective_lon and heat_df is not None:
        # Filter to radius around selected location
        distances = heat_df.apply(
            lambda row: haversine_km(effective_lat, effective_lon, row['lat'], row['lon']),
            axis=1
        )
        display_df = heat_df[distances <= local_radius_km]
        
        if len(display_df) == 0:
            st.sidebar.warning(f"⚠️ No data within {local_radius_km}km - showing nearest 500 points")
            # Show nearest 500 points
            heat_df['distance'] = distances
            display_df = heat_df.nsmallest(500, 'distance').drop(columns=['distance'])
        else:
            st.sidebar.success(f"✅ Showing {len(display_df):,} locations within {local_radius_km}km")

    # Build map
    m = folium.Map(location=map_center, zoom_start=zoom_start)

    # Add heatmap layer
    if show_heat and display_df is not None and not display_df.empty:
        try:
            from branca.colormap import linear
            vmin = float(display_df["score"].min())
            vmax = float(display_df["score"].max())
            cmap = linear.YlOrRd_09.scale(vmin, vmax)
            cmap.caption = f"Livability Score ({len(display_df):,} locations)"

            for _, row in display_df.iterrows():
                try:
                    score = float(row["score"])
                    color = cmap(score)
                    folium.CircleMarker(
                        location=[row["lat"], row["lon"]],
                        radius=heat_radius,
                        color=color,
                        fill=True,
                        fill_opacity=heat_opacity,
                        stroke=False,
                        popup=f"Score: {score:.1f}",
                        tooltip=f"Score: {score:.1f}"
                    ).add_to(m)
                except:
                    continue

            m.add_child(cmap)
        except Exception as e:
            st.sidebar.error(f"Could not render heatmap: {e}")

    # Add selected location marker
    if effective_lat and effective_lon:
        folium.CircleMarker(
            location=[effective_lat, effective_lon],
            radius=12,
            color="#0000ff",
            fill=True,
            fill_color="#0000ff",
            fill_opacity=1.0,
            weight=3,
            popup="📍 Selected Location",
            tooltip="📍 Your Selection"
        ).add_to(m)
        
        # Add radius circle if localizing
        if localize_heat:
            folium.Circle(
                location=[effective_lat, effective_lon],
                radius=local_radius_km * 1000,  # Convert km to meters
                color="gray",
                fill=False,
                weight=2,
                opacity=0.4,
                popup=f"Search radius: {local_radius_km}km"
            ).add_to(m)

    map_data = st_folium(m, width=900, height=500)

    # Handle map clicks
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]
        st.session_state.selected_location = {"lat": clicked_lat, "lon": clicked_lon}
        st.rerun()

    # Show selected location
    if effective_lat and effective_lon:
        st.sidebar.markdown(f"### 📍 Selected: **{effective_lat:.5f}, {effective_lon:.5f}**")
        
        if st.sidebar.button("🔍 Analyze this location", use_container_width=True):
            with st.spinner("Computing livability score..."):
                result = predict_point(effective_lat, effective_lon)

            st.markdown("---")
            st.markdown("## 📌 Analysis Results")
            audio_file = render_score_card(result)

            if audio_file:
                st.markdown("### 🔊 Hear the Analysis")
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.audio(audio_file, format="audio/mp3")
                with col2:
                    st.info("📢 Click play to hear audio summary")
                
                with st.expander("📄 View transcript"):
                    score = result.get("score")
                    comps = result.get("components", {})
                    transcript = f"This location has a livability score of {score:.0f} out of 100. "
                    for key, val in comps.items():
                        if val:
                            comp_name = key.replace("_score", "").replace("_", " ").title()
                            transcript += f"{comp_name} score is {val:.0f}. "
                    st.write(transcript)
    else:
        st.sidebar.info("👆 Select a location to analyze")

# ============================================================================
# MODE 2: COMPARE TWO LOCATIONS
# ============================================================================
else:
    st.sidebar.markdown("---")
    st.sidebar.info("ℹ️ Select two locations to compare")
    
    # Location A selector
    lat_a, lon_a = render_location_selector("Location A", "location_a_selected", "locA")
    st.sidebar.markdown("---")
    
    # Location B selector
    lat_b, lon_b = render_location_selector("Location B", "location_b_selected", "locB")
    st.sidebar.markdown("---")

    # Map showing both locations
    if lat_a and lon_a:
        avg_lat = lat_a if not (lat_b and lon_b) else (lat_a + lat_b) / 2
        avg_lon = lon_a if not (lat_b and lon_b) else (lon_a + lon_b) / 2
        
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)
        
        # Location A marker (blue)
        folium.CircleMarker(
            location=[lat_a, lon_a],
            radius=12,
            color="#0000ff",
            fill=True,
            fill_color="#0000ff",
            fill_opacity=0.9,
            popup="📍 Location A",
            tooltip="📍 Location A",
            weight=3
        ).add_to(m)
        
        # Location B marker (red)
        if lat_b and lon_b:
            folium.CircleMarker(
                location=[lat_b, lon_b],
                radius=12,
                color="#ff0000",
                fill=True,
                fill_color="#ff0000",
                fill_opacity=0.9,
                popup="📍 Location B",
                tooltip="📍 Location B",
                weight=3
            ).add_to(m)
            
            # Line connecting them
            folium.PolyLine(
                locations=[[lat_a, lon_a], [lat_b, lon_b]],
                color="gray",
                weight=2,
                opacity=0.6
            ).add_to(m)
        
        st_folium(m, width=900, height=500)
    else:
        st.info("👆 Select Location A to see the map")

    # Compare button
    if both_locations_selected():
        if st.sidebar.button("🔄 Compare Locations", use_container_width=True):
            with st.spinner("Analyzing both locations..."):
                result = compare_two(lat_a, lon_a, lat_b, lon_b)
                st.session_state.comparison_result = result
                
                if not result.get("error"):
                    audio_file = speak_and_render(result, use_short=True)
                    st.session_state.comparison_audio_path = audio_file
    else:
        st.sidebar.warning("⚠️ Select both locations")

    # Display comparison results
    if st.session_state.comparison_result:
        st.markdown("---")
        st.markdown("## 🔄 Comparison Results")
        render_comparison_ui(st.session_state.comparison_result)
        
        # Audio section
        st.markdown("---")
        st.markdown("### 🔊 Audio Narration")
        
        col_audio, col_info = st.columns([1, 2])
        with col_audio:
            if st.session_state.comparison_audio_path:
                st.audio(st.session_state.comparison_audio_path, format="audio/mp3")
        with col_info:
            st.info("📢 Click play to hear the comparison")
        
        # Transcript
        with st.expander("📄 View transcript"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Short Summary:**")
                st.write(st.session_state.comparison_result.get("narration_short", "N/A"))
            with col2:
                st.markdown("**Detailed Analysis:**")
                st.write(st.session_state.comparison_result.get("narration_detailed", "N/A"))
        
        # Export
        with st.expander("📊 Export Data"):
            import io
            comp = st.session_state.comparison_result.get("comparison", {})
            export_data = {
                "Location A Score": comp.get("score_a"),
                "Location B Score": comp.get("score_b"),
                "Difference": comp.get("score_diff"),
                "% Difference": f"{comp.get('score_pct_diff'):.1f}%",
                "Winner": comp.get("better_location"),
            }
            df_export = pd.DataFrame([export_data]).T
            df_export.columns = ["Value"]
            st.table(df_export)
            
            csv_buffer = io.StringIO()
            df_export.to_csv(csv_buffer)
            st.download_button(
                "📥 Download CSV",
                csv_buffer.getvalue(),
                "comparison.csv",
                "text/csv"
            )
        
        # New comparison
        if st.button("➕ Start New Comparison", use_container_width=True):
            clear_comparison_state()
            st.rerun()