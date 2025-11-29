# app/streamlit_app.py
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

from app.components.location_score_card import render_score_card
import math


def haversine_km(lat1, lon1, lat2, lon2):
    """Return distance in kilometers between two lat/lon points (Haversine)."""
    # convert degrees to radians
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(min(1, math.sqrt(a)))
    R = 6371.0
    return R * c


@st.cache_data(max_entries=2)
def load_heat_df(path: str = "data/models/predictions.geojson", max_points: int = 3000):
    """Load predictions.geojson and return a small DataFrame for pydeck.

    - Caches result to avoid re-reading the large GeoJSON on every rerun.
    - Downsamples to `max_points` to keep pydeck responsive.
    """
    try:
        preds = gpd.read_file(path)
    except Exception:
        return None

    # prefer explicit centroid fields if present
    if "centroid_lat" in preds.columns and "centroid_lon" in preds.columns:
        df = pd.DataFrame({
            "lat": preds["centroid_lat"].astype(float),
            "lon": preds["centroid_lon"].astype(float),
            "score": preds.get("predicted", preds.get("livability_score", 0)).astype(float),
        })
    else:
        c = preds.geometry.centroid
        df = pd.DataFrame({"lat": c.y, "lon": c.x, "score": preds.get("predicted", preds.get("livability_score", 0)).astype(float)})

    # drop NA weights
    df = df.dropna(subset=["score"])

    # downsample if too large
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42).reset_index(drop=True)

    return df

st.set_page_config(layout="wide", page_title="Pollution Livability")

st.title("🌍 Pollution & Livability Analysis")

# ------------------------------
# SESSION STATE INITIALIZATION
# ------------------------------
if "selected_location" not in st.session_state:
    st.session_state.selected_location = None


# ------------------------------
# SIDEBAR — INPUT MODE
# ------------------------------
st.sidebar.title("📍 Select Location")
input_mode = st.sidebar.radio(
    "Input mode:",
    ["Map click", "Search name", "Manual coordinates"]
)

lat = None
lon = None


# ------------------------------
# SEARCH INPUT MODE
# ------------------------------
if input_mode == "Search name":
    q = st.sidebar.text_input("Enter location / building / college name")

    if st.sidebar.button("Search"):
        geocoder = Nominatim(user_agent="livability_app")
        loc = geocoder.geocode(q)

        if loc:
            lat, lon = loc.latitude, loc.longitude
            # set the single selected location (no pin list)
            st.session_state.selected_location = {"lat": lat, "lon": lon}
            st.sidebar.success(f"Found: {loc.address}")
        else:
            st.sidebar.error("Location not found.")


# ------------------------------
# MANUAL INPUT MODE
# ------------------------------
elif input_mode == "Manual coordinates":
    lat = st.sidebar.number_input("Latitude", value=22.5726, format="%.6f")
    lon = st.sidebar.number_input("Longitude", value=88.3639, format="%.6f")


else:
    st.sidebar.info("Click on the map below to select a location.")


# ------------------------------
# Heatmap & display controls
# ------------------------------
st.sidebar.markdown("---")
show_heat = st.sidebar.checkbox("Show heat layer", value=True)
heat_opacity = st.sidebar.slider("Heat opacity", min_value=0.0, max_value=1.0, value=0.5)
heat_radius = st.sidebar.slider("Heat point radius", min_value=2, max_value=12, value=6)

# localize heat options
localize_heat = st.sidebar.checkbox("Localize heat to selection", value=False)
local_radius_km = st.sidebar.slider("Local radius (km)", min_value=0.5, max_value=20.0, value=2.0, step=0.5)
auto_zoom = st.sidebar.checkbox("Auto-zoom to selection", value=True)


# ------------------------------
# MAP RENDERING
# ------------------------------
DEFAULT_CENTER = [22.5726, 88.3639]

# Load heatmap-ready dataframe via cached loader (faster on reruns)
heat_df = load_heat_df()

# determine the location to use for filtering/analysis (manual input takes priority)
selection_lat = None
selection_lon = None
if lat and lon:
    selection_lat, selection_lon = float(lat), float(lon)
elif st.session_state.selected_location:
    try:
        selection_lat = float(st.session_state.selected_location["lat"])
        selection_lon = float(st.session_state.selected_location["lon"])
    except Exception:
        selection_lat = selection_lon = None


# build a single folium map and optionally draw the heat layer (colored circle markers)
# center the map on the selected location when available so clicks feel responsive
map_center = DEFAULT_CENTER
zoom_start = 12
if st.session_state.selected_location:
    s = st.session_state.selected_location
    try:
        map_center = [float(s["lat"]), float(s["lon"])]
        if auto_zoom:
            zoom_start = 14
    except Exception:
        map_center = DEFAULT_CENTER

m = folium.Map(location=map_center, zoom_start=zoom_start)

# By default display the full (cached & downsampled) heat_df. If the user
# asked to localize the heat to the selected location, filter down to points
# within `local_radius_km` of that location. This keeps the map focused and
# reduces client rendering work.
display_df = heat_df
if show_heat and heat_df is not None and not heat_df.empty:
    try:
        from branca.colormap import linear
        # if localization is requested and we have a selected location, filter
        if localize_heat and selection_lat is not None and selection_lon is not None:
            try:
                mask = heat_df.apply(
                    lambda r: haversine_km(selection_lat, selection_lon, float(r["lat"]), float(r["lon"])) <= float(local_radius_km),
                    axis=1,
                )
                display_df = heat_df[mask]
                if display_df.empty:
                    st.sidebar.info("No heat points found inside the chosen radius — showing full heatmap.")
                    display_df = heat_df
            except Exception:
                display_df = heat_df

        vmin = float(display_df["score"].min())
        vmax = float(display_df["score"].max())
        cmap = linear.YlOrRd_09.scale(vmin, vmax)
        cmap.caption = "Predicted livability"

        for _, row in display_df.iterrows():
            try:
                score = float(row["score"])
            except Exception:
                continue
            color = cmap(score)
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=heat_radius,
                color=color,
                fill=True,
                fill_opacity=heat_opacity,
                stroke=False,
            ).add_to(m)

        m.add_child(cmap)
    except Exception:
        # if anything goes wrong, fall back to an informative message in the sidebar
        st.sidebar.warning("Could not render heat layer — see logs.")

# show the single selected location as a blue marker (if any)
if st.session_state.selected_location:
    s = st.session_state.selected_location
    folium.CircleMarker(
        location=[s["lat"], s["lon"]],
        radius=10,
        color="#0000ff",
        fill=True,
        fill_color="#0000ff",
        fill_opacity=1.0,
    ).add_to(m)

map_data = st_folium(m, width=850, height=450)


# ------------------------------
# HANDLE MAP CLICK EVENT
# ------------------------------
if map_data and map_data.get("last_clicked"):
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]

    # update the single selected location
    st.session_state.selected_location = {"lat": clicked_lat, "lon": clicked_lon}


# ------------------------------
# FINAL EFFECTIVE LOCATION TO ANALYZE
# ------------------------------
effective_lat = None
effective_lon = None

# priority: explicit manual entry -> selected_location
if lat and lon:
    effective_lat, effective_lon = lat, lon
elif st.session_state.selected_location:
    effective_lat = st.session_state.selected_location["lat"]
    effective_lon = st.session_state.selected_location["lon"]


# ------------------------------
# DISPLAY SELECTED LOCATION
# ------------------------------
if effective_lat and effective_lon:
    st.sidebar.markdown(
        f"### 📍 Selected: **{effective_lat:.5f}, {effective_lon:.5f}**"
    )
else:
    st.sidebar.info("No location selected.")


# ------------------------------
# ANALYZE BUTTON
# ------------------------------
if effective_lat and effective_lon:
    if st.sidebar.button("🔍 Analyze this location"):
        with st.spinner("Computing livability score..."):
            result = predict_point(effective_lat, effective_lon)

        # render using reusable component
        audio_file = render_score_card(result)

        # if the card didn't return audio but TTS helper exists we already attempted
        # playback inside the component. Nothing further required here.
