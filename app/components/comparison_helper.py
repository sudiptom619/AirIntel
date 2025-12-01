# app/components/comparison_helper.py
"""Helper functions for managing comparison mode state and UI logic."""
import streamlit as st
from geopy.geocoders import Nominatim


def initialize_comparison_state():
    """Initialize session state variables for comparison mode."""
    if "comparison_mode_enabled" not in st.session_state:
        st.session_state.comparison_mode_enabled = False
    
    if "location_a_selected" not in st.session_state:
        st.session_state.location_a_selected = None
    
    if "location_b_selected" not in st.session_state:
        st.session_state.location_b_selected = None
    
    if "comparison_result" not in st.session_state:
        st.session_state.comparison_result = None
    
    if "comparison_audio_path" not in st.session_state:
        st.session_state.comparison_audio_path = None


def geocode_location(query: str, user_agent: str = "livability_app"):
    """
    Geocode a location name to (lat, lon, address).
    
    Args:
        query: Location name/address to search
        user_agent: User agent string for Nominatim
    
    Returns:
        Tuple of (lat, lon, address) or (None, None, None) if not found
    """
    if not query or not isinstance(query, str) or len(query.strip()) == 0:
        return None, None, None
    
    try:
        geocoder = Nominatim(user_agent=user_agent)
        loc = geocoder.geocode(query, timeout=10)
        if loc:
            return loc.latitude, loc.longitude, loc.address
        return None, None, None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None, None, None


def render_location_selector(label: str, location_key: str, prefix: str = ""):
    """
    Render a location selector UI in the sidebar.
    
    Args:
        label: Display label (e.g., "Location A")
        location_key: Session state key to store result
        prefix: Prefix for form keys to avoid conflicts
    
    Returns:
        Tuple of (lat, lon) if selected, else (None, None)
    """
    st.sidebar.markdown(f"### 📍 {label}")
    
    input_mode = st.sidebar.radio(
        "Input method:",
        ["Map click", "Search name", "Manual coordinates"],
        key=f"{prefix}_input_mode"
    )
    
    lat, lon = None, None
    
    # SEARCH INPUT MODE
    if input_mode == "Search name":
        q = st.sidebar.text_input(
            "Enter location name (e.g., 'Central Park, NYC')",
            key=f"{prefix}_search_query"
        )
        
        if st.sidebar.button("🔍 Search", key=f"{prefix}_search_btn"):
            with st.spinner(f"Searching for {q}..."):
                lat, lon, addr = geocode_location(q)
            
            if lat is not None and lon is not None:
                st.session_state[location_key] = {
                    "lat": lat,
                    "lon": lon,
                    "address": addr,
                    "source": "search"
                }
                st.sidebar.success(f"✅ Found: {addr}")
            else:
                st.sidebar.error(f"❌ Location '{q}' not found")
    
    # MANUAL INPUT MODE
    elif input_mode == "Manual coordinates":
        lat = st.sidebar.number_input(
            "Latitude",
            value=22.5726,
            format="%.6f",
            key=f"{prefix}_lat"
        )
        lon = st.sidebar.number_input(
            "Longitude",
            value=88.3639,
            format="%.6f",
            key=f"{prefix}_lon"
        )
        
        if st.sidebar.button("✓ Confirm", key=f"{prefix}_confirm_btn"):
            st.session_state[location_key] = {
                "lat": lat,
                "lon": lon,
                "address": f"{lat}, {lon}",
                "source": "manual"
            }
    
    else:  # Map click
        st.sidebar.info("👆 Click on the map below to select a location.")
    
    # Display selected location
    if st.session_state.get(location_key):
        selected = st.session_state[location_key]
        st.sidebar.success(
            f"✅ Selected: {selected['address']}\n"
            f"({selected['lat']:.4f}, {selected['lon']:.4f})"
        )
        return selected["lat"], selected["lon"]
    
    return None, None


def clear_comparison_state():
    """Clear all comparison state variables."""
    st.session_state.location_a_selected = None
    st.session_state.location_b_selected = None
    st.session_state.comparison_result = None
    st.session_state.comparison_audio_path = None


def get_selected_locations():
    """
    Get currently selected locations.
    
    Returns:
        Dict with keys 'a' and 'b', each containing (lat, lon) or None
    """
    loc_a = st.session_state.get("location_a_selected")
    loc_b = st.session_state.get("location_b_selected")
    
    return {
        "a": (loc_a["lat"], loc_a["lon"]) if loc_a else None,
        "b": (loc_b["lat"], loc_b["lon"]) if loc_b else None,
    }


def both_locations_selected():
    """Check if both locations are selected."""
    loc_a = st.session_state.get("location_a_selected")
    loc_b = st.session_state.get("location_b_selected")
    return loc_a is not None and loc_b is not None